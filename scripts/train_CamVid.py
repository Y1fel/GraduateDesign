import math
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.CamVid import CamVidFolderDataset
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv
from src.utils.Id2Mask import color_mask_to_id
from src.viz.visualizer import save_predictions_triplet
from src.losses.composite import CrossEntropyDiceLoss


@dataclass
class TrainConfig:
    data_root: Path = PROJECT_ROOT / "data" / "archive" / "CamVid"

    num_classes: int = 11
    ignore_index: int = 255

    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    lr_0: float = 5e-4
    weight_decay: float = 1e-4

    ce_weight: float = 1.0
    dice_weight: float = 0.5
    label_smoothing: float = 0.0

    output_stride: int = 8
    backbone_pretrained: bool = True
    head_norm: str = "bn"
    use_mid_level_fusion: bool = True

    resize_h: int = 960
    resize_w: int = 1280
    crop_h: int = 720
    crop_w: int = 960
    train_multi_scale_min: float = 0.5
    train_multi_scale_max: float = 2.0
    hflip_prob: float = 0.5

    photo_aug_prob: float = 0.35
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.08
    saturation_jitter: float = 0.06
    gamma_min: float = 0.95
    gamma_max: float = 1.05
    photo_op_prob: float = 0.50
    blur_prob: float = 0.03
    blur_radius_min: float = 0.1
    blur_radius_max: float = 0.6
    jpeg_prob: float = 0.03
    jpeg_quality_min: int = 90
    jpeg_quality_max: int = 98
    photo_aug_warmup_epochs: int = 20

    auto_contrast: bool = True
    auto_contrast_cutoff: float = 1.0

    ignore_0001tp_prefix: bool = False

    ce_variant: str = "ce"  # ce | focal | ohem
    use_class_balanced_ce: bool = False
    loss_preset: str = "baseline"  # baseline | cbce | focal | ohem
    cb_beta: float = 0.999
    focal_gamma: float = 2.0
    ohem_min_kept: int = 100000
    ohem_thresh: float = 0.7
    boundary_weight: float = 1.5

    save_vis_every: int = 50
    save_vis_max_items: int = 8

    outputs_root: Path = PROJECT_ROOT / "outputs"
    seed: int = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on CamVid")
    parser.add_argument(
        "--loss_preset",
        type=str,
        default=None,
        choices=["baseline", "cbce", "focal", "ohem"],
        help="Loss preset: baseline(plain CE), cbce(class-balanced CE), focal, or ohem.",
    )
    parser.add_argument(
        "--ignore_0001tp_prefix",
        action="store_true",
        help="Ignore image files whose names start with '0001TP_' (very dark captures).",
    )
    return parser.parse_args()


def apply_loss_preset(cfg: TrainConfig, loss_preset: str) -> None:
    cfg.loss_preset = loss_preset

    if loss_preset == "baseline":
        cfg.ce_variant = "ce"
        cfg.use_class_balanced_ce = False
    elif loss_preset == "cbce":
        cfg.ce_variant = "ce"
        cfg.use_class_balanced_ce = True
    elif loss_preset == "focal":
        cfg.ce_variant = "focal"
        cfg.use_class_balanced_ce = False
    elif loss_preset == "ohem":
        cfg.ce_variant = "ohem"
        cfg.use_class_balanced_ce = False
    else:
        raise ValueError(f"Unsupported loss_preset: {loss_preset}")


def compute_class_pixel_distribution(
    masks_dir: Path,
    color2id,
    label_lut: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    mask_paths = sorted([p for p in masks_dir.iterdir() if p.is_file()])
    if not mask_paths:
        raise RuntimeError(f"No masks found for pixel statistics in {masks_dir}")

    counts = np.zeros((num_classes,), dtype=np.int64)
    for p in mask_paths:
        mask_rgb = np.array(torchvision_safe_open_rgb(p), dtype=np.uint8)
        mask_old = color_mask_to_id(mask_rgb, color2id, ignore_index)
        mapped = label_lut[mask_old]
        valid = mapped != ignore_index
        if np.any(valid):
            binc = np.bincount(mapped[valid], minlength=num_classes)
            counts += binc[:num_classes]
    return counts


def torchvision_safe_open_rgb(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def class_balanced_weights_from_counts(counts: np.ndarray, beta: float, eps: float = 1e-8) -> np.ndarray:
    counts = counts.astype(np.float64)
    beta = float(beta)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, eps)
    weights[counts <= 0] = 0.0

    nz = weights > 0
    if np.any(nz):
        weights[nz] = weights[nz] * (nz.sum() / np.sum(weights[nz]))
    return weights.astype(np.float32)


def print_small_object_metrics(metric_name: str, values: Sequence[float], names: Sequence[str], indices: Sequence[int]) -> None:
    msg = []
    for cls_name, idx in zip(names, indices):
        v = values[idx]
        if np.isnan(v):
            msg.append(f"{cls_name}=nan")
        else:
            msg.append(f"{cls_name}={v:.4f}")
    print(f"[VAL-small] {metric_name}: " + " | ".join(msg))


def build_merge_lut(groups_11, ignore_index: int = 255) -> np.ndarray:
    if len(groups_11) != 11:
        raise ValueError(f"groups_11 must have length=11, got {len(groups_11)}")

    lut = np.full((256,), fill_value=ignore_index, dtype=np.uint8)
    used = set()

    for new_id, group in enumerate(groups_11):
        for old_id in group:
            old_id = int(old_id)
            if old_id in used:
                raise ValueError(f"old_id {old_id} appears in multiple groups")
            used.add(old_id)
            lut[old_id] = np.uint8(new_id)

    lut[30] = np.uint8(ignore_index)

    missing = [i for i in range(32) if lut[i] == ignore_index]
    if missing:
        print(f"[WARN] these old ids are not assigned (will be ignored): {missing}")
    return lut


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler: GradScaler | None,
    epoch: int,
    total_iters: int,
    base_lr: float,
    power: float = 0.9,
) -> float:
    model.train()
    total_loss, n = 0.0, 0

    use_amp = (device.type == "cuda") and (scaler is not None)

    for it, (imgs, masks, _names) in enumerate(loader):
        global_step = (epoch - 1) * len(loader) + it
        lr = base_lr * (1 - global_step / total_iters) ** power
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.inference_mode()
def save_vis_using_best_ckpt(
    model,
    val_loader,
    device,
    out_dir: Path,
    id2color,
    ignore_index: int,
    epoch: int,
    max_items: int,
    best_ckpt_path: Path,
) -> None:
    cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

    model.eval()
    save_predictions_triplet(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=out_dir,
        id2color=id2color,
        ignore_index=ignore_index,
        epoch=epoch,
        max_items=max_items,
    )

    model.load_state_dict(cur_state, strict=True)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    apply_loss_preset(cfg, args.loss_preset or cfg.loss_preset)
    if args.ignore_0001tp_prefix:
        cfg.ignore_0001tp_prefix = True
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=(device.type == "cuda"))
    print(f"[INFO] device = {device}")

    # 32类颜色表/映射仍用于 RGB->old_id 解码
    csv_path = cfg.data_root / "class_dict.csv"
    color2id, id2color, _id2name = load_class_dict_csv(csv_path)

    # 11类分组（old_id -> new_id）
    GROUPS_11 = [
        [21],                 # 0 Sky
        [4, 31, 1, 3, 28],     # 1 Building: Building, Wall, Archway, Bridge, Tunnel
        [8, 23],              # 2 Pole: Column_Pole + TrafficCone
        [17, 10, 11],         # 3 Road: Road + LaneMkgsDriv + LaneMkgsNonDriv
        [19, 18, 15],         # 4 Pavement: Sidewalk + RoadShoulder + ParkingBlock
        [26, 29],             # 5 Tree: Tree + VegetationMisc
        [20, 24, 12],         # 6 SignSymbol: SignSymbol + TrafficLight + Misc_Text
        [9],                  # 7 Fence
        [5, 22, 27, 25, 14, 13],  # 8 Car: Car + SUVPickupTruck + Truck_Bus + Train + OtherMoving + MotorcycleScooter
        [16, 7, 0, 6],         # 9 Pedestrian: Pedestrian + Child + Animal + CartLuggagePram
        [2],                  # 10 Bicyclist
    ]

    label_lut = build_merge_lut(GROUPS_11, ignore_index=cfg.ignore_index)
    label_lut[30] = np.uint8(cfg.ignore_index)  # Void ignore

    # 11类可视化颜色（代表色）
    rep_old_ids_11 = [21, 4, 8, 17, 19, 26, 20, 9, 5, 16, 2]
    id2color_11 = [id2color[i] for i in rep_old_ids_11]

    # outputs
    out = OutputManager(cfg.outputs_root, exp_name="camvid_deeplabv3plus")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")

    class_names_11 = [
        "Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
    ]

    train_class_pixel_counts = compute_class_pixel_distribution(
        masks_dir=cfg.data_root / "train_labels",
        color2id=color2id,
        label_lut=label_lut,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )
    pixel_ratio = train_class_pixel_counts / np.maximum(train_class_pixel_counts.sum(), 1)
    print("[DATA] train pixel ratios:")
    for i, name in enumerate(class_names_11):
        print(f"  - {name:<10} count={int(train_class_pixel_counts[i]):>10d} ratio={pixel_ratio[i] * 100:6.3f}%")

    for idx in [2, 9, 10]:
        print(f"[TAIL-CHECK] {class_names_11[idx]} ratio={pixel_ratio[idx] * 100:.3f}%")

    print(f"[LOSS-PRESET] selected={cfg.loss_preset}")
    if cfg.loss_preset == "baseline":
        print("[LOSS-PRESET] baseline enforces ce_variant=ce and class_weights=None.")
    elif cfg.loss_preset == "cbce":
        print("[LOSS-PRESET] cbce enabled: use only after validating long-tail gains.")

    class_weights_t = None
    if cfg.use_class_balanced_ce:
        cb_w = class_balanced_weights_from_counts(train_class_pixel_counts, beta=cfg.cb_beta)
        class_weights_t = torch.tensor(cb_w, dtype=torch.float32, device=device)
        print("[LOSS] class-balanced CE weights:", np.array2string(cb_w, precision=4, separator=", "))

    class_weights_state = "enabled" if class_weights_t is not None else "disabled(None)"
    print(
        "[LOSS] effective setup: "
        f"preset={cfg.loss_preset} | ce_variant={cfg.ce_variant} | "
        f"class_balanced_ce={cfg.use_class_balanced_ce} | class_weights={class_weights_state} | "
        f"ce_weight={cfg.ce_weight} | dice_weight={cfg.dice_weight} | boundary_weight={cfg.boundary_weight}"
    )

    # datasets
    train_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="train",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        hflip_prob=cfg.hflip_prob,
        ignore_index=cfg.ignore_index,
        training=True,
        label_lut=label_lut,
        photo_aug_prob=cfg.photo_aug_prob,
        brightness_jitter=cfg.brightness_jitter,
        contrast_jitter=cfg.contrast_jitter,
        saturation_jitter=cfg.saturation_jitter,
        gamma_range=(cfg.gamma_min, cfg.gamma_max),
        photo_op_prob=cfg.photo_op_prob,
        blur_prob=cfg.blur_prob,
        blur_radius_range=(cfg.blur_radius_min, cfg.blur_radius_max),
        jpeg_prob=cfg.jpeg_prob,
        jpeg_quality_range=(cfg.jpeg_quality_min, cfg.jpeg_quality_max),
        multi_scale_range=(cfg.train_multi_scale_min, cfg.train_multi_scale_max),
        random_crop_size=(cfg.crop_w, cfg.crop_h),
        auto_contrast=cfg.auto_contrast,
        auto_contrast_cutoff=cfg.auto_contrast_cutoff,
        ignore_filename_prefixes=(("0001TP_",) if cfg.ignore_0001tp_prefix else ()),
    )
    val_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="val",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        hflip_prob=0.0,
        ignore_index=cfg.ignore_index,
        training=False,
        label_lut=label_lut,
        auto_contrast=cfg.auto_contrast,
        auto_contrast_cutoff=cfg.auto_contrast_cutoff,
        ignore_filename_prefixes=(("0001TP_",) if cfg.ignore_0001tp_prefix else ()),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # model
    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=cfg.backbone_pretrained,
        output_stride=cfg.output_stride,
        head_norm=cfg.head_norm,
        use_mid_level_fusion=cfg.use_mid_level_fusion,
    ).to(device)

    criterion = CrossEntropyDiceLoss(
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ce_weight=cfg.ce_weight,
        dice_weight=cfg.dice_weight,
        label_smoothing=cfg.label_smoothing,
        dice_include_background=True,
        ce_variant=cfg.ce_variant,
        class_weights=class_weights_t,
        focal_gamma=cfg.focal_gamma,
        ohem_min_kept=cfg.ohem_min_kept,
        ohem_thresh=cfg.ohem_thresh,
        boundary_weight=cfg.boundary_weight,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr_0,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    best_miou = -1.0

    for epoch in range(1, cfg.epochs + 1):
        total_iters = cfg.epochs * len(train_loader)
        t0 = time.time()

        if hasattr(train_ds, "set_photo_aug_scale"):
            if cfg.photo_aug_warmup_epochs > 0:
                aug_scale = min(1.0, epoch / float(cfg.photo_aug_warmup_epochs))
            else:
                aug_scale = 1.0
            train_ds.set_photo_aug_scale(aug_scale)
            print(f"[AUG] photo_aug_prob={train_ds.photo_aug_prob_current:.3f} (scale={aug_scale:.2f})")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            epoch=epoch,
            total_iters=total_iters,
            base_lr=cfg.lr_0,
        )
        val_metrics = compute_segmentation_metrics(model, val_loader, device, cfg.num_classes, cfg.ignore_index)
        val_miou = float(val_metrics["miou"])

        dt = time.time() - t0
        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] loss={train_loss:.4f}  val_mIoU={val_miou:.4f} "
            f" val_BF1={val_metrics['boundary_fscore']:.4f} val_TrimapIoU={val_metrics['trimap_iou']:.4f}  time={dt:.1f}s"
        )
        small_indices = [2, 9, 10]  # Pole, Pedestrian(Person+Rider), Bicyclist
        print_small_object_metrics("IoU", val_metrics["iou_per_class"], class_names_11, small_indices)
        print_small_object_metrics("Recall", val_metrics["recall_per_class"], class_names_11, small_indices)
        if cfg.use_class_balanced_ce:
            print_small_object_metrics("Precision", val_metrics["precision_per_class"], class_names_11, small_indices)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(epoch, train_loss, val_miou, dt)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
        }

        if epoch % 10 == 0:
            torch.save(ckpt, out.ckpt_dir / f"epoch_{epoch:03d}.pth")

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            ckpt["best_miou"] = best_miou
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU = {best_miou:.4f} -> saved best.pth")

        if epoch % cfg.save_vis_every == 0:
            print(f"[INFO] Saving visualizations (best.pth) at epoch {epoch} ...")
            save_vis_using_best_ckpt(
                model=model,
                val_loader=val_loader,
                device=device,
                out_dir=out.vis_dir,
                id2color=id2color_11,
                ignore_index=cfg.ignore_index,
                epoch=epoch,
                max_items=cfg.save_vis_max_items,
                best_ckpt_path=out.ckpt_dir / "best.pth",
            )

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"... lr={cur_lr:.6f}")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
