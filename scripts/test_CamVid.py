import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.CamVid import CamVidFolderDataset
from src.eval.mIoU import compute_miou
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv, id_mask_to_color
from src.utils.remap import assert_camvid_key_old_ids, build_camvid_11_groups_from_names
from src.viz.visualizer import save_predictions_triplet


CLASS_NAMES_11 = [
    "Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
]


@dataclass
class TestConfig:
    data_root: Path
    ckpt_path: Path
    out_dir: Path

    num_classes: int = 11
    ignore_index: int = 255

    output_stride: int = 8
    head_norm: str = "bn"

    resize_h: int = 720
    resize_w: int = 960

    batch_size: int = 8
    num_workers: int = 8

    save_triplet_max: int = 100

    eval_auto_contrast_enable: bool = False
    eval_auto_contrast_cutoff: float = 1.0
    eval_low_light_preprocess_enable: bool = False
    low_light_gamma: float = 1.0
    low_light_brightness_gain: float = 1.0

    drift_upper_ratio: float = 2.0
    drift_lower_ratio: float = 0.5
    collapse_warn_ratio: float = 0.60
    confusion_topk: int = 10


def resolve_ckpt_path(ckpt: Path) -> Path:
    if ckpt.is_file():
        return ckpt
    p1 = ckpt / "checkpoints" / "best.pth"
    if p1.exists():
        return p1
    p2 = ckpt / "best.pth"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find checkpoint under: {ckpt}")


def load_model(cfg: TestConfig, device: torch.device) -> torch.nn.Module:
    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=False,
        output_stride=cfg.output_stride,
        head_norm=cfg.head_norm,
    ).to(device)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


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

    # Void ignore
    lut[30] = np.uint8(ignore_index)

    missing = [i for i in range(32) if lut[i] == ignore_index]
    if missing:
        print(f"[WARN] these old ids are not assigned (will be ignored): {missing}")
    return lut


@torch.inference_mode()
def save_all_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    id2color,
    ignore_index: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_color_dir = out_dir / "pred_color"
    pred_id_dir = out_dir / "pred_id"
    pred_color_dir.mkdir(parents=True, exist_ok=True)
    pred_id_dir.mkdir(parents=True, exist_ok=True)

    pred_counts = np.zeros((num_classes,), dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)  # [gt, pred]

    for imgs, masks, names in loader:
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        pred = torch.argmax(logits, dim=1)  # (N,H,W)

        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        gt_np = masks.detach().cpu().numpy().astype(np.int64)

        valid_pred = (pred_np >= 0) & (pred_np < num_classes)
        if np.any(valid_pred):
            pred_counts += np.bincount(pred_np[valid_pred], minlength=num_classes)[:num_classes]

        valid_gt = (gt_np != ignore_index) & (gt_np >= 0) & (gt_np < num_classes)
        valid_pair = valid_gt & valid_pred
        if np.any(valid_pair):
            linear = gt_np[valid_pair] * num_classes + pred_np[valid_pair]
            confusion += np.bincount(linear, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        pred_u8 = pred_np.astype(np.uint8)
        for i in range(pred_u8.shape[0]):
            stem = Path(names[i]).stem
            pr_id = pred_u8[i]  # (H,W) 离散标签：不要做 bilinear resize
            pr_rgb = id_mask_to_color(pr_id, id2color, ignore_index)  # (H,W,3)

            Image.fromarray(pr_rgb).save(pred_color_dir / f"{stem}.png")
            Image.fromarray(pr_id).save(pred_id_dir / f"{stem}.png")

    return pred_counts, confusion


def compute_split_class_distribution(
    cfg: TestConfig,
    color2id,
    label_lut: np.ndarray,
    split: str,
    train_preprocess: dict,
    eval_preprocess: dict,
) -> np.ndarray:
    ds = CamVidFolderDataset(
        root=cfg.data_root,
        split=split,
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=False,
        label_lut=label_lut,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    counts = np.zeros((cfg.num_classes,), dtype=np.int64)
    for _imgs, masks, _names in loader:
        masks_np = masks.detach().cpu().numpy().astype(np.int64)
        valid = (masks_np != cfg.ignore_index) & (masks_np >= 0) & (masks_np < cfg.num_classes)
        if np.any(valid):
            counts += np.bincount(masks_np[valid], minlength=cfg.num_classes)[:cfg.num_classes]
    return counts


def print_distribution_comparison(
    class_names: list[str],
    pred_counts: np.ndarray,
    train_counts: np.ndarray,
    lower: float,
    upper: float,
    collapse_warn_ratio: float,
) -> None:
    pred_total = max(int(pred_counts.sum()), 1)
    train_total = max(int(train_counts.sum()), 1)

    pred_ratio = pred_counts / pred_total
    train_ratio = train_counts / train_total

    print("[DIST] class pixel ratio: test prediction vs train ground-truth")
    for i, cls_name in enumerate(class_names):
        p = float(pred_ratio[i])
        t = float(train_ratio[i])
        if t <= 0.0:
            drift = np.inf if p > 0 else 1.0
        else:
            drift = p / t

        drift_flag = ""
        if drift < lower or drift > upper:
            drift_flag = " <-- 异常偏离"
        print(
            f"  - {cls_name:<11} pred={p * 100:6.2f}% | train={t * 100:6.2f}% | "
            f"ratio={drift:6.2f}{drift_flag}"
        )

    max_idx = int(np.argmax(pred_ratio))
    if pred_ratio[max_idx] > collapse_warn_ratio:
        print(
            "[WARN] 疑似类别塌缩/映射异常: "
            f"{class_names[max_idx]} 占比 {pred_ratio[max_idx] * 100:.2f}% (> {collapse_warn_ratio * 100:.0f}%)"
        )


def save_topk_confusion_pairs(
    confusion: np.ndarray,
    class_names: list[str],
    out_dir: Path,
    topk: int,
) -> None:
    pairs = []
    n = confusion.shape[0]
    for gt in range(n):
        for pred in range(n):
            if gt == pred:
                continue
            cnt = int(confusion[gt, pred])
            if cnt > 0:
                pairs.append((cnt, gt, pred))

    pairs.sort(reverse=True)
    top_pairs = pairs[:max(1, int(topk))]

    out_path = out_dir / "topk_confusion_pairs.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Top-K confusion pairs (gt -> pred):\n")
        for rank, (cnt, gt, pred) in enumerate(top_pairs, start=1):
            line = f"{rank:02d}. {class_names[gt]} -> {class_names[pred]}: {cnt}\n"
            f.write(line)

    print("[CONFUSION] Top-K 错分对:")
    for rank, (cnt, gt, pred) in enumerate(top_pairs, start=1):
        print(f"  {rank:02d}. {class_names[gt]} -> {class_names[pred]}: {cnt}")
    print(f"[CONFUSION] Saved summary to: {out_path}")


def build_loader(cfg: TestConfig, color2id, label_lut: np.ndarray, train_preprocess: dict, eval_preprocess: dict) -> DataLoader:
    test_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="test",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=False,
        label_lut=label_lut,  # ✅ 关键：和训练一致（32->11）
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )
    return DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resize_w", type=int, default=960)
    p.add_argument("--resize_h", type=int, default=720)

    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--ignore_index", type=int, default=255)
    p.add_argument("--save_triplet_max", type=int, default=100)

    # ✅ 与训练保持一致
    p.add_argument("--output_stride", type=int, default=8, choices=[8, 16])
    p.add_argument("--head_norm", type=str, default="bn", choices=["bn", "gn"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TestConfig(
        data_root=data_root,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        output_stride=args.output_stride,
        head_norm=args.head_norm,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_triplet_max=args.save_triplet_max,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] ckpt  = {cfg.ckpt_path}")
    print(f"[INFO] out   = {cfg.out_dir}")
    print(f"[INFO] os={cfg.output_stride}  head_norm={cfg.head_norm}")

    color2id, id2color_32, id2name = load_class_dict_csv(cfg.data_root / "class_dict.csv")
    assert_camvid_key_old_ids(id2name)

    GROUPS_11 = build_camvid_11_groups_from_names(id2name)
    label_lut = build_merge_lut(GROUPS_11, ignore_index=cfg.ignore_index)

    rep_old_ids_11 = [group[0] for group in GROUPS_11]
    id2color_11 = [id2color_32[i] for i in rep_old_ids_11]

    train_preprocess = {
        "hflip_prob": 0.0,
        "photo_aug_prob": 0.0,
        "brightness_jitter": 0.0,
        "contrast_jitter": 0.0,
        "saturation_jitter": 0.0,
        "gamma_range": (1.0, 1.0),
        "photo_op_prob": 0.0,
        "blur_prob": 0.0,
        "blur_radius_range": (0.0, 0.0),
        "jpeg_prob": 0.0,
        "jpeg_quality_range": (95, 100),
        "multi_scale_range": (1.0, 1.0),
        "random_crop_size": None,
        "auto_contrast": False,
        "auto_contrast_cutoff": 1.0,
        "low_light_preprocess_enable": False,
        "low_light_gamma": 1.0,
        "low_light_brightness_gain": 1.0,
    }
    eval_preprocess = {
        "auto_contrast": cfg.eval_auto_contrast_enable,
        "auto_contrast_cutoff": cfg.eval_auto_contrast_cutoff,
        "low_light_preprocess_enable": cfg.eval_low_light_preprocess_enable,
        "low_light_gamma": cfg.low_light_gamma,
        "low_light_brightness_gain": cfg.low_light_brightness_gain,
    }

    print(
        "[PREPROCESS][test] "
        f"low_light={cfg.eval_low_light_preprocess_enable} gamma={cfg.low_light_gamma:.2f} "
        f"brightness_gain={cfg.low_light_brightness_gain:.2f} "
        f"auto_contrast={cfg.eval_auto_contrast_enable} cutoff={cfg.eval_auto_contrast_cutoff:.2f}"
    )

    test_loader = build_loader(cfg, color2id, label_lut, train_preprocess, eval_preprocess)
    model = load_model(cfg, device)

    test_miou = compute_miou(model, test_loader, device, cfg.num_classes, cfg.ignore_index)
    print(f"[TEST] mIoU = {test_miou:.4f}")

    save_predictions_triplet(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=cfg.out_dir / "triplets",
        id2color=id2color_11,  # ✅ 11类颜色
        ignore_index=cfg.ignore_index,
        epoch=0,
        max_items=cfg.save_triplet_max,
    )

    pred_counts, confusion = save_all_predictions(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=cfg.out_dir,
        id2color=id2color_11,  # ✅ 11类颜色
        ignore_index=cfg.ignore_index,
        num_classes=cfg.num_classes,
    )

    train_counts = compute_split_class_distribution(
        cfg=cfg,
        color2id=color2id,
        label_lut=label_lut,
        split="train",
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )
    print_distribution_comparison(
        class_names=CLASS_NAMES_11,
        pred_counts=pred_counts,
        train_counts=train_counts,
        lower=cfg.drift_lower_ratio,
        upper=cfg.drift_upper_ratio,
        collapse_warn_ratio=cfg.collapse_warn_ratio,
    )
    save_topk_confusion_pairs(
        confusion=confusion,
        class_names=CLASS_NAMES_11,
        out_dir=cfg.out_dir,
        topk=cfg.confusion_topk,
    )

    print("[DONE] Test inference finished.")


if __name__ == "__main__":
    main()
