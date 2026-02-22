import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.CamVid import CamVidFolderDataset
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv
from src.viz.visualizer import save_predictions_triplet
from config.config import TrainConfig


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    total_iters: int,
    base_lr: float,
    use_amp: bool,
    power: float = 0.9,
) -> float:
    model.train()
    freeze_bn(model)
    total_loss, n = 0.0, 0
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    for it, (imgs, masks, _names) in enumerate(loader):
        global_step = (epoch - 1) * len(loader) + it
        lr = base_lr * (1 - global_step / total_iters) ** power
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda',enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    n = 0

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda',enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)

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


def build_eval_preprocess(auto_contrast: bool, auto_contrast_cutoff: float, low_light: bool, gamma: float, brightness_gain: float) -> dict:
    return {
        "auto_contrast": auto_contrast,
        "auto_contrast_cutoff": auto_contrast_cutoff,
        "low_light_preprocess_enable": low_light,
        "low_light_gamma": gamma,
        "low_light_brightness_gain": brightness_gain,
    }


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(f"[INFO] AMP enabled = {amp_enabled}")

    csv_path = cfg.data_root / "class_dict.csv"
    color2id, id2color, id2name = load_class_dict_csv(csv_path)

    id2color_vis = id2color

    out = OutputManager(cfg.outputs_root, exp_name="camvid_deeplabv3plus")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")

    train_preprocess = {
        "hflip_prob": cfg.hflip_prob,
        "photo_aug_prob": cfg.photo_aug_prob,
        "brightness_jitter": cfg.brightness_jitter,
        "contrast_jitter": cfg.contrast_jitter,
        "saturation_jitter": cfg.saturation_jitter,
        "gamma_range": (cfg.gamma_min, cfg.gamma_max),
        "photo_op_prob": cfg.photo_op_prob,
        "blur_prob": cfg.blur_prob,
        "blur_radius_range": (cfg.blur_radius_min, cfg.blur_radius_max),
        "jpeg_prob": cfg.jpeg_prob,
        "jpeg_quality_range": (cfg.jpeg_quality_min, cfg.jpeg_quality_max),
        "multi_scale_range": (cfg.train_multi_scale_min, cfg.train_multi_scale_max),
        "random_crop_size": None,
        "auto_contrast": cfg.train_auto_contrast_enable,
        "auto_contrast_cutoff": cfg.train_auto_contrast_cutoff,
        "low_light_preprocess_enable": cfg.train_low_light_preprocess_enable,
        "low_light_gamma": cfg.low_light_gamma,
        "low_light_brightness_gain": cfg.low_light_brightness_gain,
    }
    eval_preprocess = build_eval_preprocess(
        auto_contrast=cfg.eval_auto_contrast_enable,
        auto_contrast_cutoff=cfg.eval_auto_contrast_cutoff,
        low_light=cfg.eval_low_light_preprocess_enable,
        gamma=cfg.low_light_gamma,
        brightness_gain=cfg.low_light_brightness_gain,
    )

    train_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="train",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=True,
        label_lut=None,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )
    val_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="val",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=False,
        label_lut=None,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
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

    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=cfg.backbone_pretrained,
        output_stride=cfg.output_stride,
        head_norm=cfg.head_norm,
    ).to(device)

    print("====== 检查 backbone 是否加载预训练 ======")

    # 假设 model.encoder 是 backbone（如果你是 model.backbone 自己改一下）
    backbone = model.backbone

    # 找到第一层卷积
    first_conv = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    w = first_conv.weight.data
    print("conv1 weight mean:", w.mean().item())
    print("conv1 weight std :", w.std().item())

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=cfg.ignore_index,
        label_smoothing=cfg.label_smoothing,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr_0,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    best_miou = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        total_iters = cfg.epochs * len(train_loader)
        t0 = time.time()

        if hasattr(train_ds, "reset_aug_stats"):
            train_ds.reset_aug_stats()

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
            epoch=epoch,
            total_iters=total_iters,
            base_lr=cfg.lr_0,
            use_amp=amp_enabled,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device, use_amp=amp_enabled)
        val_metrics = compute_segmentation_metrics(model, val_loader, device, cfg.num_classes, cfg.ignore_index)
        val_miou = float(val_metrics["miou"])

        iou_per_class = val_metrics["iou_per_class"]
        precision_per_class = val_metrics["precision_per_class"]
        recall_per_class = val_metrics["recall_per_class"]

        dt = time.time() - t0
        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} "
            f"val_BF1={val_metrics['boundary_fscore']:.4f} val_TrimapIoU={val_metrics['trimap_iou']:.4f} time={dt:.1f}s"
        )
        print("[PER-CLASS] class_id class_name iou precision recall")
        per_class_rows = []
        for class_id in range(cfg.num_classes):
            class_name = id2name.get(class_id, id2name.get(str(class_id), f"class_{class_id}"))
            iou_val = float(iou_per_class[class_id])
            precision_val = float(precision_per_class[class_id])
            recall_val = float(recall_per_class[class_id])
            print(
                f"[PER-CLASS] {class_id:02d} {class_name:<18} "
                f"iou={iou_val:.4f} precision={precision_val:.4f} recall={recall_val:.4f}"
            )
            per_class_rows.append((class_id, class_name, iou_val, precision_val, recall_val))

        valid_rows = [row for row in per_class_rows if not math.isnan(row[2])]
        if valid_rows:
            bottom_k = min(5, len(valid_rows))
            bottom_rows = sorted(valid_rows, key=lambda row: row[2])[:bottom_k]
            print(f"[PER-CLASS][BOTTOM-{bottom_k}] Lowest IoU classes:")
            for class_id, class_name, iou_val, precision_val, recall_val in bottom_rows:
                print(
                    f"[PER-CLASS][BOTTOM] {class_id:02d} {class_name:<18} "
                    f"iou={iou_val:.4f} precision={precision_val:.4f} recall={recall_val:.4f}"
                )

        if hasattr(train_ds, "consume_aug_stats"):
            aug_stats = train_ds.consume_aug_stats()
            print(
                "[AUG-STATS] "
                f"photometric={aug_stats['photometric_applied']}/{aug_stats['samples_seen']} "
                f"blur={aug_stats['blur_applied']} jpeg={aug_stats['jpeg_applied']}"
            )
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=val_miou,
            val_bf1=float(val_metrics["boundary_fscore"]),
            dt=dt,
            train_auto_contrast_enable=cfg.train_auto_contrast_enable,
            train_low_light_preprocess_enable=cfg.train_low_light_preprocess_enable,
            low_light_gamma=cfg.low_light_gamma,
            low_light_brightness_gain=cfg.low_light_brightness_gain,
        )
        for class_id, class_name, iou_val, precision_val, recall_val in per_class_rows:
            out.append_per_class_metrics(
                epoch=epoch,
                class_id=class_id,
                class_name=class_name,
                iou=iou_val,
                precision=precision_val,
                recall=recall_val,
            )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "best_val_loss": best_val_loss,
        }

        if epoch % 10 == 0:
            torch.save(ckpt, out.ckpt_dir / f"epoch_{epoch:03d}.pth")

        if (not math.isnan(val_loss)) and (val_loss < best_val_loss):
            best_val_loss = val_loss

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            ckpt["best_miou"] = best_miou
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU = {best_miou:.4f} -> saved best.pth (current val_loss={val_loss:.4f})")

        if epoch % cfg.save_vis_every == 0:
            print(f"[INFO] Saving visualizations (best.pth) at epoch {epoch} ...")
            save_vis_using_best_ckpt(
                model=model,
                val_loader=val_loader,
                device=device,
                out_dir=out.vis_dir,
                id2color=id2color_vis,
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
