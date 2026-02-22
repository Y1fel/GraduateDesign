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

def resolve_effective_tone_aug(cfg: TrainConfig) -> dict[str, float]:
    effective = {
        "photo_aug_prob": float(cfg.photo_aug_prob),
        "photo_op_prob": float(cfg.photo_op_prob),
        "brightness_jitter": float(cfg.brightness_jitter),
        "contrast_jitter": float(cfg.contrast_jitter),
        "saturation_jitter": float(cfg.saturation_jitter),
    }

    if cfg.avoid_overstrong_tone_ops and cfg.train_low_light_preprocess_enable and cfg.train_auto_contrast_enable:
        scale = max(0.0, min(1.0, float(cfg.jitter_scale_when_tone_stack)))
        effective["photo_aug_prob"] = min(effective["photo_aug_prob"], float(cfg.photo_aug_prob_cap_when_tone_stack))
        effective["photo_op_prob"] = min(effective["photo_op_prob"], float(cfg.photo_op_prob_cap_when_tone_stack))
        effective["brightness_jitter"] *= scale
        effective["contrast_jitter"] *= scale
        effective["saturation_jitter"] *= scale
        print(
            "[AUG-TONE] low_light + auto_contrast + photometric stack detected; "
            f"scaled photo_aug_prob={effective['photo_aug_prob']:.3f}, "
            f"photo_op_prob={effective['photo_op_prob']:.3f}, jitter_scale={scale:.2f}"
        )

    return effective


def summarize_tone_cfg(prefix: str, cfg: dict) -> str:
    return (
        f"[{prefix}] auto_contrast={bool(cfg['auto_contrast'])} "
        f"cutoff={float(cfg['auto_contrast_cutoff']):.2f} "
        f"low_light={bool(cfg['low_light_preprocess_enable'])} "
        f"gamma={float(cfg['low_light_gamma']):.2f} "
        f"brightness_gain={float(cfg['low_light_brightness_gain']):.2f}"
    )

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

    tone_aug = resolve_effective_tone_aug(cfg)

    train_preprocess = {
        "hflip_prob": cfg.hflip_prob,
        "photo_aug_prob": tone_aug["photo_aug_prob"],
        "brightness_jitter": tone_aug["brightness_jitter"],
        "contrast_jitter": tone_aug["contrast_jitter"],
        "saturation_jitter": tone_aug["saturation_jitter"],
        "gamma_range": (cfg.gamma_min, cfg.gamma_max),
        "photo_op_prob": tone_aug["photo_op_prob"],
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
    tone_on_eval_preprocess = build_eval_preprocess(
        auto_contrast=cfg.train_auto_contrast_enable,
        auto_contrast_cutoff=cfg.train_auto_contrast_cutoff,
        low_light=cfg.train_low_light_preprocess_enable,
        gamma=cfg.low_light_gamma,
        brightness_gain=cfg.low_light_brightness_gain,
    )
    tone_off_eval_preprocess = build_eval_preprocess(
        auto_contrast=False,
        auto_contrast_cutoff=cfg.eval_auto_contrast_cutoff,
        low_light=False,
        gamma=cfg.low_light_gamma,
        brightness_gain=cfg.low_light_brightness_gain,
    )

    if cfg.sync_eval_tone_with_train:
        eval_preprocess = tone_on_eval_preprocess
    else:
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

        print(
            "[TONE] "
            + summarize_tone_cfg("train", train_preprocess)
            + " | "
            + summarize_tone_cfg("eval", eval_preprocess)
            + f" | sync_eval_tone_with_train={cfg.sync_eval_tone_with_train}"
        )

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

        dt = time.time() - t0
        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} "
            f"val_BF1={val_metrics['boundary_fscore']:.4f} val_TrimapIoU={val_metrics['trimap_iou']:.4f} time={dt:.1f}s"
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
            sync_eval_tone_with_train=cfg.sync_eval_tone_with_train,
            train_auto_contrast_enable=cfg.train_auto_contrast_enable,
            train_low_light_preprocess_enable=cfg.train_low_light_preprocess_enable,
            low_light_gamma=cfg.low_light_gamma,
            low_light_brightness_gain=cfg.low_light_brightness_gain,
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

    best_ckpt_path = out.ckpt_dir / "best.pth"
    if best_ckpt_path.exists():
        print("[TONE-ABLATION] Evaluating best checkpoint with tone on/off val pipeline...")
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

        tone_eval_settings = {
            "tone_on": tone_on_eval_preprocess,
            "tone_off": tone_off_eval_preprocess,
        }
        compare_lines = ["setting,miou,bf1,trimap_iou"]
        for setting_name, setting in tone_eval_settings.items():
            eval_ds_cmp = CamVidFolderDataset(
                root=cfg.data_root,
                split="val",
                color2id=color2id,
                resize_w=cfg.resize_w,
                resize_h=cfg.resize_h,
                ignore_index=cfg.ignore_index,
                training=False,
                label_lut=None,
                train_preprocess=train_preprocess,
                eval_preprocess=setting,
            )
            eval_loader_cmp = DataLoader(
                eval_ds_cmp,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )
            cmp_metrics = compute_segmentation_metrics(model, eval_loader_cmp, device, cfg.num_classes, cfg.ignore_index)
            print(
                f"[TONE-ABLATION] {setting_name} -> "
                f"mIoU={cmp_metrics['miou']:.4f} BF1={cmp_metrics['boundary_fscore']:.4f} "
                f"TrimapIoU={cmp_metrics['trimap_iou']:.4f} "
                f"({summarize_tone_cfg(setting_name, setting)})"
            )
            compare_lines.append(
                f"{setting_name},{cmp_metrics['miou']:.6f},{cmp_metrics['boundary_fscore']:.6f},{cmp_metrics['trimap_iou']:.6f}"
            )
            save_predictions_triplet(
                model=model,
                loader=eval_loader_cmp,
                device=device,
                out_dir=out.vis_dir / "tone_ablation" / setting_name,
                id2color=id2color_vis,
                ignore_index=cfg.ignore_index,
                epoch=cfg.epochs,
                max_items=cfg.save_vis_max_items,
            )

        (out.log_dir / "tone_ablation.csv").write_text("\n".join(compare_lines) + "\n", encoding="utf-8")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
