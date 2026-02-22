import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.cityscapes import CityscapesDataset
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.datasets.cityscapes_labels import CITYSCAPES_19_CLASS_NAMES, CITYSCAPES_19_ID2COLOR
from src.viz.visualizer import save_predictions_triplet
from config.config import TrainConfig


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def _accumulate_pred_hist(pred: torch.Tensor, hist: torch.Tensor, num_classes: int) -> None:
    bins = torch.bincount(pred.view(-1), minlength=num_classes)
    hist += bins.to(hist.device, dtype=hist.dtype)


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float().norm(2).item()
            total += g * g
    return total ** 0.5

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
    num_classes: int,
    freeze_bn_enabled: bool,
    power: float = 0.9,
) -> dict:
    model.train()
    if freeze_bn_enabled:
        freeze_bn(model)
    total_loss, n = 0.0, 0
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    pred_hist = torch.zeros(num_classes, dtype=torch.int64, device=device)
    grad_norm_sum = 0.0
    grad_steps = 0

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
        if use_amp:
            scaler.unscale_(optimizer)
        grad_norm_sum += _compute_grad_norm(model)
        grad_steps += 1
        scaler.step(optimizer)
        scaler.update()

        pred = torch.argmax(logits.detach(), dim=1)
        _accumulate_pred_hist(pred, pred_hist, num_classes=num_classes)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "pred_hist": pred_hist.detach().cpu(),
        "avg_grad_norm": grad_norm_sum / max(grad_steps, 1),
    }


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


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(f"[INFO] AMP enabled = {amp_enabled}")

    id2name = {idx: name for idx, name in enumerate(CITYSCAPES_19_CLASS_NAMES)}
    id2color_vis = CITYSCAPES_19_ID2COLOR

    out = OutputManager(cfg.outputs_root, exp_name="cityscapes_deeplabv3plus")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")

    train_ds = CityscapesDataset(
        root=cfg.data_root,
        split="train",
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=True,
        hflip_prob=cfg.hflip_prob,
        multi_scale_range=(cfg.train_multi_scale_min, cfg.train_multi_scale_max),
        random_crop_size=(cfg.crop_w, cfg.crop_h),
    )
    val_ds = CityscapesDataset(
        root=cfg.data_root,
        split="val",
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=False,
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
    print(f"[INFO] Optimizer = SGD (lr_0={cfg.lr_0:.2e}, momentum=0.9, nesterov=True)")
    print(f"[INFO] freeze_bn = {cfg.freeze_bn}")

    best_miou = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        total_iters = cfg.epochs * len(train_loader)
        t0 = time.time()

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            total_iters=total_iters,
            base_lr=cfg.lr_0,
            use_amp=amp_enabled,
            num_classes=cfg.num_classes,
            freeze_bn_enabled=cfg.freeze_bn,
        )
        train_loss = float(train_stats["loss"])
        val_loss = evaluate_loss(model, val_loader, criterion, device, use_amp=amp_enabled)
        val_metrics = compute_segmentation_metrics(model, val_loader, device, cfg.num_classes, cfg.ignore_index)
        val_miou = float(val_metrics["miou"])
        recall_per_class = val_metrics["recall_per_class"]
        effective_mask = [not math.isnan(float(v)) for v in recall_per_class]
        effective_ious = [
            float(val_metrics["iou_per_class"][i])
            for i, ok in enumerate(effective_mask)
            if ok and not math.isnan(float(val_metrics["iou_per_class"][i]))
        ]
        val_miou_effective = float(sum(effective_ious) / len(effective_ious)) if effective_ious else float("nan")

        iou_per_class = val_metrics["iou_per_class"]
        precision_per_class = val_metrics["precision_per_class"]
        dt = time.time() - t0
        pred_hist = train_stats["pred_hist"]
        pred_total = int(pred_hist.sum().item())
        dominant_ratio = float(pred_hist.max().item() / pred_total) if pred_total > 0 else 0.0
        dominant_class = int(torch.argmax(pred_hist).item()) if pred_total > 0 else -1

        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU(all)={val_miou:.4f} val_mIoU(effective)={val_miou_effective:.4f} "
            f"val_BF1={val_metrics['boundary_fscore']:.4f} val_TrimapIoU={val_metrics['trimap_iou']:.4f} "
            f"grad_norm(avg)={train_stats['avg_grad_norm']:.4f} time={dt:.1f}s"
        )
        print(f"[TRAIN-PRED-HIST] counts={pred_hist.tolist()}")
        print(
            f"[TRAIN-PRED-HIST] dominant_class={dominant_class} dominant_ratio={dominant_ratio:.4f} "
            f"warn_threshold={cfg.dominant_class_warn_ratio:.2f}"
        )
        if dominant_ratio >= cfg.dominant_class_warn_ratio:
            print(
                "[ALERT] Predicted class distribution is highly imbalanced: "
                f"class={dominant_class}, ratio={dominant_ratio:.4f}."
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

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=val_miou_effective,
            val_bf1=float(val_metrics["boundary_fscore"]),
            dt=dt,
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

        if (not math.isnan(val_miou_effective)) and (val_miou_effective > best_miou):
            best_miou = val_miou_effective
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
