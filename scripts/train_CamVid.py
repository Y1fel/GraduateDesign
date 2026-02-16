import time
from dataclasses import dataclass
from pathlib import Path

from fontTools.misc.arrayTools import scaleRect
from torch.amp import autocast, GradScaler

import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.CamVid import CamVidFolderDataset
from src.eval.mIoU import compute_miou
from src.losses.composite import CrossEntropyDiceLoss
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv
from src.viz.visualizer import save_predictions_triplet
from src.losses.Focal import FocalLoss


@dataclass
class TrainConfig:
    data_root: Path = PROJECT_ROOT / "data" / "archive" / "CamVid"
    num_classes: int = 32
    ignore_index: int = 255

    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    lr_0: float = 1e-4
    weight_decay: float = 1e-4

    output_stride: int = 8
    backbone_pretrained: bool = True

    resize_h: int = 720
    resize_w: int = 960
    hflip_prob: float = 0.5

    save_vis_every: int = 50
    save_vis_max_items: int = 8

    outputs_root: Path = PROJECT_ROOT / "outputs"
    seed: int = 40

@torch.inference_mode()
def compute_class_weights_enet(
    loader: DataLoader,
    num_classes: int,
    ignore_index: int = 255,
    c: float = 1.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)

    for batch in loader:
        masks = batch[1]  # (B, H, W)
        masks = masks.detach().to("cpu", dtype=torch.int64)

        valid = masks != ignore_index
        if valid.any():
            binc = torch.bincount(masks[valid].view(-1), minlength=num_classes).to(torch.float64)
            counts += binc

    freq = counts / (counts.sum() + eps)
    weights = 1.0 / torch.log(c + freq + eps)  # ENet 常用的 log 平衡形式
    return weights.to(torch.float32)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler:GradScaler | None,
                    epoch: int, total_iters: int, base_lr: float, power: float = 0.9) -> float:
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

    # 1) 备份当前权重（在 CPU 上备份即可）
    cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 2) 如果 best.pth 存在，加载 best 权重
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

    # 3) 恢复当前训练权重
    model.load_state_dict(cur_state, strict=True)


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(enabled=(device.type == "cuda"))

    print(f"[INFO] device = {device}")

    # mapping
    csv_path = cfg.data_root / "class_dict.csv"
    color2id, id2color, _id2name = load_class_dict_csv(csv_path)

    # outputs
    out = OutputManager(cfg.outputs_root, exp_name="camvid_deeplabv3plus")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")

    # datasets（不需要 splits，直接用文件夹）
    train_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="train",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        hflip_prob=cfg.hflip_prob,
        ignore_index=cfg.ignore_index,
        training=True,
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
        head_norm="bn",
    ).to(device)

    class_weights = compute_class_weights_enet(
        train_loader,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )

    print("class_weights:", class_weights.detach().cpu().tolist())

    criterion = FocalLoss(
        gamma=2.0,
        alpha=class_weights,
        ignore_index=cfg.ignore_index,
        reduction="mean",
    ).to(device)

    #criterion = CrossEntropyDiceLoss(
    #    num_classes=cfg.num_classes,
    #    ignore_index=cfg.ignore_index,
    #    ce_weight=1.0,
    #    dice_weight=0.1,
    #    class_weights=None,
    #)
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

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            epoch=epoch, total_iters=total_iters, base_lr=cfg.lr_0
        )
        val_miou = compute_miou(model, val_loader, device, cfg.num_classes, cfg.ignore_index)

        dt = time.time() - t0
        print(f"[EPOCH {epoch:03d}/{cfg.epochs}] loss={train_loss:.4f}  val_mIoU={val_miou:.4f}  time={dt:.1f}s")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(epoch, train_loss, val_miou, dt)

        # ckpt
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
                id2color=id2color,
                ignore_index=cfg.ignore_index,
                epoch=epoch,
                max_items=cfg.save_vis_max_items,
                best_ckpt_path=out.ckpt_dir / "best.pth",
            )

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"... lr={cur_lr:.6f}") # print lr

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
