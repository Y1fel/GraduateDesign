import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.CamVid import CamVidFolderDataset
from src.eval.mIoU import compute_miou
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv
from src.viz.visualizer import save_predictions_triplet
from src.losses.composite import CrossEntropyDiceLoss
from src.models.switch2Norm import freeze_bn_stats


@dataclass
class TrainConfig:
    data_root: Path = Path("D:/MachineLearning/GraduateDesign/data/archive/CamVid")  # 改成你的
    num_classes: int = 32
    ignore_index: int = 255

    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4
    lr_0: float = 1e-4
    weight_decay: float = 1e-4

    output_stride: int = 8
    backbone_pretrained: bool = True

    resize_h: int = 480
    resize_w: int = 600
    hflip_prob: float = 0.5

    save_vis_every: int = 50
    save_vis_max_items: int = 8

    outputs_root: Path = Path("D:/MachineLearning/GraduateDesign/outputs")
    seed: int = 21


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    model.backbone.apply(freeze_bn_stats)
    total_loss, n = 0.0, 0

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    N: int = 369

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # mapping
    csv_path = cfg.data_root / "class_dict.csv"
    color2id, id2color, _id2name = load_class_dict_csv(csv_path)

    # outputs
    out = OutputManager(cfg.outputs_root, exp_name="camvid_deeplabv3plus")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")


    # datasets (不需要 splits，直接用文件夹)
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
    ).to(device)

    criterion = CrossEntropyDiceLoss(
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ce_weight=1.0,
        dice_weight=0.5,
        class_weights=None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_0, weight_decay=cfg.weight_decay)

    best_miou = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_miou = compute_miou(model, val_loader, device, cfg.num_classes, cfg.ignore_index)

        dt = time.time() - t0
        print(f"[EPOCH {epoch:03d}/{cfg.epochs}] loss={train_loss:.4f}  val_mIoU={val_miou:.4f}  time={dt:.1f}s")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(epoch, train_loss, val_miou, dt)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
        }
        if epoch % 10 == 0:
            torch.save(ckpt, out.ckpt_dir / f"epoch_{epoch:03d}.pth") # save every 10 epoch

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU = {best_miou:.4f} -> saved best.pth")

        if epoch % cfg.save_vis_every == 0:
            print(f"[INFO] Saving visualizations at epoch {epoch} ...")
            save_predictions_triplet(
                model=model,
                loader=val_loader,
                device=device,
                out_dir=out.vis_dir,
                id2color=id2color,
                ignore_index=cfg.ignore_index,
                epoch=epoch,
                max_items=cfg.save_vis_max_items,
            )


        lr_1: float = cfg.lr_0 * (1 - epoch / cfg.epochs)**0.9  # poly LR schedule
        for pg in optimizer.param_groups:
            pg["lr"] = lr_1

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
