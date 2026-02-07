from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import csv
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn

from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv, id_mask_to_color, color_mask_to_id

RGB = Tuple[int, int, int]

# Config
@dataclass
class TrainConfig:
    data_root = Path("data/camvid")
    num_classes = 32
    ignore_index = 255

    epochs = 50
    batch_size = 4
    num_workers = 4
    lr = 1e-4
    weight_decay = 1e-4

    output_stride = 16
    backbone_pretrained = True

    resize_h = 360
    resize_w = 480
    hflip_prob = 0.5

    save_vis_every = 10
    save_vis_max_items = 8

    ckpt_dir = Path("outputs/checkpoints")
    log_dir = Path("outputs/logs")
    vis_dir = Path("outputs/visual")

    seed = 42

# Reproducibility & dirs
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: TrainConfig) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.vis_dir.mkdir(parents=True, exist_ok=True)

# Image normalization
def normalize_img(img: torch.Tensor) -> torch.Tensor:
    """
    ImageNet normalization for pretrained ResNet.
    img: (3,H,W) float in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    return (img - mean) / std

# Dataset
class CamVidDataset(Dataset):
    def __init__(self, root: Path, split_file: str, cfg: TrainConfig, color2id: Dict[RGB, int], training: bool):
        self.root = root
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        self.splits_dir = root / "splits"

        self.cfg = cfg
        self.color2id = color2id
        self.training = training

        split_path = self.splits_dir / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        self.names = [ln.strip() for ln in split_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def __len__(self) -> int:
        return len(self.names)

    def _resolve(self, base: Path, name: str, exts: list[str]) -> Path:
        p = base / name
        if p.exists():
            return p
        stem = Path(name).stem
        for ext in exts:
            p2 = base / f"{stem}{ext}"
            if p2.exists():
                return p2
        return p  # will fail later

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_path = self._resolve(self.images_dir, name, [".jpg", ".jpeg", ".png", ".bmp"])
        mask_path = self._resolve(self.masks_dir, name, [".png", ".jpg", ".jpeg", ".bmp"])

        img = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path).convert("RGB")

        # resize
        img = img.resize((self.cfg.resize_w, self.cfg.resize_h), resample=Image.Resampling.BILINEAR)
        mask_rgb = mask_rgb.resize((self.cfg.resize_w, self.cfg.resize_h), resample=Image.Resampling.NEAREST)

        # hflip aug
        if self.training and random.random() < self.cfg.hflip_prob:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_rgb = mask_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # to tensor
        img_t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0  # (3,H,W)
        img_t = normalize_img(img_t)

        mask_id = color_mask_to_id(mask_rgb, self.color2id, self.cfg.ignore_index)  # (H,W) uint8
        mask_t = torch.from_numpy(mask_id.astype(np.int64))

        return img_t, mask_t, name

# mIoU
@torch.no_grad()
def update_confusion_matrix(conf: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> None:
    pred = pred.view(-1)
    target = target.view(-1)

    m = target != ignore_index
    pred = pred[m]
    target = target[m]
    if pred.numel() == 0:
        return

    k = (target * num_classes + pred).to(torch.int64)
    bins = torch.bincount(k, minlength=num_classes * num_classes)
    conf += bins.view(num_classes, num_classes)


@torch.no_grad()
def compute_iou_from_conf(conf: torch.Tensor) -> tuple[np.ndarray, float]:
    conf_np = conf.detach().cpu().numpy().astype(np.float64)
    tp = np.diag(conf_np)
    fp = conf_np.sum(axis=0) - tp
    fn = conf_np.sum(axis=1) - tp
    denom = tp + fp + fn

    iou = np.full(tp.shape, np.nan, dtype=np.float64)
    valid = denom > 0
    iou[valid] = tp[valid] / denom[valid]

    miou = float(np.nanmean(iou)) if np.any(valid) else float("nan")
    return iou, miou


# Train
def train(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0

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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> tuple[np.ndarray, float]:
    model.eval()
    conf = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.int64, device=device)

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        pred = torch.argmax(logits, dim=1)
        update_confusion_matrix(conf, pred, masks, cfg.num_classes, cfg.ignore_index)

    return compute_iou_from_conf(conf)


@torch.no_grad()
def save_visualizations(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig, id2color: List[RGB], epoch: int) -> None:
    model.eval()
    out_dir = cfg.vis_dir / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for imgs, masks, names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        pred = torch.argmax(logits, dim=1)

        imgs_cpu = imgs.detach().cpu()
        masks_cpu = masks.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()

        for i in range(imgs_cpu.shape[0]):
            if saved >= cfg.save_vis_max_items:
                return

            # de-normalize for saving (approx)
            img = imgs_cpu[i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img * std + mean).clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            gt_id = masks_cpu[i].astype(np.uint8)
            pr_id = pred_cpu[i].astype(np.uint8)

            gt_rgb = id_mask_to_color(gt_id, id2color, cfg.ignore_index)
            pr_rgb = id_mask_to_color(pr_id, id2color, cfg.ignore_index)

            triplet = np.concatenate([img_np, gt_rgb, pr_rgb], axis=1)

            stem = Path(names[i]).stem
            Image.fromarray(triplet).save(out_dir / f"{stem}_triplet.png")
            Image.fromarray(pr_rgb).save(out_dir / f"{stem}_pred_color.png")
            Image.fromarray(pr_id).save(out_dir / f"{stem}_pred_id.png")
            saved += 1


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    if not cfg.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {cfg.data_root}")

    # load mapping
    color2id, id2color, id2name = load_class_dict_csv(cfg.data_root / "class_dict.csv")
    if len(id2color) != cfg.num_classes:
        print(f"[WARN] class_dict.csv has {len(id2color)} classes, but cfg.num_classes={cfg.num_classes}. "
              f"Will train with cfg.num_classes={cfg.num_classes} anyway.")

    # datasets
    train_ds = CamVidDataset(cfg.data_root, "train.txt", cfg, color2id, training=True)
    val_ds = CamVidDataset(cfg.data_root, "val.txt", cfg, color2id, training=False)

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

    # loss/optim
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # logs
    log_path = cfg.log_dir / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_miou", "time_sec"])

    best_miou = -1.0

    # train loop
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device)
        _iou, val_miou = evaluate(model, val_loader, device, cfg)

        dt = time.time() - t0
        print(f"[EPOCH {epoch:03d}/{cfg.epochs}] loss={train_loss:.4f}  val_mIoU={val_miou:.4f}  time={dt:.1f}s")

        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{val_miou:.6f}", f"{dt:.2f}"])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "best_miou": best_miou,
        }
        torch.save(ckpt, cfg.ckpt_dir / f"epoch_{epoch:03d}.pth")

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            torch.save(ckpt, cfg.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU = {best_miou:.4f} -> saved to best.pth")

        if epoch % cfg.save_vis_every == 0:
            print(f"[INFO] Saving visualizations at epoch {epoch} ...")
            save_visualizations(model, val_loader, device, cfg, id2color, epoch)

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
