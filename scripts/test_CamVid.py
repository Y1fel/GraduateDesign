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
from src.viz.visualizer import save_predictions_triplet


@dataclass
class TestConfig:
    data_root: Path
    ckpt_path: Path
    out_dir: Path

    num_classes: int = 32
    ignore_index: int = 255

    resize_h: int = 480
    resize_w: int = 600


    batch_size: int = 4
    num_workers: int = 4

    save_triplet_max: int = 25
    use_amp: bool = False


def resolve_ckpt_path(ckpt: Path) -> Path:
    if ckpt.is_file():
        return ckpt

    # run_dir/checkpoints/best.pth
    p1 = ckpt / "checkpoints" / "best.pth"
    if p1.exists():
        return p1

    # checkpoints/best.pth
    p2 = ckpt / "best.pth"
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Cannot find checkpoint under: {ckpt}")


def load_model(cfg: TestConfig, device: torch.device) -> torch.nn.Module:
    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=False,
        output_stride=16,
    ).to(device)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def save_all_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    id2color,
    ignore_index: int,
    use_amp: bool,
) -> None:
    pred_color_dir = out_dir / "pred_color"
    pred_id_dir = out_dir / "pred_id"
    pred_color_dir.mkdir(parents=True, exist_ok=True)
    pred_id_dir.mkdir(parents=True, exist_ok=True)

    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=True) if (use_amp and device.type == "cuda")
        else torch.cuda.amp.autocast(enabled=False)
    )

    for imgs, _masks, names in loader:
        imgs = imgs.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(imgs)
            pred = torch.argmax(logits, dim=1)  # (N,H,W)

        pred_np = pred.detach().cpu().numpy().astype(np.uint8)

        for i in range(pred_np.shape[0]):
            stem = Path(names[i]).stem
            pr_id = pred_np[i]  # (H,W)

            pr_rgb = id_mask_to_color(pr_id, id2color, ignore_index)  # (H,W,3)

            Image.fromarray(pr_rgb).save(pred_color_dir / f"{stem}.png")
            Image.fromarray(pr_id).save(pred_id_dir / f"{stem}.png")


def build_loader(cfg: TestConfig, color2id) -> DataLoader:
    test_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="test",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        hflip_prob=0.0,
        ignore_index=cfg.ignore_index,
        training=False,
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
    p.add_argument("--data_root", type=str, required=True, help="CamVid root, contains train/val/test and *_labels")
    p.add_argument("--ckpt", type=str, required=True, help="best.pth / epoch_xxx.pth / run_dir / checkpoints_dir")
    p.add_argument("--out_dir", type=str, default="", help="output dir for test results (default: <ckpt_parent>/test_results)")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resize_w", type=int, default=600)
    p.add_argument("--resize_h", type=int, default=450)
    p.add_argument("--num_classes", type=int, default=32)
    p.add_argument("--ignore_index", type=int, default=255)
    p.add_argument("--save_triplet_max", type=int, default=232)
    p.add_argument("--amp", action="store_true", help="use autocast for inference (CUDA only)")
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
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_triplet_max=args.save_triplet_max,
        use_amp=args.amp,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] ckpt  = {cfg.ckpt_path}")
    print(f"[INFO] out   = {cfg.out_dir}")

    # class_dict
    color2id, id2color, _id2name = load_class_dict_csv(cfg.data_root / "class_dict.csv")

    # loader / model
    test_loader = build_loader(cfg, color2id)
    model = load_model(cfg, device)

    # quantitative
    test_miou = compute_miou(model, test_loader, device, cfg.num_classes, cfg.ignore_index)
    print(f"[TEST] mIoU = {test_miou:.4f}")

    save_predictions_triplet(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=cfg.out_dir / "triplets",
        id2color=id2color,
        ignore_index=cfg.ignore_index,
        epoch=0,
        max_items=cfg.save_triplet_max,
    )
    save_all_predictions(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=cfg.out_dir,
        id2color=id2color,
        ignore_index=cfg.ignore_index,
        use_amp=cfg.use_amp,
    )

    print("[DONE] Test inference finished.")


if __name__ == "__main__":
    main()
