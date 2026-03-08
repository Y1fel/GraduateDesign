from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config.config import TrainConfig
from src.datasets.cityscapes import CityscapesDataset
from src.datasets.cityscapes_labels import CITYSCAPES_19_ID2COLOR
from src.models.deeplabv3_plus import DeepLabV3Plus

# 直接运行脚本时使用的固定配置（无命令行参数）
SAVE_COLOR = True
OUT_DIR_NAME = "test_predictions"


def _find_best_ckpt(cfg: TrainConfig) -> Path:
    # 优先使用固定路径 outputs/best.pth
    direct_best = cfg.outputs_root / "best.pth"
    if direct_best.exists():
        return direct_best

    # 否则在训练输出目录中找最新实验的 checkpoints/best.pth
    run_dirs = sorted(cfg.outputs_root.glob("cityscapes_deeplabv3plus_*/checkpoints/best.pth"))
    if run_dirs:
        return run_dirs[-1]

    raise FileNotFoundError(
        "未找到 best checkpoint。请确认存在以下任一路径：\n"
        f"1) {direct_best}\n"
        f"2) {cfg.outputs_root}/cityscapes_deeplabv3plus_*/checkpoints/best.pth"
    )


def _build_model(cfg: TrainConfig, ckpt_path: Path, device: torch.device) -> DeepLabV3Plus:
    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=False,
        backbone_name=cfg.backbone_name,
        output_stride=cfg.output_stride,
        aspp_dropout=cfg.aspp_dropout,
        decoder_dropout=cfg.decoder_dropout,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _save_train_id_mask(pred: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred.astype(np.uint8), mode="L").save(save_path)


def _save_color_mask(pred: np.ndarray, save_path: Path) -> None:
    color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(CITYSCAPES_19_ID2COLOR):
        color[pred == class_id] = rgb
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color, mode="RGB").save(save_path)


def _build_save_paths(out_dir: Path, rel_name: str) -> tuple[Path, Path]:
    rel_path = Path(rel_name)
    city = rel_path.parent
    stem = rel_path.stem

    train_id_path = out_dir / "pred_trainIds" / city / f"{stem}_predTrainIds.png"
    color_path = out_dir / "pred_color" / city / f"{stem}_color.png"
    return train_id_path, color_path


@torch.inference_mode()
def main() -> None:
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _find_best_ckpt(cfg)
    out_dir = cfg.outputs_root / OUT_DIR_NAME

    test_ds = CityscapesDataset(
        root=cfg.data_root,
        split="test",
        ignore_index=cfg.ignore_index,
        training=False,
        remap_to_19=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )

    model = _build_model(cfg=cfg, ckpt_path=ckpt_path, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(test_ds)
    seen = 0

    print("[INFO] 开始测试集推理")
    print(f"[INFO] data_root={cfg.data_root}")
    print(f"[INFO] total_samples={total}, batch_size={cfg.batch_size}, device={device}")
    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] output_dir={out_dir}")

    use_amp = device.type == "cuda"
    for imgs, _masks, names in test_loader:
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        for pred, rel_name in zip(preds, names):
            train_id_path, color_path = _build_save_paths(out_dir, rel_name)
            _save_train_id_mask(pred, train_id_path)
            if SAVE_COLOR:
                _save_color_mask(pred, color_path)

        seen += imgs.size(0)
        print(f"[INFO] progress: {seen}/{total}")

    print("✅ 测试集推理完成")


if __name__ == "__main__":
    main()
