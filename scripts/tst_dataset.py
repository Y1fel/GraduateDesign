from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config.config import MobileTrainConfig, TrainConfig
from src.commom.constants import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.cityscapes import CityscapesDataset
from src.datasets.cityscapes_labels import CITYSCAPES_19_ID2COLOR
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.deeplabv3_plus_moblie import DeepLabV3PlusMobile

SAVE_COLOR = True
SAVE_COMPARE = True
OUT_DIR_NAME = "test_predictions"


def _find_best_ckpt(cfg: TrainConfig) -> Path:
    return cfg.outputs_root / "best.pth"


def _infer_model_type_from_state(state: dict[str, torch.Tensor]) -> str:
    keys = state.keys()
    if any(k.startswith("backbone.features.") for k in keys):
        return "student"
    if any(k.startswith("backbone.layer1.") for k in keys):
        return "teacher"
    raise ValueError("无法从 checkpoint 自动识别模型类型，请检查 ckpt 是否为 DeepLabV3+ / DeepLabV3PlusMobile。")


def _build_model(cfg: TrainConfig, ckpt_path: Path, device: torch.device, model_type: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint format: expected dict-like state_dict, got {type(state)}")

    inferred_model_type = _infer_model_type_from_state(state)
    if inferred_model_type != model_type:
        print(f"[WARN] --model-type={model_type} 与 checkpoint 不匹配，自动切换为 {inferred_model_type}")
        model_type = inferred_model_type

    if model_type == "teacher":
        model = DeepLabV3Plus(
            num_classes=cfg.num_classes,
            backbone_pretrained=False,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            aspp_dropout=cfg.aspp_dropout,
            decoder_dropout=cfg.decoder_dropout,
        )
    else:
        mobile_output_stride = cfg.output_stride if cfg.output_stride in (16, 32) else 16
        model = DeepLabV3PlusMobile(
            num_classes=cfg.num_classes,
            output_stride=mobile_output_stride,
            aspp_dropout=cfg.aspp_dropout,
            decoder_dropout=cfg.decoder_dropout,
        )

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test dataset prediction export")
    parser.add_argument(
        "--model-type",
        choices=["teacher", "student"],
        default="student",
        help="选择推理模型：teacher(DeepLabV3+) 或 student(DeepLabV3PlusMobile)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="可选 checkpoint 路径；不传则使用对应配置默认 outputs_root/best.pth",
    )
    return parser.parse_args()


def _colorize_pred(pred: np.ndarray) -> np.ndarray:
    color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(CITYSCAPES_19_ID2COLOR):
        color[pred == class_id] = rgb
    return color


def _save_train_id_mask(pred: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred.astype(np.uint8), mode="L").save(save_path)


def _save_color_mask(color: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color, mode="RGB").save(save_path)


def _to_uint8_image(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _save_compare_image(img_tensor: torch.Tensor, color_mask: np.ndarray, save_path: Path) -> None:
    original = _to_uint8_image(img_tensor)
    merged = np.concatenate([original, color_mask], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(merged, mode="RGB").save(save_path)


def _build_save_paths(out_dir: Path, rel_name: str) -> tuple[Path, Path, Path]:
    rel_path = Path(rel_name)
    city = rel_path.parent
    stem = rel_path.stem

    train_id_path = out_dir / "pred_trainIds" / city / f"{stem}_predTrainIds.png"
    color_path = out_dir / "pred_color" / city / f"{stem}_color.png"
    compare_path = out_dir / "compare" / city / f"{stem}_compare.png"
    return train_id_path, color_path, compare_path


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    cfg = TrainConfig() if args.model_type == "teacher" else MobileTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_time = 0.0
    ckpt_path = args.ckpt if args.ckpt is not None else _find_best_ckpt(cfg)
    out_dir = cfg.outputs_root / OUT_DIR_NAME / args.model_type

    test_ds = CityscapesDataset(
        root=cfg.data_root,
        split="test",
        ignore_index=cfg.ignore_index,
        training=False,
        remap_to_19=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=12,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )

    model = _build_model(cfg=cfg, ckpt_path=ckpt_path, device=device, model_type=args.model_type)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(test_ds)
    seen = 0

    print("[INFO] Begin")
    print(f"[INFO] data_root={cfg.data_root}")
    print(f"[INFO] model_type={args.model_type}")
    print(f"[INFO] total_samples={total}, batch_size={cfg.batch_size}, device={device}")
    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] output_dir={out_dir}")

    use_amp = device.type == "cuda"
    for imgs, _masks, names in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        for i, (pred, rel_name) in enumerate(zip(preds, names)):
            train_id_path, color_path, compare_path = _build_save_paths(out_dir, rel_name)
            _save_train_id_mask(pred, train_id_path)
            color_mask = _colorize_pred(pred)
            if SAVE_COLOR:
                _save_color_mask(color_mask, color_path)
            if SAVE_COMPARE:
                _save_compare_image(imgs[i], color_mask, compare_path)

        seen += imgs.size(0)
        print(f"[INFO] progress: {seen}/{total}")

    fps = total / max(total_time, 1e-12)
    average_ms = (total_time / max(total, 1)) * 1000.0
    print("[INFO] Done")
    print(f"[METRIC] infer_total_time={total_time:.4f}s")
    print(f"[METRIC] infer_avg_time_per_image={average_ms:.3f}ms")
    print(f"[METRIC] FPS={fps:.2f}")


if __name__ == "__main__":
    main()
