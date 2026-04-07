import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config.config import MobileTrainConfig, TrainConfig
from src.commom.constants import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.factory import apply_dataset_profile, build_dataset, resolve_dataset_root
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.deeplabv3_plus_moblie import DeepLabV3PlusMobile

SAVE_COLOR = True
SAVE_COMPARE = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def detect_teacher_segmentation_head(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.") for k in state.keys()):
        return "hybrid"
    if any(k.startswith("ocr_pre.") or k.startswith("ocr_head.") for k in state.keys()):
        return "ocr"
    return "aspp"


def detect_teacher_hybrid_variant(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.mid_kernel_branch.") or k == "hybrid_neck.mid_scale_logit" for k in state.keys()):
        return "large_v3"
    return "large"


def detect_decoder_upsample_mode(state: dict[str, torch.Tensor]) -> str:
    if any(
        k.startswith("decoder.aspp_upsample.pre.") or k.startswith("decoder.aspp_upsample.post.")
        for k in state.keys()
    ):
        return "learnable"
    return "bilinear"


def build_model(cfg: TrainConfig, ckpt_path: Path, device: torch.device, model_type: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint format: expected dict-like state_dict, got {type(state)}")

    decoder_upsample_mode = detect_decoder_upsample_mode(state)

    if model_type == "teacher":
        teacher_head = detect_teacher_segmentation_head(state)
        teacher_hybrid_variant = detect_teacher_hybrid_variant(state) if teacher_head == "hybrid" else "large"
        model = DeepLabV3Plus(
            num_classes=cfg.num_classes,
            backbone_pretrained=False,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            segmentation_head=teacher_head,
            aspp_dropout=cfg.aspp_dropout,
            hybrid_variant=teacher_hybrid_variant,
            hybrid_use_strip=cfg.hybrid_use_strip,
            hybrid_strip_kernel=cfg.hybrid_strip_kernel,
            hybrid_mid_kernel=cfg.hybrid_mid_kernel,
            hybrid_large_kernel=cfg.hybrid_large_kernel,
            hybrid_gate_reduction=cfg.hybrid_gate_reduction,
            hybrid_residual_channels=cfg.hybrid_residual_channels,
            hybrid_residual_init=cfg.hybrid_residual_init,
            hybrid_dropout=cfg.hybrid_dropout,
            ocr_mid_channels=cfg.ocr_mid_channels,
            ocr_key_channels=cfg.ocr_key_channels,
            ocr_dropout=cfg.ocr_dropout,
            decoder_upsample_mode=decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )
    else:
        mobile_output_stride = cfg.output_stride if cfg.output_stride in (16, 32) else 16
        model = DeepLabV3PlusMobile(
            num_classes=cfg.num_classes,
            output_stride=mobile_output_stride,
            backbone_pretrained=False,
            aspp_dropout=cfg.aspp_dropout,
            decoder_upsample_mode=decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model

def colorize_pred(pred: np.ndarray, id2color) -> np.ndarray:
    color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(id2color):
        color[pred == class_id] = rgb
    return color


def build_save_paths(out_dir: Path, rel_name: str) -> tuple[Path, Path, Path]:
    rel_path = Path(rel_name)
    city = rel_path.parent
    stem = rel_path.stem

    train_id_path = out_dir / "pred_trainIds" / city / f"{stem}_predTrainIds.png"
    color_path = out_dir / "pred_color" / city / f"{stem}_color.png"
    compare_path = out_dir / "compare" / city / f"{stem}_compare.png"
    return train_id_path, color_path, compare_path


def denormalize_image_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=img_tensor.dtype,
        device=img_tensor.device,
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=img_tensor.dtype,
        device=img_tensor.device,
    ).view(3, 1, 1)

    img = (img_tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


@torch.inference_mode()
def main() -> None:
    model_type = "teacher"
    cfg = TrainConfig() if model_type == "teacher" else MobileTrainConfig()
    apply_dataset_profile(cfg)
    cfg.data_root = resolve_dataset_root(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_time = 0.0
    ckpt_path = PROJECT_ROOT / "outputs" / "best.pth"
    out_dir = PROJECT_ROOT / "outputs" / "Predictions"

    test_ds = build_dataset(cfg, split="test", training=False)
    cfg.num_classes = int(test_ds.meta.num_classes)
    id2color = list(test_ds.meta.id2color)
    test_loader = DataLoader(
        test_ds,
        batch_size=12,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
    )

    model = build_model(cfg=cfg, ckpt_path=ckpt_path, device=device, model_type=model_type)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(test_ds)
    seen = 0

    print("[INFO] Begin")
    print(f"[INFO] dataset_name={cfg.dataset_name}")
    print(f"[INFO] data_root={cfg.data_root}")
    print(f"[INFO] model_type={model_type}")
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
            train_id_path, color_path, compare_path = build_save_paths(out_dir, rel_name)

            train_id_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred.astype(np.uint8), mode="L").save(train_id_path)

            color_mask = colorize_pred(pred, id2color)
            if SAVE_COLOR:
                color_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(color_mask, mode="RGB").save(color_path)

            if SAVE_COMPARE:
                original = denormalize_image_to_uint8(imgs[i])
                merged = np.concatenate([original, color_mask], axis=1)
                compare_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(merged, mode="RGB").save(compare_path)

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
