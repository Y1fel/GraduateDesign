from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_omp_threads = os.environ.get("OMP_NUM_THREADS")
if _omp_threads is not None:
    try:
        if int(_omp_threads) <= 0:
            os.environ.pop("OMP_NUM_THREADS", None)
    except ValueError:
        os.environ.pop("OMP_NUM_THREADS", None)

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

from config.config import TrainConfig
from src.datasets.factory import apply_dataset_profile, build_dataset, normalize_dataset_name, resolve_dataset_root
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.deeplabv3_plus_moblie import DeepLabV3PlusMobile
from src.models.factory import build_segmentation_model, normalize_model_name
from src.utils.Id2Mask import id_mask_to_color

RESAMPLING_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
RESAMPLING_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


# -----------------------------------------------------------------------------
# Editable config
# -----------------------------------------------------------------------------
DATASET_NAME = "camvid"
DATASET_SPLIT = "test"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "visualizer" / "camvid_model_compare_test.png"

# Keep four samples here to match the paper-style layout in your reference.
SAMPLE_KEYS = [
    "Seq05VD_f01500",
    "Seq05VD_f03420",
    "Seq05VD_f04080",
    "Seq05VD_f04920",
]

# Control how many models are rendered.
# The script uses the first NUM_MODELS entries in MODEL_SPECS.
NUM_MODELS = 4

MODEL_SPECS = [
    {
        "label": "Teacher",
        "run_dir": PROJECT_ROOT / "outputs" / "Camvid",
    },
    {
        "label": "FCN",
        "run_dir": PROJECT_ROOT / "outputs" / "FCN_Camvid",
    },
    {
        "label": "U-Net",
        "run_dir": PROJECT_ROOT / "outputs" / "Unet-camvid",
    },
    {
        "label": "PSPNet",
        "run_dir": PROJECT_ROOT / "outputs" / "PSPNet_Camvid",
    },
]

# Panel appearance.
OUTER_PADDING = 18
ROW_GAP = 0
COL_GAP = 0
TILE_WIDTH = 300
BORDER_WIDTH = 1
BACKGROUND_RGB = (255, 255, 255)
BORDER_RGB = (222, 226, 230)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    run_dir: Path


@dataclass
class SampleItem:
    key: str
    rel_name: str
    image_rgb: np.ndarray
    gt_color: np.ndarray
    image_tensor: torch.Tensor


def _ensure_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _find_camvid_root() -> Path | None:
    candidates = [
        PROJECT_ROOT / "data" / "CamVid",
        PROJECT_ROOT / "data" / "Camvid",
        PROJECT_ROOT / "data" / "camvid",
    ]
    for root in candidates:
        if root.is_dir() and (root / "val").is_dir():
            return root
    return None


def _safe_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint format at {ckpt_path}: expected state_dict, got {type(state)}")
    return state


def _apply_run_config(cfg: TrainConfig, raw_cfg: dict[str, Any], dataset_name: str) -> TrainConfig:
    for key, value in raw_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.dataset_name = normalize_dataset_name(dataset_name)
    cfg.backbone_pretrained = False
    cfg.data_root = resolve_dataset_root(cfg)
    apply_dataset_profile(cfg)
    return cfg


def _detect_mobile_state(state: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("backbone.features.") for key in state.keys())


def _detect_teacher_segmentation_head(state: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("hybrid_neck.") for key in state.keys()):
        return "hybrid"
    return "aspp"


def _detect_teacher_hybrid_variant(state: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("hybrid_neck.mid_kernel_branch.") or key == "hybrid_neck.mid_scale_logit" for key in state.keys()):
        return "large_v3"
    return "large"


def _detect_decoder_upsample_mode(state: dict[str, torch.Tensor]) -> str:
    if any(
        key.startswith("decoder.aspp_upsample.pre.") or key.startswith("decoder.aspp_upsample.post.")
        for key in state.keys()
    ):
        return "learnable"
    return "bilinear"


def _infer_kernel_from_weight(state: dict[str, torch.Tensor], key: str, default: int) -> int:
    weight = state.get(key)
    if weight is None or weight.ndim < 4:
        return int(default)
    kh = int(weight.shape[-2])
    kw = int(weight.shape[-1])
    if kh == 1:
        return kw
    if kw == 1:
        return kh
    return kw


def _infer_residual_channels(state: dict[str, torch.Tensor], default: int = 128) -> int:
    weight = state.get("hybrid_neck.large_kernel_branch.pre.0.weight")
    if weight is None or weight.ndim < 4:
        return int(default)
    return int(weight.shape[0])


def _load_shape_compatible_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected)


def _build_teacher_model(cfg: TrainConfig, state: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    segmentation_head = _detect_teacher_segmentation_head(state)
    hybrid_variant = _detect_teacher_hybrid_variant(state) if segmentation_head == "hybrid" else "large"
    decoder_upsample_mode = _detect_decoder_upsample_mode(state)
    hybrid_use_strip = any(key.startswith("hybrid_neck.strip_branch.") for key in state.keys())
    has_large_gate = any(key.startswith("hybrid_neck.large_gate.") for key in state.keys())
    has_mid_gate = any(key.startswith("hybrid_neck.mid_gate.") for key in state.keys())
    has_strip_gate = any(key.startswith("hybrid_neck.strip_gate.") for key in state.keys())

    model = DeepLabV3Plus(
        num_classes=int(cfg.num_classes),
        backbone_pretrained=False,
        backbone_name=cfg.backbone_name,
        output_stride=cfg.output_stride,
        segmentation_head=segmentation_head,
        aspp_dropout=cfg.aspp_dropout,
        hybrid_variant=hybrid_variant,
        hybrid_use_strip=hybrid_use_strip,
        hybrid_strip_kernel=_infer_kernel_from_weight(
            state,
            "hybrid_neck.strip_branch.horizontal.block.0.0.weight",
            cfg.hybrid_strip_kernel,
        ),
        hybrid_mid_kernel=_infer_kernel_from_weight(
            state,
            "hybrid_neck.mid_kernel_branch.large_kernel.block.0.0.weight",
            cfg.hybrid_mid_kernel,
        ),
        hybrid_large_kernel=_infer_kernel_from_weight(
            state,
            "hybrid_neck.large_kernel_branch.large_kernel.block.0.0.weight",
            cfg.hybrid_large_kernel,
        ),
        hybrid_gate_reduction=cfg.hybrid_gate_reduction,
        hybrid_residual_channels=_infer_residual_channels(state, cfg.hybrid_residual_channels),
        hybrid_residual_init=cfg.hybrid_residual_init,
        hybrid_dropout=cfg.hybrid_dropout,
        decoder_upsample_mode=decoder_upsample_mode,
        decoder_dropout=cfg.decoder_dropout,
    )

    if segmentation_head == "hybrid":
        if not has_large_gate and hasattr(model.hybrid_neck, "large_gate"):
            model.hybrid_neck.large_gate = nn.Identity()
        if hybrid_variant == "large_v3" and not has_mid_gate and hasattr(model.hybrid_neck, "mid_gate"):
            model.hybrid_neck.mid_gate = nn.Identity()
        if hybrid_use_strip and not has_strip_gate and hasattr(model.hybrid_neck, "strip_gate"):
            model.hybrid_neck.strip_gate = nn.Identity()

    _load_shape_compatible_state(model, state)
    model = model.to(device)
    model.eval()
    return model


def _build_mobile_model(cfg: TrainConfig, state: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    output_stride = cfg.output_stride if int(cfg.output_stride) in (16, 32) else 16
    model = DeepLabV3PlusMobile(
        num_classes=int(cfg.num_classes),
        output_stride=output_stride,
        backbone_pretrained=False,
        aspp_dropout=cfg.aspp_dropout,
        decoder_upsample_mode=_detect_decoder_upsample_mode(state),
        decoder_dropout=cfg.decoder_dropout,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _build_run_model(run_dir: Path, dataset_name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found under {run_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best checkpoint not found under {run_dir}")

    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    state = _safe_state_dict(ckpt_path)

    cfg = TrainConfig()
    cfg = _apply_run_config(cfg, raw_cfg, dataset_name=dataset_name)
    cfg.num_classes = int(num_classes)

    model_name = normalize_model_name(raw_cfg.get("model_name", "deeplabv3plus"))
    if model_name in {"fcn", "unet", "pspnet"}:
        cfg.model_name = model_name
        model = build_segmentation_model(cfg)
        _load_shape_compatible_state(model, state)
        model = model.to(device)
        model.eval()
        return model

    if _detect_mobile_state(state):
        return _build_mobile_model(cfg, state, device)

    return _build_teacher_model(cfg, state, device)


def _resolve_active_models() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for idx, item in enumerate(MODEL_SPECS):
        if "run_dir" not in item:
            raise ValueError(f"MODEL_SPECS[{idx}] must contain 'run_dir'")
        label = str(item.get("label", f"model_{idx + 1}"))
        specs.append(ModelSpec(label=label, run_dir=_ensure_path(item["run_dir"])))
    if NUM_MODELS <= 0:
        raise ValueError(f"NUM_MODELS must be positive, got {NUM_MODELS}")
    if NUM_MODELS > len(specs):
        raise ValueError(f"NUM_MODELS={NUM_MODELS} exceeds MODEL_SPECS length={len(specs)}")
    return specs[:NUM_MODELS]


def _find_sample_indices(dataset, sample_keys: list[str]) -> list[int]:
    names = [str(Path(path).stem) for path in dataset.img_paths]
    indices: list[int] = []
    for key in sample_keys:
        exact = [idx for idx, name in enumerate(names) if name == key or dataset.img_paths[idx].name == key]
        if len(exact) == 1:
            indices.append(exact[0])
            continue
        if len(exact) > 1:
            raise ValueError(f"Multiple exact matches found for sample key '{key}'")

        partial = [idx for idx, name in enumerate(names) if key in name or key in str(dataset.img_paths[idx])]
        if len(partial) == 1:
            indices.append(partial[0])
            continue
        if not partial:
            raise FileNotFoundError(f"No dataset sample matched key '{key}' in split={dataset.split}")
        raise ValueError(f"Multiple partial matches found for sample key '{key}': {partial[:5]}")
    return indices


def _load_samples(dataset, sample_keys: list[str]) -> list[SampleItem]:
    indices = _find_sample_indices(dataset, sample_keys)
    samples: list[SampleItem] = []
    for idx, key in zip(indices, sample_keys):
        image_tensor, mask_tensor, rel_name = dataset[idx]
        image_path = dataset.img_paths[idx]
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gt_color = id_mask_to_color(
            mask_tensor.numpy().astype(np.uint8),
            list(dataset.meta.id2color),
            ignore_index=dataset.ignore_index,
        )
        samples.append(
            SampleItem(
                key=key,
                rel_name=rel_name,
                image_rgb=image_rgb,
                gt_color=gt_color,
                image_tensor=image_tensor,
            )
        )
    return samples


@torch.inference_mode()
def _predict_panel_rows(
    model_specs: list[ModelSpec],
    samples: list[SampleItem],
    dataset,
    device: torch.device,
) -> list[list[np.ndarray]]:
    predictions: list[list[np.ndarray]] = []
    palette = list(dataset.meta.id2color)
    for model_spec in model_specs:
        model = _build_run_model(
            run_dir=model_spec.run_dir,
            dataset_name=dataset.meta.dataset_name,
            num_classes=dataset.meta.num_classes,
            device=device,
        )
        pred_rows: list[np.ndarray] = []
        for sample in samples:
            logits = model(sample.image_tensor.unsqueeze(0).to(device, non_blocking=True))
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
            pred_rows.append(id_mask_to_color(pred, palette, ignore_index=dataset.ignore_index))
        predictions.append(pred_rows)
    return predictions


def _resize_tile(arr: np.ndarray, tile_size: tuple[int, int], mode: str) -> Image.Image:
    image = Image.fromarray(arr)
    if image.size == tile_size:
        return image
    resample = RESAMPLING_BILINEAR if mode == "rgb" else RESAMPLING_NEAREST
    return image.resize(tile_size, resample=resample)


def _build_canvas(
    samples: list[SampleItem],
    predictions: list[list[np.ndarray]],
) -> Image.Image:
    first_height, first_width = samples[0].image_rgb.shape[:2]
    tile_width = min(int(TILE_WIDTH), int(first_width))
    scale = float(tile_width) / float(first_width)
    tile_height = max(1, int(round(first_height * scale)))
    tile_size = (tile_width, tile_height)

    n_rows = 2 + len(predictions)
    n_cols = len(samples)
    content_width = n_cols * tile_width + max(0, n_cols - 1) * COL_GAP
    content_height = n_rows * tile_height + max(0, n_rows - 1) * ROW_GAP

    canvas_width = OUTER_PADDING * 2 + content_width
    canvas_height = OUTER_PADDING * 2 + content_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND_RGB)
    draw = ImageDraw.Draw(canvas)
    row_data: list[list[np.ndarray]] = [
        [sample.image_rgb for sample in samples],
        [sample.gt_color for sample in samples],
    ]
    for model_rows in predictions:
        row_data.append(model_rows)

    for row_idx in range(n_rows):
        y = OUTER_PADDING + row_idx * (tile_height + ROW_GAP)
        for col_idx in range(n_cols):
            x = OUTER_PADDING + col_idx * (tile_width + COL_GAP)
            tile_mode = "rgb"
            tile = _resize_tile(row_data[row_idx][col_idx], tile_size, tile_mode)
            canvas.paste(tile, (x, y))
            if BORDER_WIDTH > 0:
                draw.rectangle(
                    (x, y, x + tile_width, y + tile_height),
                    outline=BORDER_RGB,
                    width=BORDER_WIDTH,
                )

    return canvas


def main() -> None:
    model_specs = _resolve_active_models()

    cfg = TrainConfig()
    cfg.dataset_name = normalize_dataset_name(DATASET_NAME)
    if cfg.dataset_name == "camvid":
        camvid_root = _find_camvid_root()
        if camvid_root is None:
            raise FileNotFoundError(
                f"Unable to locate CamVid root under {PROJECT_ROOT / 'data'}. "
                "Expected one of: CamVid, Camvid, camvid."
            )
        cfg.camvid_root = camvid_root
        cfg.data_root = camvid_root
    cfg.data_root = resolve_dataset_root(cfg)
    apply_dataset_profile(cfg)
    dataset = build_dataset(cfg, split=DATASET_SPLIT, training=False)

    samples = _load_samples(dataset, SAMPLE_KEYS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rendering predictions for {len(model_specs)} model(s).")
    for idx, model_spec in enumerate(model_specs, start=1):
        print(f"[{idx}] {model_spec.run_dir}")
    predictions = _predict_panel_rows(model_specs=model_specs, samples=samples, dataset=dataset, device=device)

    panel = _build_canvas(samples=samples, predictions=predictions)

    output_path = OUTPUT_PATH.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
