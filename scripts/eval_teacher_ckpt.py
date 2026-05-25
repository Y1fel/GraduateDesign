import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MobileTrainConfig, TrainConfig
from src.datasets.factory import apply_dataset_profile, build_dataset, resolve_dataset_root
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.deeplabv3_plus_moblie import DeepLabV3PlusMobile


DEFAULT_TEACHER_CKPT = PROJECT_ROOT / "outputs" / "Teacher_Baseline_large" / "checkpoints" / "best.pth"
DEFAULT_MOBILE_CKPT = PROJECT_ROOT / "outputs" / "waste_Stu" / "notrained_Student_Baseline" / "checkpoints" / "best.pth"


def load_checkpoint_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint format: expected state_dict, got {type(state)}")
    return state


def detect_checkpoint_model_type(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("backbone.features.") for k in state.keys()):
        return "mobile"
    return "teacher"


def detect_teacher_segmentation_head(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.") for k in state.keys()):
        return "hybrid"
    return "aspp"


def detect_teacher_hybrid_variant(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.mid_kernel_branch.") or k == "hybrid_neck.mid_scale_logit" for k in state.keys()):
        return "large_v3"
    return "large"


def detect_teacher_use_strip(state: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("hybrid_neck.strip_branch.") for k in state.keys())


def detect_decoder_upsample_mode(state: dict[str, torch.Tensor]) -> str:
    if any(
        k.startswith("decoder.aspp_upsample.pre.") or k.startswith("decoder.aspp_upsample.post.")
        for k in state.keys()
    ):
        return "learnable"
    return "bilinear"


def infer_kernel_from_weight(
    state: dict[str, torch.Tensor],
    key: str,
    default: int,
) -> int:
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


def infer_residual_channels(state: dict[str, torch.Tensor], default: int = 128) -> int:
    weight = state.get("hybrid_neck.large_kernel_branch.pre.0.weight")
    if weight is None or weight.ndim < 4:
        return int(default)
    return int(weight.shape[0])


def infer_num_classes(state: dict[str, torch.Tensor], default: int) -> int:
    weight = state.get("classifier.weight")
    if weight is None or weight.ndim < 4:
        return int(default)
    return int(weight.shape[0])


def infer_mobile_output_stride(state: dict[str, torch.Tensor], default: int = 16) -> int:
    weight = state.get("aux_classifier.weight")
    if weight is None or weight.ndim < 4:
        return int(default)
    in_channels = int(weight.shape[1])
    if in_channels == 96:
        return 16
    if in_channels == 1280:
        return 32
    return int(default)


def infer_first_conv_out_channels(state: dict[str, torch.Tensor], keys: list[str], default: int) -> int:
    for key in keys:
        weight = state.get(key)
        if weight is not None and weight.ndim >= 4:
            return int(weight.shape[0])
    return int(default)


def infer_classifier_in_channels(state: dict[str, torch.Tensor], default: int) -> int:
    weight = state.get("classifier.weight")
    if weight is None or weight.ndim < 4:
        return int(default)
    return int(weight.shape[1])


def build_teacher_model(
    cfg: TrainConfig,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, str | int | bool | list[str]]]:
    segmentation_head = detect_teacher_segmentation_head(state)
    hybrid_variant = detect_teacher_hybrid_variant(state) if segmentation_head == "hybrid" else "large"
    hybrid_use_strip = detect_teacher_use_strip(state) if segmentation_head == "hybrid" else False
    decoder_upsample_mode = detect_decoder_upsample_mode(state)
    has_large_gate = any(k.startswith("hybrid_neck.large_gate.") for k in state.keys())
    has_mid_gate = any(k.startswith("hybrid_neck.mid_gate.") for k in state.keys())
    has_strip_gate = any(k.startswith("hybrid_neck.strip_gate.") for k in state.keys())
    residual_channels = infer_residual_channels(state, default=cfg.hybrid_residual_channels)
    hybrid_strip_kernel = infer_kernel_from_weight(
        state,
        "hybrid_neck.strip_branch.horizontal.block.0.0.weight",
        cfg.hybrid_strip_kernel,
    )
    hybrid_mid_kernel = infer_kernel_from_weight(
        state,
        "hybrid_neck.mid_kernel_branch.large_kernel.block.0.0.weight",
        cfg.hybrid_mid_kernel,
    )
    hybrid_large_kernel = infer_kernel_from_weight(
        state,
        "hybrid_neck.large_kernel_branch.large_kernel.block.0.0.weight",
        cfg.hybrid_large_kernel,
    )

    ckpt_num_classes = infer_num_classes(state, default=cfg.num_classes)
    if ckpt_num_classes != int(cfg.num_classes):
        raise ValueError(
            f"Checkpoint num_classes={ckpt_num_classes} does not match dataset num_classes={cfg.num_classes}."
        )

    model = DeepLabV3Plus(
        num_classes=ckpt_num_classes,
        backbone_pretrained=False,
        backbone_name=cfg.backbone_name,
        output_stride=cfg.output_stride,
        segmentation_head=segmentation_head,
        aspp_dropout=cfg.aspp_dropout,
        hybrid_variant=hybrid_variant,
        hybrid_use_strip=hybrid_use_strip,
        hybrid_strip_kernel=hybrid_strip_kernel,
        hybrid_mid_kernel=hybrid_mid_kernel,
        hybrid_large_kernel=hybrid_large_kernel,
        hybrid_gate_reduction=cfg.hybrid_gate_reduction,
        hybrid_residual_channels=residual_channels,
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

    model_state = model.state_dict()
    compatible_state = {
        k: v
        for k, v in state.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    model = model.to(device)
    model.eval()
    info = {
        "model_type": "teacher",
        "segmentation_head": segmentation_head,
        "hybrid_variant": hybrid_variant,
        "hybrid_use_strip": hybrid_use_strip,
        "hybrid_strip_kernel": hybrid_strip_kernel,
        "hybrid_mid_kernel": hybrid_mid_kernel,
        "hybrid_large_kernel": hybrid_large_kernel,
        "decoder_upsample_mode": decoder_upsample_mode,
        "residual_channels": residual_channels,
        "compat_missing": list(missing),
        "compat_unexpected": list(unexpected),
        "has_large_gate": has_large_gate,
        "output_stride": int(cfg.output_stride),
    }
    return model, info


def build_mobile_model(
    cfg: MobileTrainConfig,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, str | int]]:
    output_stride = infer_mobile_output_stride(state, default=cfg.output_stride)
    decoder_upsample_mode = detect_decoder_upsample_mode(state)
    ckpt_num_classes = infer_num_classes(state, default=cfg.num_classes)
    if ckpt_num_classes != int(cfg.num_classes):
        raise ValueError(
            f"Checkpoint num_classes={ckpt_num_classes} does not match dataset num_classes={cfg.num_classes}."
        )

    aspp_out_channels = infer_first_conv_out_channels(
        state,
        keys=["aspp.project.0.0.weight", "aspp.project.0.weight"],
        default=256,
    )
    decoder_channels = infer_classifier_in_channels(state, default=256)

    model = DeepLabV3PlusMobile(
        num_classes=ckpt_num_classes,
        output_stride=output_stride,
        backbone_pretrained=False,
        aspp_out_channels=aspp_out_channels,
        decoder_channels=decoder_channels,
        aspp_dropout=cfg.aspp_dropout,
        decoder_upsample_mode=decoder_upsample_mode,
        decoder_dropout=cfg.decoder_dropout,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    info = {
        "model_type": "mobile",
        "output_stride": output_stride,
        "decoder_upsample_mode": decoder_upsample_mode,
        "aspp_out_channels": aspp_out_channels,
        "decoder_channels": decoder_channels,
    }
    return model, info


def build_model(
    cfg: TrainConfig | MobileTrainConfig,
    state: dict[str, torch.Tensor],
    device: torch.device,
    model_type: str,
) -> tuple[torch.nn.Module, dict[str, str | int | bool | list[str]]]:
    if model_type == "teacher":
        return build_teacher_model(cfg, state, device)
    if model_type == "mobile":
        return build_mobile_model(cfg, state, device)
    raise ValueError(f"Unsupported model_type={model_type}. Use teacher/mobile/auto.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint on a dataset split.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=(
            "Path to checkpoint. "
            f"Defaults: teacher={DEFAULT_TEACHER_CKPT}, mobile={DEFAULT_MOBILE_CKPT}."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "teacher", "mobile"],
        help="Checkpoint model type. Use auto to infer from the state_dict.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti_semantic",
        choices=["cityscapes", "camvid", "kitti_semantic"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset root.")
    parser.add_argument("--batch-size", type=int, default=4, help="Eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Eval dataloader workers.")
    return parser.parse_args()


def resolve_ckpt_path(args: argparse.Namespace) -> Path:
    if args.ckpt is not None:
        return Path(args.ckpt)
    if args.model_type == "mobile":
        return DEFAULT_MOBILE_CKPT
    return DEFAULT_TEACHER_CKPT


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    ckpt_path = resolve_ckpt_path(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = load_checkpoint_state(ckpt_path)
    detected_model_type = detect_checkpoint_model_type(state)
    if args.model_type != "auto" and args.model_type != detected_model_type:
        raise ValueError(
            f"--model-type={args.model_type} does not match checkpoint structure; detected {detected_model_type}."
        )
    resolved_model_type = detected_model_type if args.model_type == "auto" else args.model_type

    cfg: TrainConfig | MobileTrainConfig
    cfg = TrainConfig() if resolved_model_type == "teacher" else MobileTrainConfig()
    cfg.dataset_name = args.dataset
    apply_dataset_profile(cfg)
    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)
    cfg.data_root = resolve_dataset_root(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(cfg, split=args.split, training=False)
    cfg.num_classes = int(ds.meta.num_classes)

    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if args.num_workers > 0 else None),
    )

    model, info = build_model(cfg, state, device, resolved_model_type)
    metrics = compute_segmentation_metrics(
        model=model,
        loader=loader,
        device=device,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )

    class_names = list(ds.meta.class_names)
    macc = float(np.nanmean(np.asarray(metrics["precision_per_class"], dtype=np.float64)))

    print("[INFO] Segmentation checkpoint evaluation")
    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] model_type={resolved_model_type} detected={detected_model_type}")
    print(f"[INFO] dataset={cfg.dataset_name}")
    print(f"[INFO] data_root={cfg.data_root}")
    print(f"[INFO] split={args.split}")
    print(f"[INFO] num_samples={len(ds)} batch_size={args.batch_size} device={device}")

    if resolved_model_type == "teacher":
        print(
            "[INFO] model="
            f"head={info['segmentation_head']} variant={info['hybrid_variant']} "
            f"strip={info['hybrid_use_strip']} upsample={info['decoder_upsample_mode']} "
            f"output_stride={info['output_stride']}"
        )
        if info["segmentation_head"] == "hybrid":
            print(
                "[INFO] hybrid="
                f"strip_k={info['hybrid_strip_kernel']} mid_k={info['hybrid_mid_kernel']} "
                f"large_k={info['hybrid_large_kernel']} residual_channels={info['residual_channels']}"
            )
            if not bool(info["has_large_gate"]):
                print("[INFO] checkpoint uses legacy ungated large branch compatibility path")
        if info["compat_missing"] or info["compat_unexpected"]:
            print(f"[INFO] load_compat missing={info['compat_missing']} unexpected={info['compat_unexpected']}")
    else:
        print(
            "[INFO] model="
            f"output_stride={info['output_stride']} upsample={info['decoder_upsample_mode']} "
            f"aspp_channels={info['aspp_out_channels']} decoder_channels={info['decoder_channels']}"
        )

    print(
        "[METRIC] "
        f"mIoU={float(metrics['miou']):.6f} "
        f"BF1={float(metrics['boundary_fscore']):.6f} "
        f"mACC={macc:.6f} "
        f"TrimapIoU={float(metrics['trimap_iou']):.6f}"
    )
    print("[PER-CLASS] class_id class_name iou precision recall")
    for class_id, class_name in enumerate(class_names):
        iou_val = float(metrics["iou_per_class"][class_id])
        precision_val = float(metrics["precision_per_class"][class_id])
        recall_val = float(metrics["recall_per_class"][class_id])
        print(
            f"[PER-CLASS] {class_id:02d} {class_name:<18} "
            f"iou={iou_val:.6f} precision={precision_val:.6f} recall={recall_val:.6f}"
        )


if __name__ == "__main__":
    main()
