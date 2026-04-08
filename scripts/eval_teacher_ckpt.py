import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import TrainConfig
from src.datasets.factory import apply_dataset_profile, build_dataset, resolve_dataset_root
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_CKPT = PROJECT_ROOT / "outputs" / "Teacher_Baseline_large" / "checkpoints" / "best.pth"


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


def build_teacher_model(cfg: TrainConfig, ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, str | int | bool]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint format: expected state_dict, got {type(state)}")

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

    num_classes = int(state["classifier.weight"].shape[0]) if "classifier.weight" in state else int(cfg.num_classes)

    model = DeepLabV3Plus(
        num_classes=num_classes,
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
    }
    return model, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a teacher checkpoint on a segmentation dataset split.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_TEACHER_CKPT,
        help=f"Path to teacher checkpoint (default: {DEFAULT_TEACHER_CKPT}).",
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    cfg = TrainConfig()
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

    model, info = build_teacher_model(cfg, args.ckpt, device)
    metrics = compute_segmentation_metrics(
        model=model,
        loader=loader,
        device=device,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )

    class_names = list(ds.meta.class_names)
    print("[INFO] Teacher checkpoint evaluation")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] dataset={cfg.dataset_name}")
    print(f"[INFO] data_root={cfg.data_root}")
    print(f"[INFO] split={args.split}")
    print(f"[INFO] num_samples={len(ds)} batch_size={args.batch_size} device={device}")
    print(
        "[INFO] model="
        f"head={info['segmentation_head']} variant={info['hybrid_variant']} "
        f"strip={info['hybrid_use_strip']} upsample={info['decoder_upsample_mode']}"
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

    print(
        "[METRIC] "
        f"mIoU={float(metrics['miou']):.6f} "
        f"BF1={float(metrics['boundary_fscore']):.6f} "
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
