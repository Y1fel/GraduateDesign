import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MobileTrainConfig, TrainConfig
from scripts.eval_teacher_ckpt import build_model, detect_checkpoint_model_type, load_checkpoint_state
from src.commom.constants import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.factory import apply_dataset_profile, build_dataset, normalize_dataset_name, resolve_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint inference on a segmentation split, export predictions, "
            "and record FLOPs/FPS/profile stats."
        )
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=("auto", "teacher", "mobile"),
        help="Checkpoint model type. Use auto to infer from the state_dict.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        choices=("cityscapes", "camvid", "kitti_semantic"),
        help="Dataset to run inference on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to run.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset root.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit processed samples. <=0 means all.")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations for FPS measurement.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "test_set_runs",
        help="Root directory for exported predictions and stats.",
    )
    parser.add_argument(
        "--save-color",
        dest="save_color",
        action="store_true",
        help="Save color masks.",
    )
    parser.add_argument(
        "--no-save-color",
        dest="save_color",
        action="store_false",
        help="Do not save color masks.",
    )
    parser.set_defaults(save_color=True)
    parser.add_argument(
        "--save-compare",
        dest="save_compare",
        action="store_true",
        help="Save side-by-side original/prediction images.",
    )
    parser.add_argument(
        "--no-save-compare",
        dest="save_compare",
        action="store_false",
        help="Do not save side-by-side images.",
    )
    parser.set_defaults(save_compare=True)
    parser.add_argument(
        "--save-color-max-items",
        type=int,
        default=-1,
        help="Maximum number of color masks to save. <0 means all. Default: all.",
    )
    parser.add_argument(
        "--save-compare-max-items",
        type=int,
        default=-1,
        help="Maximum number of compare images to save. <0 means all. Default: all.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable AMP even when CUDA is available.",
    )
    return parser.parse_args()


def _resolve_limit(limit: int) -> int:
    return int(limit) if int(limit) >= 0 else 10**18


def _denormalize_image_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def _colorize_pred(pred: np.ndarray, id2color: list[tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(id2color):
        color[pred == class_id] = rgb
    return color


def _human_count(value: float) -> str:
    units = ["", "K", "M", "G", "T"]
    scaled = float(value)
    unit = 0
    while abs(scaled) >= 1000.0 and unit < len(units) - 1:
        scaled /= 1000.0
        unit += 1
    return f"{scaled:.3f}{units[unit]}"


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _profile_flops(model: nn.Module, sample: torch.Tensor, device: torch.device) -> dict[str, float | int | list[int] | str]:
    flops_by_type: dict[str, float] = {}
    hooks = []

    def add_flops(name: str, value: float) -> None:
        flops_by_type[name] = flops_by_type.get(name, 0.0) + float(value)

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        x = inputs[0]
        y = _extract_tensor(output)
        if not isinstance(x, torch.Tensor) or y is None:
            return
        batch = int(y.shape[0])
        out_channels = int(y.shape[1])
        out_h = int(y.shape[2]) if y.ndim >= 3 else 1
        out_w = int(y.shape[3]) if y.ndim >= 4 else 1
        kernel_h, kernel_w = module.kernel_size
        in_per_group = int(module.in_channels // module.groups)
        mul_add = 2.0 * kernel_h * kernel_w * in_per_group
        bias_ops = 1.0 if module.bias is not None else 0.0
        add_flops("Conv2d", batch * out_channels * out_h * out_w * (mul_add + bias_ops))

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        x = inputs[0]
        y = _extract_tensor(output)
        if not isinstance(x, torch.Tensor) or y is None:
            return
        batch_mul = max(int(x.numel() // module.in_features), 1)
        bias_ops = 1.0 if module.bias is not None else 0.0
        add_flops("Linear", batch_mul * module.out_features * (2.0 * module.in_features + bias_ops))

    def bn_hook(module: nn.BatchNorm2d, inputs, output) -> None:
        y = _extract_tensor(output)
        if y is None:
            return
        add_flops("BatchNorm2d", 4.0 * y.numel())

    def relu_hook(module: nn.Module, inputs, output) -> None:
        y = _extract_tensor(output)
        if y is None:
            return
        add_flops(type(module).__name__, float(y.numel()))

    def sigmoid_hook(module: nn.Sigmoid, inputs, output) -> None:
        y = _extract_tensor(output)
        if y is None:
            return
        add_flops("Sigmoid", 4.0 * y.numel())

    def adaptive_avg_pool_hook(module: nn.AdaptiveAvgPool2d, inputs, output) -> None:
        x = inputs[0]
        y = _extract_tensor(output)
        if not isinstance(x, torch.Tensor) or y is None or x.ndim < 4 or y.ndim < 4:
            return
        in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
        out_h, out_w = int(y.shape[-2]), int(y.shape[-1])
        kernel = max((in_h * in_w) // max(out_h * out_w, 1), 1)
        add_flops("AdaptiveAvgPool2d", float(y.numel()) * kernel)

    def pool_hook(module: nn.Module, inputs, output) -> None:
        y = _extract_tensor(output)
        if y is None:
            return
        kernel_size = getattr(module, "kernel_size", 1)
        if isinstance(kernel_size, tuple):
            k_ops = int(kernel_size[0]) * int(kernel_size[1])
        else:
            k_ops = int(kernel_size) * int(kernel_size)
        add_flops(type(module).__name__, float(y.numel()) * max(k_ops, 1))

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.BatchNorm2d):
            hooks.append(mod.register_forward_hook(bn_hook))
        elif isinstance(mod, (nn.ReLU, nn.ReLU6)):
            hooks.append(mod.register_forward_hook(relu_hook))
        elif isinstance(mod, nn.Sigmoid):
            hooks.append(mod.register_forward_hook(sigmoid_hook))
        elif isinstance(mod, nn.AdaptiveAvgPool2d):
            hooks.append(mod.register_forward_hook(adaptive_avg_pool_hook))
        elif isinstance(mod, (nn.AvgPool2d, nn.MaxPool2d)):
            hooks.append(mod.register_forward_hook(pool_hook))

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        _ = model(sample.to(device))
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    total_flops = int(sum(flops_by_type.values()))
    params = int(sum(p.numel() for p in model.parameters()))
    return {
        "params": params,
        "params_human": _human_count(params),
        "flops": total_flops,
        "flops_human": _human_count(total_flops),
        "macs": int(total_flops // 2),
        "macs_human": _human_count(total_flops / 2.0),
        "input_shape": list(sample.shape),
        "flops_by_type": {k: int(v) for k, v in sorted(flops_by_type.items())},
        "note": (
            "FLOPs are hook-based and include module-level Conv2d/Linear/BatchNorm/activation/pooling ops. "
            "Tensor adds and F.interpolate are not counted."
        ),
    }


def _build_save_paths(dataset_name: str, out_dir: Path, rel_name: str) -> tuple[Path, Path, Path]:
    rel_path = Path(rel_name)
    stem = rel_path.stem
    parent = rel_path.parent
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "cityscapes":
        pred_path = out_dir / "pred_trainIds" / parent / f"{stem}_predTrainIds.png"
    else:
        pred_path = out_dir / "pred_ids" / parent / f"{stem}_pred.png"
    color_path = out_dir / "pred_color" / parent / f"{stem}_color.png"
    compare_path = out_dir / "compare" / parent / f"{stem}_compare.png"
    return pred_path, color_path, compare_path


def _build_run_dir(output_root: Path, dataset_name: str, split: str, ckpt_path: Path, model_type: str) -> Path:
    ckpt_tag = ckpt_path.stem
    if ckpt_path.parent.name == "checkpoints" and ckpt_path.parent.parent.name:
        ckpt_tag = ckpt_path.parent.parent.name
    run_name = f"{normalize_dataset_name(dataset_name)}_{split}_{ckpt_tag}_{model_type}"
    return output_root / run_name


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
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
    amp_enabled = bool(cfg.use_amp and not bool(args.disable_amp) and device.type == "cuda")

    ds = build_dataset(cfg, split=args.split, training=False)
    cfg.num_classes = int(ds.meta.num_classes)
    id2color = list(ds.meta.id2color)

    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if args.num_workers > 0 else None),
        drop_last=False,
    )

    model, model_info = build_model(cfg, state, device, resolved_model_type)
    sample_img, _sample_mask, _sample_name = ds[0]
    sample = sample_img.unsqueeze(0)
    profile_stats = _profile_flops(model, sample=sample, device=device)

    run_dir = _build_run_dir(args.output_dir, cfg.dataset_name, args.split, ckpt_path, resolved_model_type)
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_json = run_dir / "stats.json"

    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] model_type={resolved_model_type} detected={detected_model_type}")
    print(f"[INFO] dataset={cfg.dataset_name} split={args.split}")
    print(f"[INFO] data_root={cfg.data_root}")
    print(f"[INFO] device={device} amp={amp_enabled}")
    print(f"[INFO] output_dir={run_dir}")
    print(
        "[INFO] profile="
        f"params={profile_stats['params_human']} flops={profile_stats['flops_human']} "
        f"macs={profile_stats['macs_human']} input_shape={profile_stats['input_shape']}"
    )
    if resolved_model_type == "teacher":
        print(
            "[INFO] teacher_model="
            f"head={model_info['segmentation_head']} variant={model_info['hybrid_variant']} "
            f"upsample={model_info['decoder_upsample_mode']} output_stride={model_info['output_stride']}"
        )
    else:
        print(
            "[INFO] mobile_model="
            f"output_stride={model_info['output_stride']} upsample={model_info['decoder_upsample_mode']} "
            f"aspp_channels={model_info['aspp_out_channels']} decoder_channels={model_info['decoder_channels']}"
        )

    warmup_iters = max(0, int(args.warmup_iters))
    if warmup_iters > 0:
        warmup_sample = sample.to(device, non_blocking=True)
        with torch.inference_mode():
            for _ in range(warmup_iters):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    _ = model(warmup_sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
        print(f"[INFO] warmup_done iters={warmup_iters}")

    total_target = len(ds) if int(args.max_samples) <= 0 else min(len(ds), int(args.max_samples))
    total_time = 0.0
    seen = 0
    saved_color = 0
    saved_compare = 0
    color_limit = _resolve_limit(int(args.save_color_max_items))
    compare_limit = _resolve_limit(int(args.save_compare_max_items))

    for imgs, _masks, names in loader:
        if seen >= total_target:
            break

        remain = total_target - seen
        if imgs.size(0) > remain:
            imgs = imgs[:remain]
            names = names[:remain]

        imgs = imgs.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        imgs_cpu = imgs.detach().cpu()
        for idx, (pred, rel_name) in enumerate(zip(preds, names)):
            pred_path, color_path, compare_path = _build_save_paths(cfg.dataset_name, run_dir, rel_name)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred.astype(np.uint8)).save(pred_path)

            color_mask = None
            if bool(args.save_color) or bool(args.save_compare):
                color_mask = _colorize_pred(pred, id2color)

            if bool(args.save_color) and saved_color < color_limit and color_mask is not None:
                color_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(color_mask).save(color_path)
                saved_color += 1

            if bool(args.save_compare) and saved_compare < compare_limit and color_mask is not None:
                original = _denormalize_image_to_uint8(imgs_cpu[idx])
                merged = np.concatenate([original, color_mask], axis=1)
                compare_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(merged).save(compare_path)
                saved_compare += 1

        seen += len(names)
        print(f"[INFO] progress={seen}/{total_target}")

    fps = float(seen / max(total_time, 1e-12))
    avg_ms = float(total_time / max(seen, 1) * 1000.0)

    payload = {
        "ckpt": str(ckpt_path),
        "model_type": resolved_model_type,
        "detected_model_type": detected_model_type,
        "dataset": cfg.dataset_name,
        "split": args.split,
        "data_root": str(cfg.data_root),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "num_samples_total": len(ds),
        "num_samples_processed": seen,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "warmup_iters": warmup_iters,
        "infer_total_time_sec": total_time,
        "infer_avg_ms": avg_ms,
        "infer_fps": fps,
        "save_color": bool(args.save_color),
        "save_compare": bool(args.save_compare),
        "saved_color_items": saved_color,
        "saved_compare_items": saved_compare,
        "run_dir": str(run_dir),
        "profile": profile_stats,
        "model_info": model_info,
    }
    stats_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] done")
    print(f"[METRIC] infer_total_time={total_time:.4f}s")
    print(f"[METRIC] infer_avg_ms={avg_ms:.3f}")
    print(f"[METRIC] FPS={fps:.2f}")
    print(f"[METRIC] params={profile_stats['params_human']}")
    print(f"[METRIC] flops={profile_stats['flops_human']}")
    print(f"[METRIC] macs={profile_stats['macs_human']}")
    print(f"[INFO] stats_json={stats_json}")


if __name__ == "__main__":
    main()
