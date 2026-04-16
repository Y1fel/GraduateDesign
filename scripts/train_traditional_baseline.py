from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from tqdm import tqdm

from config.config import TrainConfig
from src.datasets.cityscapes_labels import CITYSCAPES_19_CLASS_NAMES, CITYSCAPES_19_ID2COLOR, CITYSCAPES_34_TO_19
from src.eval.mIoU import boundary_fscore, trimap_iou, update_confusion_matrix

try:
    from skimage.color import rgb2gray, rgb2hsv, rgb2lab
    from skimage.filters import sobel, sobel_h, sobel_v
    from skimage.segmentation import felzenszwalb, slic
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'scikit-image'. Install it first, e.g. `python -m pip install scikit-image`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TraditionalBaselineConfig:
    cityscapes_root: Path = TrainConfig().cityscapes_root
    outputs_root: Path = PROJECT_ROOT / "outputs"
    train_split: str = "train"
    val_split: str = "val"
    ignore_index: int = 255
    num_classes: int = 19

    resize_width: int = 1024
    max_train_images: int = 0
    max_val_images: int = 0
    max_regions_per_image: int = 512
    save_vis_count: int = 8
    random_state: int = 42

    segmenter: str = "slic"
    n_segments: int = 900
    compactness: float = 12.0
    sigma: float = 1.0
    enforce_connectivity: bool = True
    felzenszwalb_scale: float = 180.0
    felzenszwalb_sigma: float = 0.8
    felzenszwalb_min_size: int = 60

    classifier: str = "extratrees"
    trees: int = 500
    max_depth: int = 28
    min_samples_leaf: int = 1
    max_features: str = "sqrt"

    texton_clusters: int = 32
    texton_batch_size: int = 4096
    max_texton_pixels: int = 250000
    grad_bins: int = 8

    graph_lambda: float = 0.45
    graph_color_sigma: float = 0.12
    graph_boundary_gamma: float = 2.0
    graph_iterations: int = 5


def parse_args() -> TraditionalBaselineConfig:
    parser = argparse.ArgumentParser(
        description="Traditional Cityscapes baseline: superpixels + texton features + tree classifier + graph smoothing."
    )
    parser.add_argument("--cityscapes-root", type=Path, default=TrainConfig().cityscapes_root)
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--val-split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--resize-width", type=int, default=1024)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-regions-per-image", type=int, default=512)
    parser.add_argument("--save-vis-count", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--segmenter", type=str, default="slic", choices=["slic", "felzenszwalb"])
    parser.add_argument("--n-segments", type=int, default=900)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--felzenszwalb-scale", type=float, default=180.0)
    parser.add_argument("--felzenszwalb-sigma", type=float, default=0.8)
    parser.add_argument("--felzenszwalb-min-size", type=int, default=60)

    parser.add_argument("--classifier", type=str, default="extratrees", choices=["extratrees", "rf"])
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=28)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")

    parser.add_argument("--texton-clusters", type=int, default=32)
    parser.add_argument("--texton-batch-size", type=int, default=4096)
    parser.add_argument("--max-texton-pixels", type=int, default=250000)
    parser.add_argument("--grad-bins", type=int, default=8)

    parser.add_argument("--graph-lambda", type=float, default=0.45)
    parser.add_argument("--graph-color-sigma", type=float, default=0.12)
    parser.add_argument("--graph-boundary-gamma", type=float, default=2.0)
    parser.add_argument("--graph-iterations", type=int, default=5)
    args = parser.parse_args()

    return TraditionalBaselineConfig(
        cityscapes_root=args.cityscapes_root,
        outputs_root=args.outputs_root,
        train_split=args.train_split,
        val_split=args.val_split,
        resize_width=args.resize_width,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        max_regions_per_image=args.max_regions_per_image,
        save_vis_count=args.save_vis_count,
        random_state=args.random_state,
        segmenter=args.segmenter,
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=args.sigma,
        felzenszwalb_scale=args.felzenszwalb_scale,
        felzenszwalb_sigma=args.felzenszwalb_sigma,
        felzenszwalb_min_size=args.felzenszwalb_min_size,
        classifier=args.classifier,
        trees=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        texton_clusters=args.texton_clusters,
        texton_batch_size=args.texton_batch_size,
        max_texton_pixels=args.max_texton_pixels,
        grad_bins=args.grad_bins,
        graph_lambda=args.graph_lambda,
        graph_color_sigma=args.graph_color_sigma,
        graph_boundary_gamma=args.graph_boundary_gamma,
        graph_iterations=args.graph_iterations,
    )


def make_run_dirs(cfg: TraditionalBaselineConfig) -> dict[str, Path]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.outputs_root / f"cityscapes_traditional_{cfg.segmenter}_{cfg.classifier}_graph_{ts}"
    log_dir = run_dir / "logs"
    ckpt_dir = run_dir / "checkpoints"
    vis_dir = run_dir / "visualizations"
    for path in (run_dir, log_dir, ckpt_dir, vis_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "vis_dir": vis_dir,
        "config_json": run_dir / "config.json",
        "metrics_json": log_dir / "metrics.json",
        "per_class_csv": log_dir / "per_class_metrics.csv",
        "model_path": ckpt_dir / "traditional_model.joblib",
    }


def collect_cityscapes_pairs(root: Path, split: str) -> list[tuple[Path, Path, str]]:
    images_root = root / "leftImg8bit" / split
    labels_root = root / "gtFine" / split
    pairs: list[tuple[Path, Path, str]] = []
    for img_path in sorted(images_root.glob("*/*_leftImg8bit.png")):
        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "")
        mask_path = labels_root / city / f"{stem}_gtFine_labelIds.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Cityscapes mask not found: {mask_path}")
        pairs.append((img_path, mask_path, f"{city}/{img_path.name}"))
    if not pairs:
        raise RuntimeError(f"No Cityscapes samples found under {images_root}")
    return pairs


def load_cityscapes_pair(
    img_path: Path,
    mask_path: Path,
    cfg: TraditionalBaselineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    mask_raw = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    valid = mask_raw <= 33
    mask = np.full(mask_raw.shape, fill_value=cfg.ignore_index, dtype=np.uint8)
    mask[valid] = np.asarray(CITYSCAPES_34_TO_19, dtype=np.uint8)[mask_raw[valid]]

    if cfg.resize_width > 0 and image.shape[1] != cfg.resize_width:
        scale = float(cfg.resize_width) / float(image.shape[1])
        resize_h = max(1, int(round(image.shape[0] * scale)))
        image = np.asarray(Image.fromarray(image).resize((cfg.resize_width, resize_h), resample=Image.BILINEAR), dtype=np.uint8)
        mask = np.asarray(
            Image.fromarray(mask, mode="L").resize((cfg.resize_width, resize_h), resample=Image.NEAREST),
            dtype=np.uint8,
        )
    return image, mask


def compute_segments(image: np.ndarray, cfg: TraditionalBaselineConfig) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    if cfg.segmenter == "slic":
        segments = slic(
            image_f,
            n_segments=int(cfg.n_segments),
            compactness=float(cfg.compactness),
            sigma=float(cfg.sigma),
            start_label=0,
            channel_axis=-1,
            convert2lab=True,
            enforce_connectivity=bool(cfg.enforce_connectivity),
        )
    else:
        segments = felzenszwalb(
            image_f,
            scale=float(cfg.felzenszwalb_scale),
            sigma=float(cfg.felzenszwalb_sigma),
            min_size=int(cfg.felzenszwalb_min_size),
            channel_axis=-1,
        )
    return segments.astype(np.int32, copy=False)


def _pixel_descriptors(image_f: np.ndarray) -> np.ndarray:
    lab = rgb2lab(image_f).astype(np.float32)
    hsv = rgb2hsv(image_f).astype(np.float32)
    gray = rgb2gray(image_f).astype(np.float32)
    gx = sobel_h(gray).astype(np.float32)
    gy = sobel_v(gray).astype(np.float32)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    edge = sobel(gray).astype(np.float32)
    return np.concatenate(
        [
            lab,
            hsv,
            gx[..., None],
            gy[..., None],
            mag[..., None],
            edge[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def fit_texton_codebook(
    cfg: TraditionalBaselineConfig,
    train_pairs: list[tuple[Path, Path, str]],
) -> MiniBatchKMeans | None:
    if int(cfg.texton_clusters) <= 0:
        return None

    rng = np.random.default_rng(cfg.random_state)
    sample_target = max(int(cfg.max_texton_pixels), int(cfg.texton_clusters) * 200)
    per_image = max(512, sample_target // max(len(train_pairs), 1))
    samples: list[np.ndarray] = []
    total = 0

    for img_path, mask_path, _rel_name in tqdm(train_pairs, desc="texton-sample", leave=False):
        image, _mask = load_cityscapes_pair(img_path, mask_path, cfg)
        desc = _pixel_descriptors(image.astype(np.float32) / 255.0).reshape(-1, 10)
        if desc.shape[0] > per_image:
            keep = rng.choice(desc.shape[0], size=per_image, replace=False)
            desc = desc[keep]
        samples.append(desc)
        total += int(desc.shape[0])
        if total >= sample_target:
            break

    if not samples:
        return None

    x = np.concatenate(samples, axis=0)
    if x.shape[0] > sample_target:
        keep = rng.choice(x.shape[0], size=sample_target, replace=False)
        x = x[keep]

    codebook = MiniBatchKMeans(
        n_clusters=int(cfg.texton_clusters),
        batch_size=int(cfg.texton_batch_size),
        n_init=5,
        max_iter=200,
        reassignment_ratio=0.01,
        random_state=int(cfg.random_state),
    )
    codebook.fit(x)
    return codebook


def _bincount_mean(values: np.ndarray, seg_ids: np.ndarray, counts: np.ndarray, n_segments: int) -> np.ndarray:
    means = np.zeros((n_segments, values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        means[:, dim] = np.bincount(seg_ids, weights=values[:, dim], minlength=n_segments).astype(np.float32) / counts
    return means


def _bincount_std(
    values: np.ndarray,
    seg_ids: np.ndarray,
    counts: np.ndarray,
    means: np.ndarray,
    n_segments: int,
) -> np.ndarray:
    stds = np.zeros((n_segments, values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        sq = np.bincount(seg_ids, weights=values[:, dim] ** 2, minlength=n_segments).astype(np.float32) / counts
        stds[:, dim] = np.sqrt(np.maximum(sq - means[:, dim] ** 2, 0.0), dtype=np.float32)
    return stds


def build_region_adjacency(
    segments: np.ndarray,
    edge_map: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    h, w = segments.shape
    n_segments = int(segments.max()) + 1
    pair_stats: dict[tuple[int, int], list[float]] = {}

    right_diff = segments[:, 1:] != segments[:, :-1]
    if np.any(right_diff):
        a = segments[:, :-1][right_diff].astype(np.int64)
        b = segments[:, 1:][right_diff].astype(np.int64)
        e = 0.5 * (edge_map[:, :-1][right_diff] + edge_map[:, 1:][right_diff])
        for u, v, edge_val in zip(a, b, e, strict=False):
            i, j = (int(u), int(v)) if u < v else (int(v), int(u))
            stats = pair_stats.setdefault((i, j), [0.0, 0.0])
            stats[0] += 1.0
            stats[1] += float(edge_val)

    down_diff = segments[1:, :] != segments[:-1, :]
    if np.any(down_diff):
        a = segments[:-1, :][down_diff].astype(np.int64)
        b = segments[1:, :][down_diff].astype(np.int64)
        e = 0.5 * (edge_map[:-1, :][down_diff] + edge_map[1:, :][down_diff])
        for u, v, edge_val in zip(a, b, e, strict=False):
            i, j = (int(u), int(v)) if u < v else (int(v), int(u))
            stats = pair_stats.setdefault((i, j), [0.0, 0.0])
            stats[0] += 1.0
            stats[1] += float(edge_val)

    neigh_idx: list[list[int]] = [[] for _ in range(n_segments)]
    neigh_meta: list[list[float]] = [[] for _ in range(n_segments)]
    norm = float(max(h, w))
    for (i, j), (shared_len, edge_sum) in pair_stats.items():
        boundary_mean = edge_sum / max(shared_len, 1.0)
        shared_weight = shared_len / norm
        neigh_idx[i].append(j)
        neigh_meta[i].append([shared_weight, boundary_mean])
        neigh_idx[j].append(i)
        neigh_meta[j].append([shared_weight, boundary_mean])

    idx_np = [np.asarray(v, dtype=np.int32) if v else np.empty((0,), dtype=np.int32) for v in neigh_idx]
    meta_np = [np.asarray(v, dtype=np.float32) if v else np.empty((0, 2), dtype=np.float32) for v in neigh_meta]
    return idx_np, meta_np


def extract_region_features(
    image: np.ndarray,
    segments: np.ndarray,
    codebook: MiniBatchKMeans | None,
    cfg: TraditionalBaselineConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | list[np.ndarray]]]:
    image_f = image.astype(np.float32) / 255.0
    rgb = image_f
    lab = rgb2lab(image_f).astype(np.float32)
    hsv = rgb2hsv(image_f).astype(np.float32)
    gray = rgb2gray(image_f).astype(np.float32)
    edge = sobel(gray).astype(np.float32)
    gx = sobel_h(gray).astype(np.float32)
    gy = sobel_v(gray).astype(np.float32)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    grad_ori = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)

    seg_ids = segments.reshape(-1).astype(np.int64)
    n_segments = int(seg_ids.max()) + 1
    counts = np.bincount(seg_ids, minlength=n_segments).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    rgb_flat = rgb.reshape(-1, 3)
    lab_flat = lab.reshape(-1, 3)
    hsv_flat = hsv.reshape(-1, 3)
    edge_flat = edge.reshape(-1, 1)
    grad_flat = grad_mag.reshape(-1, 1)

    rgb_mean = _bincount_mean(rgb_flat, seg_ids, counts, n_segments)
    rgb_std = _bincount_std(rgb_flat, seg_ids, counts, rgb_mean, n_segments)
    lab_mean = _bincount_mean(lab_flat, seg_ids, counts, n_segments)
    lab_std = _bincount_std(lab_flat, seg_ids, counts, lab_mean, n_segments)
    hsv_mean = _bincount_mean(hsv_flat, seg_ids, counts, n_segments)
    hsv_std = _bincount_std(hsv_flat, seg_ids, counts, hsv_mean, n_segments)
    edge_mean = _bincount_mean(edge_flat, seg_ids, counts, n_segments)
    edge_std = _bincount_std(edge_flat, seg_ids, counts, edge_mean, n_segments)
    grad_mean = _bincount_mean(grad_flat, seg_ids, counts, n_segments)
    grad_std = _bincount_std(grad_flat, seg_ids, counts, grad_mean, n_segments)

    yy, xx = np.mgrid[0:segments.shape[0], 0:segments.shape[1]]
    yy_flat = yy.reshape(-1).astype(np.float32)
    xx_flat = xx.reshape(-1).astype(np.float32)
    cy = np.bincount(seg_ids, weights=yy_flat, minlength=n_segments).astype(np.float32) / counts
    cx = np.bincount(seg_ids, weights=xx_flat, minlength=n_segments).astype(np.float32) / counts
    centroid = np.stack([cy / max(segments.shape[0] - 1, 1), cx / max(segments.shape[1] - 1, 1)], axis=1)
    area_ratio = (counts / float(segments.shape[0] * segments.shape[1]))[:, None]

    min_y = np.full((n_segments,), segments.shape[0], dtype=np.int32)
    max_y = np.zeros((n_segments,), dtype=np.int32)
    min_x = np.full((n_segments,), segments.shape[1], dtype=np.int32)
    max_x = np.zeros((n_segments,), dtype=np.int32)
    np.minimum.at(min_y, seg_ids, yy_flat.astype(np.int32))
    np.maximum.at(max_y, seg_ids, yy_flat.astype(np.int32))
    np.minimum.at(min_x, seg_ids, xx_flat.astype(np.int32))
    np.maximum.at(max_x, seg_ids, xx_flat.astype(np.int32))
    bbox_h = (max_y - min_y + 1).astype(np.float32) / max(segments.shape[0], 1)
    bbox_w = (max_x - min_x + 1).astype(np.float32) / max(segments.shape[1], 1)
    bbox_area = np.maximum((max_y - min_y + 1) * (max_x - min_x + 1), 1).astype(np.float32)
    fill_ratio = (counts / bbox_area)[:, None]
    aspect = (bbox_w / np.maximum(bbox_h, 1e-6))[:, None]
    shape_feat = np.concatenate([bbox_h[:, None], bbox_w[:, None], aspect, fill_ratio], axis=1).astype(np.float32)

    ori_hist = np.zeros((n_segments, int(cfg.grad_bins)), dtype=np.float32)
    ori_bins = np.floor(grad_ori.reshape(-1) * float(cfg.grad_bins)).astype(np.int64)
    ori_bins = np.clip(ori_bins, 0, int(cfg.grad_bins) - 1)
    mag_flat = grad_mag.reshape(-1).astype(np.float32)
    for bin_id in range(int(cfg.grad_bins)):
        hit = ori_bins == bin_id
        if np.any(hit):
            ori_hist[:, bin_id] = np.bincount(seg_ids[hit], weights=mag_flat[hit], minlength=n_segments).astype(np.float32)
    ori_hist = ori_hist / np.maximum(ori_hist.sum(axis=1, keepdims=True), 1e-6)

    if codebook is not None:
        texton_ids = codebook.predict(_pixel_descriptors(image_f).reshape(-1, 10)).astype(np.int64)
        texton_hist = np.zeros((n_segments, int(cfg.texton_clusters)), dtype=np.float32)
        for texton_id in range(int(cfg.texton_clusters)):
            hit = texton_ids == texton_id
            if np.any(hit):
                texton_hist[:, texton_id] = np.bincount(seg_ids[hit], minlength=n_segments).astype(np.float32)
        texton_hist = texton_hist / np.maximum(texton_hist.sum(axis=1, keepdims=True), 1e-6)
    else:
        texton_hist = np.empty((n_segments, 0), dtype=np.float32)

    neigh_idx, neigh_meta = build_region_adjacency(segments, edge)
    neighbor_lab = np.zeros((n_segments, 3), dtype=np.float32)
    neighbor_edge = np.zeros((n_segments, 1), dtype=np.float32)
    for region_id in range(n_segments):
        idx = neigh_idx[region_id]
        meta = neigh_meta[region_id]
        if idx.size == 0:
            continue
        weights = meta[:, 0]
        weights = weights / np.maximum(weights.sum(), 1e-6)
        neighbor_lab[region_id] = np.sum(lab_mean[idx] * weights[:, None], axis=0)
        neighbor_edge[region_id, 0] = float(np.sum(edge_mean[idx, 0] * weights))
    neighbor_context = np.concatenate([neighbor_lab - lab_mean, neighbor_edge - edge_mean], axis=1)

    features = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            lab_mean,
            lab_std,
            hsv_mean,
            hsv_std,
            edge_mean,
            edge_std,
            grad_mean,
            grad_std,
            centroid.astype(np.float32),
            area_ratio.astype(np.float32),
            shape_feat,
            ori_hist.astype(np.float32),
            texton_hist.astype(np.float32),
            neighbor_context.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    graph_ctx = {
        "lab_mean": lab_mean.astype(np.float32),
        "neigh_idx": neigh_idx,
        "neigh_meta": neigh_meta,
    }
    return features, graph_ctx


def assign_region_labels(
    segments: np.ndarray,
    mask: np.ndarray,
    cfg: TraditionalBaselineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    seg_ids = segments.reshape(-1).astype(np.int64)
    mask_flat = mask.reshape(-1).astype(np.int64)
    n_segments = int(seg_ids.max()) + 1
    label_counts = np.zeros((n_segments, cfg.num_classes), dtype=np.int32)
    for class_id in range(cfg.num_classes):
        hit = mask_flat == class_id
        if np.any(hit):
            label_counts[:, class_id] = np.bincount(seg_ids[hit], minlength=n_segments).astype(np.int32)
    labels = label_counts.argmax(axis=1).astype(np.int64)
    valid = label_counts.sum(axis=1) > 0
    ignore_hit = np.bincount(seg_ids[mask_flat == cfg.ignore_index], minlength=n_segments) if np.any(mask_flat == cfg.ignore_index) else 0
    if isinstance(ignore_hit, np.ndarray):
        valid &= ignore_hit < np.bincount(seg_ids, minlength=n_segments)
    return labels, valid


def subsample_regions(
    features: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    max_regions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.flatnonzero(valid)
    if keep.size == 0:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    if max_regions > 0 and keep.size > max_regions:
        keep = rng.choice(keep, size=max_regions, replace=False)
        keep.sort()
    return features[keep], labels[keep]


def build_classifier(cfg: TraditionalBaselineConfig):
    common_kwargs = dict(
        n_estimators=int(cfg.trees),
        max_depth=(None if cfg.max_depth <= 0 else int(cfg.max_depth)),
        min_samples_leaf=int(cfg.min_samples_leaf),
        max_features=cfg.max_features,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(cfg.random_state),
        verbose=0,
    )
    if cfg.classifier == "extratrees":
        return ExtraTreesClassifier(**common_kwargs)
    return RandomForestClassifier(**common_kwargs)


def smooth_region_predictions(
    prob: np.ndarray,
    graph_ctx: dict[str, np.ndarray | list[np.ndarray]],
    cfg: TraditionalBaselineConfig,
) -> np.ndarray:
    unary = -np.log(np.maximum(prob.astype(np.float64), 1e-8))
    labels = unary.argmin(axis=1).astype(np.int64)
    lab_mean = np.asarray(graph_ctx["lab_mean"], dtype=np.float32)
    neigh_idx: list[np.ndarray] = graph_ctx["neigh_idx"]  # type: ignore[assignment]
    neigh_meta: list[np.ndarray] = graph_ctx["neigh_meta"]  # type: ignore[assignment]

    for _ in range(max(int(cfg.graph_iterations), 0)):
        changed = False
        for region_id in range(unary.shape[0]):
            idx = neigh_idx[region_id]
            meta = neigh_meta[region_id]
            if idx.size == 0:
                continue

            color_diff = lab_mean[idx] - lab_mean[region_id]
            color_dist2 = np.sum(color_diff * color_diff, axis=1)
            color_term = np.exp(-color_dist2 / max(float(cfg.graph_color_sigma) ** 2, 1e-6))
            shared = meta[:, 0]
            boundary_mean = meta[:, 1]
            smooth_w = shared * color_term / (1.0 + float(cfg.graph_boundary_gamma) * boundary_mean)

            pairwise = np.zeros((unary.shape[1],), dtype=np.float64)
            for neigh_label, weight in zip(labels[idx], smooth_w, strict=False):
                pairwise += float(weight) * (np.arange(unary.shape[1]) != int(neigh_label))
            energy = unary[region_id] + float(cfg.graph_lambda) * pairwise
            new_label = int(np.argmin(energy))
            if new_label != int(labels[region_id]):
                labels[region_id] = new_label
                changed = True
        if not changed:
            break
    return labels


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(CITYSCAPES_19_ID2COLOR):
        color[mask == class_id] = np.asarray(rgb, dtype=np.uint8)
    return color


def save_triplet(vis_dir: Path, rel_name: str, image: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> None:
    stem = rel_name.replace("/", "__").replace("\\", "__").replace(".png", "")
    triplet = np.concatenate([image, colorize_mask(mask), colorize_mask(pred)], axis=1)
    Image.fromarray(triplet).save(vis_dir / f"{stem}.png")


def compute_metrics_from_confusion(conf: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    c = conf.detach().cpu().numpy().astype(np.float64)
    tp = np.diag(c)
    fp = c.sum(axis=0) - tp
    fn = c.sum(axis=1) - tp
    denom = tp + fp + fn

    iou = np.full(tp.shape, np.nan, dtype=np.float64)
    valid = denom > 0
    iou[valid] = tp[valid] / denom[valid]

    precision_denom = tp + fp
    precision = np.full(tp.shape, np.nan, dtype=np.float64)
    precision_valid = precision_denom > 0
    precision[precision_valid] = tp[precision_valid] / precision_denom[precision_valid]

    recall_denom = tp + fn
    recall = np.full(tp.shape, np.nan, dtype=np.float64)
    recall_valid = recall_denom > 0
    recall[recall_valid] = tp[recall_valid] / recall_denom[recall_valid]

    miou = float(np.nanmean(iou)) if np.any(valid) else float("nan")
    return miou, iou, precision, recall


def train_classifier(
    cfg: TraditionalBaselineConfig,
    train_pairs: list[tuple[Path, Path, str]],
    codebook: MiniBatchKMeans | None,
    run_paths: dict[str, Path],
):
    rng = np.random.default_rng(cfg.random_state)
    feat_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    total_regions = 0
    start = time.time()

    for img_path, mask_path, _rel_name in tqdm(train_pairs, desc="train-features", leave=False):
        image, mask = load_cityscapes_pair(img_path, mask_path, cfg)
        segments = compute_segments(image, cfg)
        features, _graph = extract_region_features(image, segments, codebook, cfg)
        labels, valid = assign_region_labels(segments, mask, cfg)
        x, y = subsample_regions(features, labels, valid, cfg.max_regions_per_image, rng)
        if y.size == 0:
            continue
        feat_list.append(x)
        label_list.append(y)
        total_regions += int(y.size)

    if not feat_list:
        raise RuntimeError("No valid training regions extracted.")

    x_train = np.concatenate(feat_list, axis=0)
    y_train = np.concatenate(label_list, axis=0)
    clf = build_classifier(cfg)
    print(f"[INFO] classifier={cfg.classifier} train_regions={total_regions} feature_dim={x_train.shape[1]}")
    clf.fit(x_train, y_train)
    print(f"[INFO] train_done time_sec={time.time() - start:.2f}")

    joblib.dump({"classifier": clf, "texton_codebook": codebook}, run_paths["model_path"])
    return clf


def evaluate_classifier(
    clf,
    codebook: MiniBatchKMeans | None,
    cfg: TraditionalBaselineConfig,
    val_pairs: list[tuple[Path, Path, str]],
    run_paths: dict[str, Path],
) -> dict[str, object]:
    conf = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.int64)
    bf_scores: list[float] = []
    trimap_scores: list[float] = []
    vis_saved = 0
    start = time.time()

    for img_path, mask_path, rel_name in tqdm(val_pairs, desc="val", leave=False):
        image, mask = load_cityscapes_pair(img_path, mask_path, cfg)
        segments = compute_segments(image, cfg)
        features, graph_ctx = extract_region_features(image, segments, codebook, cfg)
        prob = clf.predict_proba(features).astype(np.float32, copy=False)
        region_pred = smooth_region_predictions(prob, graph_ctx, cfg)
        pred = region_pred[segments].astype(np.uint8, copy=False)

        pred_t = torch.from_numpy(pred.astype(np.int64, copy=False)).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.int64, copy=False)).unsqueeze(0)
        update_confusion_matrix(conf, pred_t, mask_t, cfg.num_classes, cfg.ignore_index)
        bf, _bp, _br = boundary_fscore(pred_t, mask_t, ignore_index=cfg.ignore_index, dilation=2)
        bf_scores.append(float(bf))
        trimap_scores.append(float(trimap_iou(pred_t, mask_t, ignore_index=cfg.ignore_index, trimap_width=3)))

        if vis_saved < cfg.save_vis_count:
            save_triplet(run_paths["vis_dir"], rel_name, image, mask, pred)
            vis_saved += 1

    eval_sec = time.time() - start
    miou, iou, precision, recall = compute_metrics_from_confusion(conf)
    return {
        "miou": miou,
        "boundary_fscore": float(np.nanmean(np.asarray(bf_scores, dtype=np.float64))) if bf_scores else float("nan"),
        "trimap_iou": float(np.nanmean(np.asarray(trimap_scores, dtype=np.float64))) if trimap_scores else float("nan"),
        "iou_per_class": iou.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "eval_time_sec": eval_sec,
    }


def save_outputs(cfg: TraditionalBaselineConfig, run_paths: dict[str, Path], metrics: dict[str, object]) -> None:
    run_paths["config_json"].write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    run_paths["metrics_json"].write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with run_paths["per_class_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "iou", "precision", "recall"])
        for class_id, class_name in enumerate(CITYSCAPES_19_CLASS_NAMES):
            writer.writerow(
                [
                    class_id,
                    class_name,
                    f"{float(metrics['iou_per_class'][class_id]):.6f}",
                    f"{float(metrics['precision_per_class'][class_id]):.6f}",
                    f"{float(metrics['recall_per_class'][class_id]):.6f}",
                ]
            )


def main() -> None:
    cfg = parse_args()
    run_paths = make_run_dirs(cfg)
    print(f"[INFO] run_dir={run_paths['run_dir']}")
    print(f"[INFO] cityscapes_root={cfg.cityscapes_root}")

    train_pairs = collect_cityscapes_pairs(cfg.cityscapes_root, cfg.train_split)
    val_pairs = collect_cityscapes_pairs(cfg.cityscapes_root, cfg.val_split)
    if cfg.max_train_images > 0:
        train_pairs = train_pairs[: cfg.max_train_images]
    if cfg.max_val_images > 0:
        val_pairs = val_pairs[: cfg.max_val_images]

    print(
        "[INFO] data="
        f"train_split={cfg.train_split} train_images={len(train_pairs)} "
        f"val_split={cfg.val_split} val_images={len(val_pairs)}"
    )
    print(
        "[INFO] traditional="
        f"segmenter={cfg.segmenter} classifier={cfg.classifier} "
        f"texton_clusters={cfg.texton_clusters} graph_lambda={cfg.graph_lambda}"
    )

    codebook = fit_texton_codebook(cfg, train_pairs)
    if codebook is not None:
        print(f"[INFO] texton_codebook fitted: clusters={cfg.texton_clusters}")

    clf = train_classifier(cfg, train_pairs, codebook, run_paths)
    metrics = evaluate_classifier(clf, codebook, cfg, val_pairs, run_paths)
    save_outputs(cfg, run_paths, metrics)

    print(
        "[METRIC] "
        f"mIoU={float(metrics['miou']):.6f} "
        f"BF1={float(metrics['boundary_fscore']):.6f} "
        f"TrimapIoU={float(metrics['trimap_iou']):.6f}"
    )
    print("[PER-CLASS] class_id class_name iou precision recall")
    for class_id, class_name in enumerate(CITYSCAPES_19_CLASS_NAMES):
        print(
            f"[PER-CLASS] {class_id:02d} {class_name:<18} "
            f"iou={float(metrics['iou_per_class'][class_id]):.6f} "
            f"precision={float(metrics['precision_per_class'][class_id]):.6f} "
            f"recall={float(metrics['recall_per_class'][class_id]):.6f}"
        )


if __name__ == "__main__":
    main()
