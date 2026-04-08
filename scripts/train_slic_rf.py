from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from config.config import TrainConfig
from src.datasets.cityscapes_labels import CITYSCAPES_19_CLASS_NAMES, CITYSCAPES_19_ID2COLOR, CITYSCAPES_34_TO_19
from src.eval.mIoU import boundary_fscore, trimap_iou, update_confusion_matrix

try:
    from skimage.color import rgb2gray, rgb2lab
    from skimage.filters import sobel
    from skimage.segmentation import slic
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency 'scikit-image'. Install it in your environment first, e.g. "
        "`python -m pip install scikit-image`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SLICRFConfig:
    cityscapes_root: Path = TrainConfig().cityscapes_root
    outputs_root: Path = PROJECT_ROOT / "outputs"
    train_split: str = "train"
    val_split: str = "val"
    ignore_index: int = 255
    num_classes: int = 19
    resize_width: int = 1024
    n_segments: int = 500
    compactness: float = 10.0
    sigma: float = 1.0
    enforce_connectivity: bool = True
    max_train_images: int = 0
    max_val_images: int = 0
    max_regions_per_image: int = 256
    rf_trees: int = 300
    rf_max_depth: int = 24
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"
    random_state: int = 42
    save_vis_count: int = 8


def parse_args() -> SLICRFConfig:
    parser = argparse.ArgumentParser(description="Traditional Cityscapes baseline: SLIC superpixels + RandomForest.")
    parser.add_argument("--cityscapes-root", type=Path, default=TrainConfig().cityscapes_root)
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--val-split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--resize-width", type=int, default=1024, help="Resize image width before SLIC. 0 keeps original.")
    parser.add_argument("--n-segments", type=int, default=500, help="Target SLIC superpixel count.")
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max-train-images", type=int, default=0, help="0 means use full split.")
    parser.add_argument("--max-val-images", type=int, default=0, help="0 means use full split.")
    parser.add_argument(
        "--max-regions-per-image",
        type=int,
        default=256,
        help="Cap sampled training superpixels per image. 0 means keep all.",
    )
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=24)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-vis-count", type=int, default=8)
    args = parser.parse_args()
    return SLICRFConfig(
        cityscapes_root=args.cityscapes_root,
        outputs_root=args.outputs_root,
        train_split=args.train_split,
        val_split=args.val_split,
        resize_width=args.resize_width,
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=args.sigma,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        max_regions_per_image=args.max_regions_per_image,
        rf_trees=args.rf_trees,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_max_features=args.rf_max_features,
        random_state=args.random_state,
        save_vis_count=args.save_vis_count,
    )


def make_run_dirs(outputs_root: Path) -> dict[str, Path]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_root / f"cityscapes_slic_rf_{ts}"
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
        "model_path": ckpt_dir / "random_forest.joblib",
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


def load_cityscapes_pair(img_path: Path, mask_path: Path, cfg: SLICRFConfig) -> tuple[np.ndarray, np.ndarray]:
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


def compute_superpixels(image: np.ndarray, cfg: SLICRFConfig) -> np.ndarray:
    return slic(
        image.astype(np.float32) / 255.0,
        n_segments=int(cfg.n_segments),
        compactness=float(cfg.compactness),
        sigma=float(cfg.sigma),
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
        enforce_connectivity=bool(cfg.enforce_connectivity),
    ).astype(np.int32, copy=False)


def _segment_means_and_stds(values: np.ndarray, seg_ids: np.ndarray, counts: np.ndarray, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((n_segments, values.shape[1]), dtype=np.float32)
    stds = np.zeros((n_segments, values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        sums = np.bincount(seg_ids, weights=values[:, dim], minlength=n_segments).astype(np.float32)
        sq_sums = np.bincount(seg_ids, weights=values[:, dim] ** 2, minlength=n_segments).astype(np.float32)
        means[:, dim] = sums / counts
        var = np.maximum(sq_sums / counts - means[:, dim] ** 2, 0.0)
        stds[:, dim] = np.sqrt(var, dtype=np.float32)
    return means, stds


def extract_superpixel_features(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    lab = rgb2lab(image_f).astype(np.float32)
    edge = sobel(rgb2gray(image_f)).astype(np.float32)[..., None]

    h, w = segments.shape
    seg_ids = segments.reshape(-1).astype(np.int64)
    n_segments = int(seg_ids.max()) + 1
    counts = np.bincount(seg_ids, minlength=n_segments).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    rgb_flat = image_f.reshape(-1, 3)
    lab_flat = lab.reshape(-1, 3)
    edge_flat = edge.reshape(-1, 1)

    rgb_mean, rgb_std = _segment_means_and_stds(rgb_flat, seg_ids, counts, n_segments)
    lab_mean, lab_std = _segment_means_and_stds(lab_flat, seg_ids, counts, n_segments)
    edge_mean, edge_std = _segment_means_and_stds(edge_flat, seg_ids, counts, n_segments)

    yy, xx = np.mgrid[0:h, 0:w]
    yy_flat = yy.reshape(-1).astype(np.float32)
    xx_flat = xx.reshape(-1).astype(np.float32)
    cy = np.bincount(seg_ids, weights=yy_flat, minlength=n_segments).astype(np.float32) / counts
    cx = np.bincount(seg_ids, weights=xx_flat, minlength=n_segments).astype(np.float32) / counts
    centroid = np.stack([cy / max(h - 1, 1), cx / max(w - 1, 1)], axis=1)
    area_ratio = (counts / float(h * w))[:, None]

    features = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            lab_mean,
            lab_std,
            edge_mean,
            edge_std,
            centroid.astype(np.float32),
            area_ratio.astype(np.float32),
        ],
        axis=1,
    )
    return features.astype(np.float32)


def assign_superpixel_labels(segments: np.ndarray, mask: np.ndarray, num_classes: int, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    seg_ids = segments.reshape(-1).astype(np.int64)
    mask_flat = mask.reshape(-1).astype(np.int64)
    n_segments = int(seg_ids.max()) + 1
    label_counts = np.zeros((n_segments, num_classes), dtype=np.int32)
    for class_id in range(num_classes):
        hits = mask_flat == class_id
        if np.any(hits):
            label_counts[:, class_id] = np.bincount(seg_ids[hits], minlength=n_segments).astype(np.int32)
    labels = label_counts.argmax(axis=1).astype(np.int64)
    valid_segments = label_counts.sum(axis=1) > 0
    ignored_hits = np.bincount(seg_ids[mask_flat == ignore_index], minlength=n_segments) if np.any(mask_flat == ignore_index) else 0
    if isinstance(ignored_hits, np.ndarray):
        valid_segments &= (ignored_hits < np.bincount(seg_ids, minlength=n_segments))
    return labels, valid_segments


def subsample_regions(
    features: np.ndarray,
    labels: np.ndarray,
    valid_segments: np.ndarray,
    max_regions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    keep_idx = np.flatnonzero(valid_segments)
    if keep_idx.size == 0:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    if max_regions > 0 and keep_idx.size > max_regions:
        keep_idx = rng.choice(keep_idx, size=max_regions, replace=False)
        keep_idx.sort()
    return features[keep_idx], labels[keep_idx]


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


def train_random_forest(cfg: SLICRFConfig, train_pairs: list[tuple[Path, Path, str]], run_paths: dict[str, Path]) -> RandomForestClassifier:
    rng = np.random.default_rng(cfg.random_state)
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_regions = 0

    start = time.time()
    for img_path, mask_path, rel_name in tqdm(train_pairs, desc="train-features", leave=False):
        image, mask = load_cityscapes_pair(img_path, mask_path, cfg)
        segments = compute_superpixels(image, cfg)
        features = extract_superpixel_features(image, segments)
        labels, valid_segments = assign_superpixel_labels(segments, mask, cfg.num_classes, cfg.ignore_index)
        sampled_features, sampled_labels = subsample_regions(
            features,
            labels,
            valid_segments,
            cfg.max_regions_per_image,
            rng,
        )
        if sampled_labels.size == 0:
            continue
        total_regions += int(sampled_labels.size)
        all_features.append(sampled_features)
        all_labels.append(sampled_labels)

    if not all_features:
        raise RuntimeError("No valid training superpixels were extracted.")

    x_train = np.concatenate(all_features, axis=0)
    y_train = np.concatenate(all_labels, axis=0)

    print(f"[INFO] train_pairs={len(train_pairs)} train_regions={total_regions} feature_dim={x_train.shape[1]}")
    print(f"[INFO] fitting RandomForest: trees={cfg.rf_trees} max_depth={cfg.rf_max_depth}")

    clf = RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        max_depth=(None if cfg.rf_max_depth <= 0 else cfg.rf_max_depth),
        min_samples_leaf=cfg.rf_min_samples_leaf,
        max_features=cfg.rf_max_features,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_state,
        verbose=0,
    )
    clf.fit(x_train, y_train)
    fit_sec = time.time() - start
    print(f"[INFO] train_done time_sec={fit_sec:.2f}")

    joblib.dump(clf, run_paths["model_path"])
    return clf


def evaluate_random_forest(
    clf: RandomForestClassifier,
    cfg: SLICRFConfig,
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
        segments = compute_superpixels(image, cfg)
        features = extract_superpixel_features(image, segments)
        region_pred = clf.predict(features).astype(np.int64, copy=False)
        pred = region_pred[segments]

        pred_t = torch.from_numpy(pred.astype(np.int64, copy=False)).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.int64, copy=False)).unsqueeze(0)
        update_confusion_matrix(conf, pred_t, mask_t, cfg.num_classes, cfg.ignore_index)
        bf, _bp, _br = boundary_fscore(pred_t, mask_t, ignore_index=cfg.ignore_index, dilation=2)
        bf_scores.append(float(bf))
        trimap_scores.append(float(trimap_iou(pred_t, mask_t, ignore_index=cfg.ignore_index, trimap_width=3)))

        if vis_saved < cfg.save_vis_count:
            save_triplet(run_paths["vis_dir"], rel_name, image, mask, pred.astype(np.uint8))
            vis_saved += 1

    eval_sec = time.time() - start
    miou, iou, precision, recall = compute_metrics_from_confusion(conf)
    metrics = {
        "miou": miou,
        "boundary_fscore": float(np.nanmean(np.asarray(bf_scores, dtype=np.float64))) if bf_scores else float("nan"),
        "trimap_iou": float(np.nanmean(np.asarray(trimap_scores, dtype=np.float64))) if trimap_scores else float("nan"),
        "iou_per_class": iou.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "eval_time_sec": eval_sec,
    }
    return metrics


def save_outputs(cfg: SLICRFConfig, run_paths: dict[str, Path], metrics: dict[str, object]) -> None:
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
    run_paths = make_run_dirs(cfg.outputs_root)
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
        "[INFO] slic="
        f"resize_width={cfg.resize_width} n_segments={cfg.n_segments} "
        f"compactness={cfg.compactness} sigma={cfg.sigma}"
    )

    clf = train_random_forest(cfg, train_pairs, run_paths)
    metrics = evaluate_random_forest(clf, cfg, val_pairs, run_paths)
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
