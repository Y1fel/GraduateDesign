from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


FIGURE_DIRNAME = "figures"


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    label: str
    best_epoch: int
    best_miou: float
    macro_precision: float
    macro_recall: float


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.1,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def resolve_run_dir(path_str: str) -> Path:
    run_dir = Path(path_str).expanduser().resolve()
    metrics_path = run_dir / "logs" / "metrics.csv"
    per_class_path = run_dir / "logs" / "per_class_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found under {run_dir}")
    if not per_class_path.exists():
        raise FileNotFoundError(f"per_class_metrics.csv not found under {run_dir}")
    return run_dir


def load_metrics(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "logs" / "metrics.csv")
    numeric_cols = ["epoch", "train_loss", "val_loss", "val_miou", "val_bf1", "lr", "time_sec"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_per_class(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "logs" / "per_class_metrics.csv")
    numeric_cols = ["epoch", "class_id", "iou", "precision", "recall"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def best_epoch_from_metrics(metrics_df: pd.DataFrame) -> int:
    if metrics_df.empty:
        raise ValueError("metrics.csv is empty")
    best_row = metrics_df.sort_values("val_miou", ascending=False).iloc[0]
    return int(best_row["epoch"])


def macro_precision_recall_curve(per_class_df: pd.DataFrame) -> pd.DataFrame:
    curve = (
        per_class_df.groupby("epoch", as_index=False)[["precision", "recall"]]
        .mean(numeric_only=True)
        .rename(columns={"precision": "macro_precision", "recall": "macro_recall"})
    )
    return curve.sort_values("epoch").reset_index(drop=True)


def summarize_run(run_dir: Path, label: str | None = None) -> RunSummary:
    metrics_df = load_metrics(run_dir)
    per_class_df = load_per_class(run_dir)
    best_epoch = best_epoch_from_metrics(metrics_df)
    best_row = metrics_df.loc[metrics_df["epoch"] == best_epoch].iloc[0]
    best_per_class = per_class_df.loc[per_class_df["epoch"] == best_epoch].copy()
    macro_precision = float(best_per_class["precision"].mean())
    macro_recall = float(best_per_class["recall"].mean())
    return RunSummary(
        run_dir=run_dir,
        label=label or run_dir.name,
        best_epoch=best_epoch,
        best_miou=float(best_row["val_miou"]),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
    )


def default_output_path(run_dir: Path, filename: str) -> Path:
    out_dir = run_dir / "logs" / FIGURE_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def latest_epoch_vis_dir(run_dir: Path) -> Path:
    vis_root = run_dir / "visualizations"
    epoch_dirs = [p for p in vis_root.glob("epoch_*") if p.is_dir()]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* directory found under {vis_root}")
    return sorted(epoch_dirs, key=lambda p: p.name)[-1]


def find_triplet_path(run_dir: Path, sample_key: str, epoch: int | None = None) -> Path:
    vis_dir = run_dir / "visualizations" / f"epoch_{epoch:03d}" if epoch is not None else latest_epoch_vis_dir(run_dir)
    exact = vis_dir / f"{sample_key}_triplet.png"
    if exact.exists():
        return exact

    matches = sorted(vis_dir.glob(f"*{sample_key}*_triplet.png"))
    if not matches:
        raise FileNotFoundError(f"No triplet image matching '{sample_key}' under {vis_dir}")
    return matches[0]


def split_triplet_image(triplet_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(triplet_path).convert("RGB"))
    h, w, _ = image.shape
    if w % 3 != 0:
        raise ValueError(f"Triplet image width must be divisible by 3: {triplet_path}")
    panel_w = w // 3
    return image[:, :panel_w], image[:, panel_w : 2 * panel_w], image[:, 2 * panel_w :]


def build_error_map(gt_rgb: np.ndarray, pred_rgb: np.ndarray) -> np.ndarray:
    mismatch = np.any(gt_rgb != pred_rgb, axis=2)
    overlay = pred_rgb.copy()
    overlay[mismatch] = np.array([220, 20, 60], dtype=np.uint8)
    return overlay


def parse_run_specs(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            label, raw_path = item.split("=", 1)
            parsed.append((label.strip(), resolve_run_dir(raw_path.strip())))
        else:
            run_dir = resolve_run_dir(item)
            parsed.append((run_dir.name, run_dir))
    return parsed
