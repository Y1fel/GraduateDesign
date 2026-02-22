import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class PlotConfig:
    metrics_csv: Optional[Path] = None
    out_dir: Optional[Path] = None
    save_fig: bool = True
    show_fig: bool = True


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"epoch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)

    for col in ("train_loss", "val_miou"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_series(df: pd.DataFrame, x: str, y: str, title: str, out_path: Optional[Path], show: bool) -> None:
    plt.figure()
    plt.plot(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics from metrics.csv")
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to logs/metrics.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to save figures")
    parser.add_argument("--show", dest="show_fig", action="store_true", help="Show figures interactively")
    parser.add_argument("--no-show", dest="show_fig", action="store_false", help="Do not show figures")
    parser.set_defaults(show_fig=True)
    parser.add_argument("--save", dest="save_fig", action="store_true", help="Save figures to --out-dir")
    parser.add_argument("--no-save", dest="save_fig", action="store_false", help="Do not save figures")
    parser.set_defaults(save_fig=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PlotConfig(
        metrics_csv=args.metrics_csv,
        out_dir=args.out_dir,
        save_fig=bool(args.save_fig),
        show_fig=bool(args.show_fig),
    )

    if cfg.save_fig and cfg.out_dir is None:
        cfg.out_dir = cfg.metrics_csv.parent / "plots"

    df = load_metrics(cfg.metrics_csv)

    if "train_loss" in df.columns:
        plot_series(
            df=df,
            x="epoch",
            y="train_loss",
            title="Train Loss vs Epoch",
            out_path=(cfg.out_dir / "train_loss.png") if cfg.save_fig and cfg.out_dir is not None else None,
            show=cfg.show_fig,
        )

    if "val_miou" in df.columns:
        plot_series(
            df=df,
            x="epoch",
            y="val_miou",
            title="Val mIoU vs Epoch",
            out_path=(cfg.out_dir / "val_miou.png") if cfg.save_fig and cfg.out_dir is not None else None,
            show=cfg.show_fig,
        )


if __name__ == "__main__":
    main()
