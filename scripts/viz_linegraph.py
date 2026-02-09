from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PlotConfig:
    metrics_csv: Path = Path("D:\MachineLearning\GraduateDesign\outputs\camvid_deeplabv3p_20260207_161811\logs\metrics.csv")
    out_dir: Path = Path("D:\MachineLearning\GraduateDesign\outputs\camvid_deeplabv3p_20260207_161811\logs\plots")
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

    # Ensure epoch numeric and sorted
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)

    # Convert numeric columns if present
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


def main() -> None:
    cfg = PlotConfig()

    df = load_metrics(cfg.metrics_csv)

    if "train_loss" in df.columns:
        plot_series(
            df=df,
            x="epoch",
            y="train_loss",
            title="Train Loss vs Epoch",
            out_path=(cfg.out_dir / "train_loss.png") if cfg.save_fig else None,
            show=cfg.show_fig,
        )

    if "val_miou" in df.columns:
        plot_series(
            df=df,
            x="epoch",
            y="val_miou",
            title="Val mIoU vs Epoch",
            out_path=(cfg.out_dir / "val_miou.png") if cfg.save_fig else None,
            show=cfg.show_fig,
        )

if __name__ == "__main__":
    main()
