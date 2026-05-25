from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (
        best_epoch_from_metrics,
        configure_matplotlib,
        default_output_path,
        load_metrics,
        load_per_class,
        macro_precision_recall_curve,
        resolve_run_dir,
    )
else:
    from .common import (
        best_epoch_from_metrics,
        configure_matplotlib,
        default_output_path,
        load_metrics,
        load_per_class,
        macro_precision_recall_curve,
        resolve_run_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a paper-style dashboard for a single training run.")
    parser.add_argument("run_dir", type=str, help="Run directory under outputs/")
    parser.add_argument("--output", type=str, default="", help="Output png path. Defaults to logs/figures/training_dashboard.png")
    parser.add_argument("--title", type=str, default="", help="Custom figure title")
    parser.add_argument("--bottom-k", type=int, default=8, help="Number of weakest classes to show")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    configure_matplotlib()
    metrics_df = load_metrics(run_dir)
    per_class_df = load_per_class(run_dir)
    macro_df = macro_precision_recall_curve(per_class_df)
    best_epoch = best_epoch_from_metrics(metrics_df)
    best_metrics = metrics_df.loc[metrics_df["epoch"] == best_epoch].iloc[0]
    best_per_class = per_class_df.loc[per_class_df["epoch"] == best_epoch].copy()
    best_per_class = best_per_class.sort_values("iou", ascending=True).head(max(1, int(args.bottom_k)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    ax_loss, ax_score, ax_pr, ax_bottom = axes.flatten()

    ax_loss.plot(metrics_df["epoch"], metrics_df["train_loss"], color="#1f77b4", linewidth=2.3, label="Train loss")
    ax_loss.plot(metrics_df["epoch"], metrics_df["val_loss"], color="#d62728", linewidth=2.3, label="Val loss")
    ax_loss.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2)
    ax_loss.set_title("Optimization Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")

    ax_score.plot(metrics_df["epoch"], metrics_df["val_miou"], color="#ff7f0e", linewidth=2.4, label="mIoU")
    ax_score.plot(metrics_df["epoch"], metrics_df["val_bf1"], color="#2ca02c", linewidth=2.0, label="BF1")
    ax_score.scatter(
        [best_epoch],
        [best_metrics["val_miou"]],
        color="#111111",
        s=42,
        zorder=5,
        label=f"Best mIoU {best_metrics['val_miou']:.4f}",
    )
    ax_score.set_title("Validation Metrics")
    ax_score.set_xlabel("Epoch")
    ax_score.set_ylabel("Score")
    ax_score.set_ylim(0.0, 1.0)
    ax_score.legend(loc="lower right")

    ax_pr.plot(macro_df["epoch"], macro_df["macro_precision"], color="#9467bd", linewidth=2.3, label="Macro Precision")
    ax_pr.plot(macro_df["epoch"], macro_df["macro_recall"], color="#8c564b", linewidth=2.3, label="Macro Recall")
    ax_pr.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2)
    ax_pr.set_title("Macro Precision / Recall")
    ax_pr.set_xlabel("Epoch")
    ax_pr.set_ylabel("Score")
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.legend(loc="lower right")

    ax_bottom.barh(best_per_class["class_name"], best_per_class["iou"], color="#4c78a8")
    for y, value in enumerate(best_per_class["iou"]):
        ax_bottom.text(float(value) + 0.01, y, f"{value:.3f}", va="center", fontsize=9)
    ax_bottom.set_title(f"Weakest Classes at Best Epoch ({best_epoch})")
    ax_bottom.set_xlabel("IoU")
    ax_bottom.set_xlim(0.0, 1.0)

    figure_title = args.title or f"{run_dir.name} | best mIoU={best_metrics['val_miou']:.4f} @ epoch {best_epoch}"
    fig.suptitle(figure_title, fontsize=16, y=1.02)

    output_path = default_output_path(run_dir, "training_dashboard.png") if not args.output else run_dir.joinpath(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
