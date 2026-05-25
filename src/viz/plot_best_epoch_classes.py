from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import best_epoch_from_metrics, configure_matplotlib, default_output_path, load_metrics, load_per_class, resolve_run_dir
else:
    from .common import best_epoch_from_metrics, configure_matplotlib, default_output_path, load_metrics, load_per_class, resolve_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot class-wise IoU / precision / recall at the best epoch.")
    parser.add_argument("run_dir", type=str, help="Run directory under outputs/")
    parser.add_argument("--output", type=str, default="", help="Output png path. Defaults to logs/figures/class_profile_best_epoch.png")
    parser.add_argument("--sort-by", type=str, default="iou", choices=["iou", "precision", "recall"], help="Metric used to sort classes")
    parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    configure_matplotlib()
    metrics_df = load_metrics(run_dir)
    per_class_df = load_per_class(run_dir)
    best_epoch = best_epoch_from_metrics(metrics_df)
    best_df = per_class_df.loc[per_class_df["epoch"] == best_epoch].copy()
    best_df = best_df.sort_values(args.sort_by, ascending=not args.descending).reset_index(drop=True)

    y = np.arange(len(best_df))
    bar_h = 0.24

    fig_h = max(8, 0.45 * len(best_df) + 2)
    fig, ax = plt.subplots(figsize=(13, fig_h), constrained_layout=True)
    ax.barh(y - bar_h, best_df["iou"], height=bar_h, color="#4c78a8", label="IoU")
    ax.barh(y, best_df["precision"], height=bar_h, color="#f58518", label="Precision")
    ax.barh(y + bar_h, best_df["recall"], height=bar_h, color="#54a24b", label="Recall")

    ax.set_yticks(y)
    ax.set_yticklabels(best_df["class_name"])
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title(f"Class-wise Metrics at Best mIoU Epoch ({best_epoch})")
    ax.legend(loc="lower right")

    output_path = default_output_path(run_dir, "class_profile_best_epoch.png") if not args.output else run_dir.joinpath(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
