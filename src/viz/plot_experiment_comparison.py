from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import FIGURE_DIRNAME, configure_matplotlib, parse_run_specs, summarize_run
else:
    from .common import FIGURE_DIRNAME, configure_matplotlib, parse_run_specs, summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare best mIoU / precision / recall across multiple runs.")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run specs. Use either '/path/to/run' or 'Label=/path/to/run'.",
    )
    parser.add_argument("--output", type=str, default="", help="Output png path.")
    parser.add_argument("--sort-by", type=str, default="miou", choices=["miou", "precision", "recall"], help="Metric used to sort runs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_specs = parse_run_specs(args.runs)
    summaries = [summarize_run(run_dir, label=label) for label, run_dir in run_specs]

    key_map = {
        "miou": lambda item: item.best_miou,
        "precision": lambda item: item.macro_precision,
        "recall": lambda item: item.macro_recall,
    }
    summaries.sort(key=key_map[args.sort_by], reverse=True)

    labels = [item.label for item in summaries]
    miou = [item.best_miou for item in summaries]
    precision = [item.macro_precision for item in summaries]
    recall = [item.macro_recall for item in summaries]

    configure_matplotlib()
    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(16, max(6, 0.7 * len(summaries) + 2)), constrained_layout=True)

    y = np.arange(len(labels))
    bar_h = 0.22
    ax_bar.barh(y - bar_h, miou, height=bar_h, color="#4c78a8", label="mIoU")
    ax_bar.barh(y, precision, height=bar_h, color="#f58518", label="Precision")
    ax_bar.barh(y + bar_h, recall, height=bar_h, color="#54a24b", label="Recall")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlim(0.0, 1.02)
    ax_bar.set_title("Best Epoch Summary")
    ax_bar.set_xlabel("Score")
    ax_bar.legend(loc="lower right")

    sizes = [220 + 900 * value for value in miou]
    ax_scatter.scatter(precision, recall, s=sizes, c=miou, cmap="viridis", alpha=0.9, edgecolors="black", linewidth=0.8)
    for idx, label in enumerate(labels):
        ax_scatter.annotate(label, (precision[idx], recall[idx]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax_scatter.set_xlim(0.0, 1.02)
    ax_scatter.set_ylim(0.0, 1.02)
    ax_scatter.set_xlabel("Macro Precision")
    ax_scatter.set_ylabel("Macro Recall")
    ax_scatter.set_title("Precision-Recall Trade-off")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = run_specs[0][1].parents[0] / FIGURE_DIRNAME / "experiment_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
