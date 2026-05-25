from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from common import build_error_map, configure_matplotlib, find_triplet_path, parse_run_specs, split_triplet_image
else:
    from .common import build_error_map, configure_matplotlib, find_triplet_path, parse_run_specs, split_triplet_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a paper-style qualitative panel from saved triplet images.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specs. Use either '/path/to/run' or 'Label=/path/to/run'.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="Sample stem or partial key, e.g. frankfurt_000000_000294_leftImg8bit",
    )
    parser.add_argument("--output", type=str, required=True, help="Output png path")
    parser.add_argument("--epoch", type=int, default=None, help="Optional fixed epoch id for all runs")
    parser.add_argument("--hide-error", action="store_true", help="Do not render error-map columns")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_specs = parse_run_specs(args.runs)
    show_error = not args.hide_error

    configure_matplotlib()

    columns: list[str] = ["Input", "GT"]
    for label, _ in run_specs:
        columns.append(label)
        if show_error:
            columns.append(f"{label} Error")

    n_rows = len(args.samples)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.8 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample_key in enumerate(args.samples):
        first_triplet = find_triplet_path(run_specs[0][1], sample_key, epoch=args.epoch)
        rgb_input, rgb_gt, _ = split_triplet_image(first_triplet)

        col_idx = 0
        axes[row_idx, col_idx].imshow(rgb_input)
        axes[row_idx, col_idx].axis("off")
        col_idx += 1

        axes[row_idx, col_idx].imshow(rgb_gt)
        axes[row_idx, col_idx].axis("off")
        col_idx += 1

        for _, run_dir in run_specs:
            triplet_path = find_triplet_path(run_dir, sample_key, epoch=args.epoch)
            _, _, rgb_pred = split_triplet_image(triplet_path)

            axes[row_idx, col_idx].imshow(rgb_pred)
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

            if show_error:
                axes[row_idx, col_idx].imshow(build_error_map(rgb_gt, rgb_pred))
                axes[row_idx, col_idx].axis("off")
                col_idx += 1

    for col_idx, title in enumerate(columns):
        axes[0, col_idx].set_title(title, fontsize=13, pad=10)

    for row_idx, sample_key in enumerate(args.samples):
        axes[row_idx, 0].set_ylabel(Path(sample_key).stem[:28], fontsize=11)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
