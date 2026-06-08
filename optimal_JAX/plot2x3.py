#!/usr/bin/env python3
"""Combine the beta_t and normalized-pressure landscape PNGs into a 2x3 plot."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_OUTPUT = OUTPUT_ROOT / "plot2x3.png"

ROWS = (
    ("landscape_beta_t_output", "beta_t landscape"),
    ("optimal_norm_p_output", "normalized pressure"),
)
COLUMNS = (
    ("fix_kappa_epsilon_delta.png", "fixed kappa"),
    ("fix_epsilon_kappa_delta.png", "fixed epsilon"),
    ("fix_delta_epsilon_kappa.png", "fixed delta"),
)


def image_path(row_dir: str, filename: str) -> Path:
    path = OUTPUT_ROOT / row_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing source image: {path}")
    return path


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=200)

    for row_index, (row_dir, row_label) in enumerate(ROWS):
        for col_index, (filename, col_label) in enumerate(COLUMNS):
            ax = axes[row_index, col_index]
            ax.imshow(plt.imread(image_path(row_dir, filename)))
            ax.axis("off")

            if row_index == 0:
                ax.set_title(col_label, fontsize=14, pad=10)
            if col_index == 0:
                ax.text(
                    -0.06,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.02, wspace=0.02, hspace=0.08)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEFAULT_OUTPUT)
    plt.close(fig)
    print(f"Saved {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
