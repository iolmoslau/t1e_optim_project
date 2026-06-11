#!/usr/bin/env python3
"""Plot Solov'ev flux contours for one requested shape."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRESSURE_DIR = REPO_ROOT / "pressure_integral"
ITER_DIR = REPO_ROOT / "ITER_Equilibria"
for path in (REPO_ROOT, PRESSURE_DIR, ITER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from pressure_integral.pressure_utils import DEFAULT_A, plot_plasma_profile


DEFAULT_GRID_SIZE = 600
DEFAULT_CONTOUR_COUNT = 20
DEFAULT_DPI = 200


def int_at_least(name: str, minimum: int):
    """Return an argparse type that accepts integers greater than a minimum."""

    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} must be at least {minimum}")
        return parsed

    return parse


def validate_shape(epsilon: float, kappa: float, delta: float) -> tuple[float, float, float]:
    """Validate shape parameters before solving the flux coefficients."""
    shape = np.asarray((epsilon, kappa, delta), dtype=float)
    if not np.all(np.isfinite(shape)):
        raise ValueError("epsilon, kappa, and delta must be finite numbers")

    epsilon, kappa, delta = (float(value) for value in shape)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if not -1.0 < delta < 1.0:
        raise ValueError("delta must be in (-1, 1)")
    return epsilon, kappa, delta


def configure_pyplot(show: bool):
    """Use Agg for noninteractive runs, matching the repository scripts."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_contour_from_shape(
    epsilon: float,
    kappa: float,
    delta: float,
    output_path: Path,
    A: float = DEFAULT_A,
    grid_size: int = DEFAULT_GRID_SIZE,
    contour_count: int = DEFAULT_CONTOUR_COUNT,
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> Path:
    """Save a PNG normalized pressure contour plot for the requested shape."""
    epsilon, kappa, delta = validate_shape(epsilon, kappa, delta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt = configure_pyplot(show)
    ax = plot_plasma_profile(
        epsilon,
        kappa,
        delta,
        A=float(A),
        N=int(grid_size),
        n_levels=int(contour_count)
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    fig = ax.get_figure()

    fig.savefig(output_path, dpi=int(dpi), format="png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save a PNG Solov'ev flux contour plot for a shape."
    )
    parser.add_argument("epsilon", type=float, help="inverse aspect ratio, in (0, 1)")
    parser.add_argument("kappa", type=float, help="elongation, positive")
    parser.add_argument("delta", type=float, help="triangularity, in (-1, 1)")
    parser.add_argument("output_path", type=Path, help="PNG path to write")
    parser.add_argument("--A", type=float, default=DEFAULT_A, help="Solov'ev A value")
    parser.add_argument(
        "--grid-size",
        type=int_at_least("grid size", 2),
        default=DEFAULT_GRID_SIZE,
        help="number of grid points along each axis",
    )
    parser.add_argument(
        "--contours",
        type=int_at_least("contour count", 1),
        default=DEFAULT_CONTOUR_COUNT,
        help="number of contour levels",
    )
    parser.add_argument(
        "--dpi",
        type=int_at_least("dpi", 1),
        default=DEFAULT_DPI,
        help="saved PNG resolution",
    )
    parser.add_argument("--show", action="store_true", help="show the figure interactively")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        output_path = plot_contour_from_shape(
            args.epsilon,
            args.kappa,
            args.delta,
            args.output_path,
            A=args.A,
            grid_size=args.grid_size,
            contour_count=args.contours,
            dpi=args.dpi,
            show=args.show,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
