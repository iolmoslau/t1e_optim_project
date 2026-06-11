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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from optimal_JAX.utils_JAX import DEFAULT_A, make_psi


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
    """Save a PNG contour plot for the requested shape."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    epsilon, kappa, delta = validate_shape(epsilon, kappa, delta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    psi, _, _ = make_psi(epsilon, kappa, delta, float(A))

    x = np.linspace(1.0 - epsilon - 0.1, 1.0 + epsilon + 0.1, int(grid_size))
    y = np.linspace(-kappa * epsilon - 0.1, kappa * epsilon + 0.1, int(grid_size))
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(psi(X, Y), dtype=float)

    if not np.isfinite(Z).any():
        raise ValueError("psi grid has no finite values; check the shape parameters")

    interior = np.where(Z <= 0.0, Z, np.nan)
    negative_psi = Z[np.isfinite(Z) & (Z < 0.0)]
    if negative_psi.size == 0:
        raise ValueError("psi grid has no finite negative values; check the shape parameters")

    psi_min = float(np.min(negative_psi))
    contour_levels = np.linspace(psi_min, 0.0, int(contour_count))

    plt = configure_pyplot(show)
    fig, ax = plt.subplots(figsize=(4.2, 4.3))
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.11)

    ax.contour(X, Y, interior, levels=contour_levels, cmap="plasma")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R/R_0$")
    ax.set_ylabel(r"$Z/R_0$")
    ax.set_title(
        rf"$\epsilon={epsilon:.3f},\;\kappa={kappa:.3f},\;\delta={delta:.3f}$",
        fontsize=9,
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))

    norm = mcolors.Normalize(vmin=psi_min, vmax=0.0)
    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cax = fig.add_axes((0.79, 0.11, 0.03, 0.82))
    fig.colorbar(sm, cax=cax, label=r"$\psi$")

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
