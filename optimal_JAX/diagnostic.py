"""Plot averaged volume pressure contours for shape-parameter pairs."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pressure_integral.pressure_utils import get_vol_av_p_from_params  # noqa: E402


DEFAULT_INITIAL_SHAPE = (0.32, 1.30, 0.20)
DEFAULT_PARAMETER_RANGES = {
    "epsilon": (0.10, 0.45),
    "kappa": (1.00, 1.70),
    "delta": (-0.30, 0.30),
}
DEFAULT_A_START = 0.0
DEFAULT_A_STOP = -0.5
DEFAULT_A_COUNT = 6
DEFAULT_CONTOUR_COUNT = 30
DEFAULT_N_QUAD = 256
DEFAULT_GRID_SIZE = 300
DEFAULT_METHOD = "parametric"
DEFAULT_OUTPUT = Path(__file__).with_name("diagnostic.png")

PARAMETER_NAMES = ("epsilon", "kappa", "delta")
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}
PARAMETER_LABELS = {
    "epsilon": r"$\epsilon$",
    "kappa": r"$\kappa$",
    "delta": r"$\delta$",
}
PARAMETER_PAIRS = (
    ("epsilon", "kappa"),
    ("kappa", "delta"),
    ("delta", "epsilon"),
)


def averaged_volume_pressure(
    epsilon,
    kappa,
    delta,
    A,
    method=DEFAULT_METHOD,
    n_quad=DEFAULT_N_QUAD,
    grid_size=DEFAULT_GRID_SIZE,
):
    if method == "parametric":
        kwargs = {"h": 2.0 * np.pi / int(n_quad)}
    else:
        kwargs = {"N": int(grid_size)}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = get_vol_av_p_from_params(
                float(epsilon),
                float(kappa),
                float(delta),
                A=float(A),
                method=method,
                **kwargs,
            )
    except (ArithmeticError, ValueError, RuntimeWarning, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def _shape_for_pair(base_shape, pair, x_value, y_value):
    shape = list(base_shape)
    shape[PARAMETER_INDEX[pair[0]]] = float(x_value)
    shape[PARAMETER_INDEX[pair[1]]] = float(y_value)
    return tuple(shape)


def scan_pressure_contours(
    a_values,
    parameter_ranges,
    base_shape=DEFAULT_INITIAL_SHAPE,
    count=DEFAULT_CONTOUR_COUNT,
    method=DEFAULT_METHOD,
    n_quad=DEFAULT_N_QUAD,
    grid_size=DEFAULT_GRID_SIZE,
):
    surfaces = []
    for A in a_values:
        row = []
        for pair in PARAMETER_PAIRS:
            x_values = np.linspace(*parameter_ranges[pair[0]], int(count))
            y_values = np.linspace(*parameter_ranges[pair[1]], int(count))
            X, Y = np.meshgrid(x_values, y_values)
            Z = np.empty_like(X, dtype=float)

            for index in np.ndindex(X.shape):
                epsilon, kappa, delta = _shape_for_pair(
                    base_shape,
                    pair,
                    X[index],
                    Y[index],
                )
                Z[index] = averaged_volume_pressure(
                    epsilon,
                    kappa,
                    delta,
                    A=A,
                    method=method,
                    n_quad=n_quad,
                    grid_size=grid_size,
                )

            row.append({"pair": pair, "X": X, "Y": Y, "Z": Z})
        surfaces.append({"A": float(A), "columns": row})
    return surfaces


def _third_parameter(pair):
    return next(name for name in PARAMETER_NAMES if name not in pair)


def _contour_levels(surfaces, count=24):
    finite_values = [
        column["Z"][np.isfinite(column["Z"])]
        for row in surfaces
        for column in row["columns"]
        if np.isfinite(column["Z"]).any()
    ]
    if not finite_values:
        return None

    values = np.concatenate(finite_values)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 1e-6, 1e-6)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, int(count))


def plot_pressure_contours(
    surfaces,
    base_shape=DEFAULT_INITIAL_SHAPE,
    output_path=DEFAULT_OUTPUT,
):
    output_path = Path(output_path)
    row_count = len(surfaces)
    column_count = len(PARAMETER_PAIRS)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.8 * column_count, 3.2 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    levels = _contour_levels(surfaces)
    mappable = None

    for row_index, row in enumerate(surfaces):
        for column_index, column in enumerate(row["columns"]):
            ax = axes[row_index, column_index]
            pair = column["pair"]
            X = column["X"]
            Y = column["Y"]
            Z = column["Z"]

            if levels is None:
                ax.text(0.5, 0.5, "No finite pressure values", ha="center", va="center")
            else:
                mappable = ax.contourf(
                    X,
                    Y,
                    Z,
                    levels=levels,
                    cmap="viridis",
                    extend="both",
                )
                ax.contour(X, Y, Z, levels=levels[::4], colors="black", linewidths=0.35)

            base_x = base_shape[PARAMETER_INDEX[pair[0]]]
            base_y = base_shape[PARAMETER_INDEX[pair[1]]]
            ax.plot(base_x, base_y, marker="x", color="white", markersize=7, mew=1.8)
            ax.plot(base_x, base_y, marker="x", color="black", markersize=5, mew=1.0)

            if row_index == 0:
                third = _third_parameter(pair)
                held_value = base_shape[PARAMETER_INDEX[third]]
                ax.set_title(
                    f"{PARAMETER_LABELS[pair[0]]} & {PARAMETER_LABELS[pair[1]]}"
                    f"\nheld {PARAMETER_LABELS[third]}={held_value:.3g}"
                )
            if row_index == row_count - 1:
                ax.set_xlabel(PARAMETER_LABELS[pair[0]])
            ax.set_ylabel(PARAMETER_LABELS[pair[1]])
            if column_index == 0:
                ax.set_ylabel(f"A={row['A']:.3g}\n{PARAMETER_LABELS[pair[1]]}")
            ax.grid(True, alpha=0.25)

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=axes,
            shrink=0.92,
            label="Averaged volume pressure",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_INITIAL_SHAPE[0])
    parser.add_argument("--kappa", type=float, default=DEFAULT_INITIAL_SHAPE[1])
    parser.add_argument("--delta", type=float, default=DEFAULT_INITIAL_SHAPE[2])
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=DEFAULT_PARAMETER_RANGES["epsilon"][0],
    )
    parser.add_argument(
        "--epsilon-max",
        type=float,
        default=DEFAULT_PARAMETER_RANGES["epsilon"][1],
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=DEFAULT_PARAMETER_RANGES["kappa"][0],
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=DEFAULT_PARAMETER_RANGES["kappa"][1],
    )
    parser.add_argument(
        "--delta-min",
        type=float,
        default=DEFAULT_PARAMETER_RANGES["delta"][0],
    )
    parser.add_argument(
        "--delta-max",
        type=float,
        default=DEFAULT_PARAMETER_RANGES["delta"][1],
    )
    parser.add_argument("--A-start", type=float, default=DEFAULT_A_START)
    parser.add_argument("--A-stop", type=float, default=DEFAULT_A_STOP)
    parser.add_argument("--A-count", type=int, default=DEFAULT_A_COUNT)
    parser.add_argument("--count", type=int, default=DEFAULT_CONTOUR_COUNT)
    parser.add_argument(
        "--method",
        choices=("parametric", "contour", "masking"),
        default=DEFAULT_METHOD,
    )
    parser.add_argument("--n-quad", type=int, default=DEFAULT_N_QUAD)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _validate_args(args):
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.A_count <= 0:
        raise ValueError("--A-count must be positive")
    if args.method == "parametric" and args.n_quad <= 0:
        raise ValueError("--n-quad must be positive")
    if args.method != "parametric" and args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")

    for name in PARAMETER_NAMES:
        low = getattr(args, f"{name}_min")
        high = getattr(args, f"{name}_max")
        if low >= high:
            raise ValueError(f"--{name}-min must be less than --{name}-max")


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    base_shape = (args.epsilon, args.kappa, args.delta)
    parameter_ranges = {
        "epsilon": (args.epsilon_min, args.epsilon_max),
        "kappa": (args.kappa_min, args.kappa_max),
        "delta": (args.delta_min, args.delta_max),
    }
    a_values = np.linspace(args.A_start, args.A_stop, int(args.A_count))
    surfaces = scan_pressure_contours(
        a_values,
        parameter_ranges,
        base_shape=base_shape,
        count=args.count,
        method=args.method,
        n_quad=args.n_quad,
        grid_size=args.grid_size,
    )
    output_path = plot_pressure_contours(surfaces, base_shape, args.output)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
