#!/usr/bin/env python3
"""Maximize physical normalized pressure with one fixed shape parameter at a time.

This script runs three small tests:

1. Fix epsilon, optimize kappa and delta, then plot the kappa-delta landscape.
2. Fix kappa, optimize epsilon and delta, then plot the epsilon-delta landscape.
3. Fix delta, optimize epsilon and kappa, then plot the epsilon-kappa landscape.

The plotted landscapes use the configured parameter ranges.  If BFGS moves
outside those ranges, widen STARTING_RANGES to show more of the path.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "pressure_integral").exists() else SCRIPT_DIR.parent
for source_dir in (ROOT, ROOT / "pressure_integral", ROOT / "ITER_Equilibria"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pressure_integral.pressure_utils import _normalized_psi_pressure  # noqa: E402


PARAMETER_NAMES = ("epsilon", "kappa", "delta")
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}
PARAMETER_LABEL = {
    "epsilon": "epsilon",
    "kappa": "kappa",
    "delta": "delta",
}

STARTING_RANGES = {
    "epsilon": (0.10, 0.45),
    "kappa": (1.00, 1.70),
    "delta": (-0.30, 0.30),
}
STARTING_POINT = np.array(
    [
        np.mean(STARTING_RANGES["epsilon"]),
        np.mean(STARTING_RANGES["kappa"]),
        np.mean(STARTING_RANGES["delta"]),
    ],
    dtype=float,
)

DEFAULT_A = -0.5
DEFAULT_METHOD = "contour"
DEFAULT_N = 60
DEFAULT_GRID_SIZE = 20
DEFAULT_MAXITER = 80
DEFAULT_OUTPUT_DIR = Path("optimal_norm_p_output")
FINITE_DIFFERENCE_STEPS = np.array([1e-4, 1e-4, 1e-4], dtype=float)
BAD_OBJECTIVE_VALUE = 1e100

TESTS = {
    "fix-epsilon": {
        "fixed": "epsilon",
        "free": ("kappa", "delta"),
        "output": "fix_epsilon_kappa_delta.png",
    },
    "fix-kappa": {
        "fixed": "kappa",
        "free": ("epsilon", "delta"),
        "output": "fix_kappa_epsilon_delta.png",
    },
    "fix-delta": {
        "fixed": "delta",
        "free": ("epsilon", "kappa"),
        "output": "fix_delta_epsilon_kappa.png",
    },
}


def pressure_from_shape(shape, A=DEFAULT_A, method=DEFAULT_METHOD, N=DEFAULT_N):
    """Call the shared normalized-psi objective as positive physical pressure."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = _normalized_psi_pressure(
                float(epsilon),
                float(kappa),
                float(delta),
                A=float(A),
                method=method,
                N=int(N),
            )
    except (ArithmeticError, ValueError, RuntimeWarning, np.linalg.LinAlgError):
        return np.nan
    return -float(value)


def pressure_gradient_from_shape(shape, A=DEFAULT_A, method=DEFAULT_METHOD, N=DEFAULT_N):
    """Finite-difference derivative used by the JAX callback wrapper."""
    shape = np.asarray(shape, dtype=float)
    gradient = np.zeros(3, dtype=float)
    center_value = pressure_from_shape(shape, A=A, method=method, N=N)

    for index, step in enumerate(FINITE_DIFFERENCE_STEPS):
        plus_shape = shape.copy()
        minus_shape = shape.copy()
        plus_shape[index] += step
        minus_shape[index] -= step

        plus_value = pressure_from_shape(plus_shape, A=A, method=method, N=N)
        minus_value = pressure_from_shape(minus_shape, A=A, method=method, N=N)

        if np.isfinite(plus_value) and np.isfinite(minus_value):
            gradient[index] = (plus_value - minus_value) / (2.0 * step)
        elif np.isfinite(plus_value) and np.isfinite(center_value):
            gradient[index] = (plus_value - center_value) / step
        elif np.isfinite(minus_value) and np.isfinite(center_value):
            gradient[index] = (center_value - minus_value) / step
        else:
            gradient[index] = np.nan

    return gradient


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def pressure_from_shape_jax(shape, A, method, N):
    """JAX-facing wrapper around the shared NumPy pressure function."""
    return jax.pure_callback(
        lambda host_shape: np.asarray(
            pressure_from_shape(host_shape, A=A, method=method, N=N),
            dtype=np.float64,
        ),
        jax.ShapeDtypeStruct((), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )


@pressure_from_shape_jax.defjvp
def pressure_from_shape_jvp(A, method, N, primals, tangents):
    """Tell JAX how to differentiate the host callback."""
    (shape,) = primals
    (shape_dot,) = tangents
    value = pressure_from_shape_jax(shape, A, method, N)
    gradient = jax.pure_callback(
        lambda host_shape: np.asarray(
            pressure_gradient_from_shape(host_shape, A=A, method=method, N=N),
            dtype=np.float64,
        ),
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )
    return value, jnp.dot(gradient, shape_dot)


def shape_from_free_values(free_values, free_names, fixed_name, fixed_value):
    """Build the full (epsilon, kappa, delta) shape from two free values."""
    shape = np.array(STARTING_POINT, dtype=float)
    shape[PARAMETER_INDEX[fixed_name]] = float(fixed_value)
    for name, value in zip(free_names, free_values):
        shape[PARAMETER_INDEX[name]] = float(value)
    return shape


def shape_from_free_values_jax(free_values, free_names, fixed_name, fixed_value):
    """JAX version of shape_from_free_values, used during differentiation."""
    values_by_name = {fixed_name: jnp.asarray(fixed_value, dtype=jnp.float64)}
    for name, value in zip(free_names, free_values):
        values_by_name[name] = value
    return jnp.stack([values_by_name[name] for name in PARAMETER_NAMES])


def free_values_from_shape(shape, free_names):
    """Read two free values out of a full shape."""
    return np.array([shape[PARAMETER_INDEX[name]] for name in free_names], dtype=float)


def maximize_two_parameters(
    test,
    A=DEFAULT_A,
    method=DEFAULT_METHOD,
    N=DEFAULT_N,
    maxiter=DEFAULT_MAXITER,
):
    """Run unconstrained BFGS on the two variables that are not fixed."""
    fixed_name = test["fixed"]
    free_names = test["free"]
    fixed_value = STARTING_POINT[PARAMETER_INDEX[fixed_name]]
    start = free_values_from_shape(STARTING_POINT, free_names)
    path = [start.copy()]
    initial_shape = shape_from_free_values(
        start,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    initial_pressure = pressure_from_shape(initial_shape, A=A, method=method, N=N)

    def pressure_for_free_values(free_values):
        shape = shape_from_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        return pressure_from_shape_jax(shape, float(A), method, int(N))

    value_and_gradient = jax.value_and_grad(pressure_for_free_values)

    def loss_and_gradient(free_values):
        try:
            value, gradient = value_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)
        return -value, -gradient

    def remember_step(free_values):
        path.append(np.asarray(free_values, dtype=float).copy())

    result = minimize(
        loss_and_gradient,
        start,
        method="BFGS",
        jac=True,
        callback=remember_step,
        options={"gtol": 1e-5, "maxiter": int(maxiter)},
    )
    if not np.allclose(path[-1], result.x):
        path.append(np.asarray(result.x, dtype=float).copy())
    final_shape = shape_from_free_values(
        result.x,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    final_pressure = pressure_from_shape(final_shape, A=A, method=method, N=N)
    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "fixed_name": fixed_name,
        "fixed_value": fixed_value,
        "free_names": free_names,
        "initial_pressure": initial_pressure,
        "final_shape": final_shape,
        "final_pressure": final_pressure,
    }


def draw_path(ax, path, label="BFGS path", color="white"):
    """Draw the optimizer path with start and finish markers."""
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        marker="o",
        markeredgecolor="black",
        linewidth=2.0,
        label=label,
    )
    ax.scatter(path[0, 0], path[0, 1], color="lime", edgecolor="black", s=90, label="start")
    ax.scatter(path[-1, 0], path[-1, 1], color="red", edgecolor="black", s=90, label="finish")


def plot_range_for_path(name, path_values, padding_fraction=0.05):
    """Use the configured range, expanded enough to include the full path."""
    starting_low, starting_high = STARTING_RANGES[name]
    low, high = starting_low, starting_high
    finite_path = np.asarray(path_values, dtype=float)
    finite_path = finite_path[np.isfinite(finite_path)]
    if finite_path.size:
        path_low = float(np.min(finite_path))
        path_high = float(np.max(finite_path))
        low = min(low, path_low)
        high = max(high, path_high)
    else:
        path_low = starting_low

    width = high - low
    if width <= 0:
        width = max(1.0, abs(low))
    padding = padding_fraction * width

    high = high + padding
    if name in ("epsilon", "kappa") and path_low > 0:
        if path_low >= starting_low:
            low = starting_low
        else:
            low = max(path_low / (1.0 + padding_fraction), 1e-6)
    else:
        low = low - padding
    return (low, high)


def should_use_log_axis(name, value_range):
    """Use log spacing when a positive parameter spans a large range."""
    low, high = value_range
    return name in ("epsilon", "kappa") and low > 0 and high / low > 20.0


def values_for_axis(name, value_range, count):
    """Build landscape sample points for one plot axis."""
    if should_use_log_axis(name, value_range):
        return np.geomspace(value_range[0], value_range[1], int(count))
    return np.linspace(*value_range, int(count))


def landscape_values(test, fixed_value, x_range, y_range, grid_size, A, method, N):
    """Evaluate the pressure landscape over the plotted ranges."""
    x_name, y_name = test["free"]
    x_values = values_for_axis(x_name, x_range, grid_size)
    y_values = values_for_axis(y_name, y_range, grid_size)
    Z = np.empty((len(y_values), len(x_values)), dtype=float)

    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            shape = shape_from_free_values(
                (x_value, y_value),
                free_names=(x_name, y_name),
                fixed_name=test["fixed"],
                fixed_value=fixed_value,
            )
            Z[row, column] = pressure_from_shape(shape, A=A, method=method, N=N)

    return x_values, y_values, Z


def plot_landscape(test_name, run, grid_size, output_dir, A, method, N):
    """Plot one pressure landscape and the BFGS path through it."""
    test = TESTS[test_name]
    x_name, y_name = run["free_names"]
    path = run["path"]
    x_range = plot_range_for_path(x_name, path[:, 0])
    y_range = plot_range_for_path(y_name, path[:, 1])

    x_values, y_values, Z = landscape_values(
        test,
        fixed_value=run["fixed_value"],
        x_range=x_range,
        y_range=y_range,
        grid_size=grid_size,
        A=A,
        method=method,
        N=N,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / test["output"]
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    finite_Z = Z[np.isfinite(Z)]
    if finite_Z.size == 0:
        ax.text(0.5, 0.5, "No finite pressure values", ha="center", va="center")
    else:
        filled = ax.contourf(x_values, y_values, Z, levels=35, cmap="viridis")
        lines = ax.contour(x_values, y_values, Z, levels=12, colors="black", alpha=0.25)
        ax.clabel(lines, inline=True, fontsize=8)
        fig.colorbar(filled, ax=ax, label="physical normalized pressure")

    draw_path(ax, path)

    ax.set_xlabel(PARAMETER_LABEL[x_name])
    ax.set_ylabel(PARAMETER_LABEL[y_name])
    ax.set_title(
        f"{PARAMETER_LABEL[x_name]} vs {PARAMETER_LABEL[y_name]} "
        f"with fixed {PARAMETER_LABEL[run['fixed_name']]}={run['fixed_value']:.4g}"
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    if should_use_log_axis(x_name, x_range):
        ax.set_xscale("log")
    if should_use_log_axis(y_name, y_range):
        ax.set_yscale("log")
    ax.grid(True, color="white", alpha=0.15)
    ax.legend(loc="best")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def print_run_summary(test_name, run, output_path):
    """Print the optimization result in a compact, readable form."""
    result = run["result"]
    epsilon, kappa, delta = run["final_shape"]
    print(f"\n{test_name}")
    print(f"  fixed {run['fixed_name']}: {run['fixed_value']:.8g}")
    print(f"  optimizer success: {bool(result.success)}")
    print(f"  optimizer message: {result.message}")
    print(f"  final epsilon: {epsilon:.8g}")
    print(f"  final kappa:   {kappa:.8g}")
    print(f"  final delta:   {delta:.8g}")
    print(f"  starting physical normalized pressure: {run['initial_pressure']:.8g}")
    print(f"  final physical normalized pressure: {run['final_pressure']:.8g}")
    print(f"  plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maximize physical normalized pressure with JAX gradients and BFGS."
    )
    parser.add_argument(
        "--test",
        choices=("all", *TESTS.keys()),
        default="all",
        help="Which fixed-parameter test to run.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Number of landscape samples along each plotted axis.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Grid resolution passed to _normalized_psi_pressure.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help="Maximum BFGS iterations for each fixed-parameter test.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=DEFAULT_A,
        help="A parameter passed to _normalized_psi_pressure.",
    )
    parser.add_argument(
        "--method",
        choices=("contour", "masking"),
        default=DEFAULT_METHOD,
        help="Evaluation method passed to _normalized_psi_pressure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the three plots are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_names = list(TESTS) if args.test == "all" else [args.test]

    for test_name in test_names:
        run = maximize_two_parameters(
            TESTS[test_name],
            A=args.A,
            method=args.method,
            N=args.N,
            maxiter=args.maxiter,
        )
        output_path = plot_landscape(
            test_name,
            run,
            grid_size=args.grid_size,
            output_dir=args.output_dir,
            A=args.A,
            method=args.method,
            N=args.N,
        )
        print_run_summary(test_name, run, output_path)


if __name__ == "__main__":
    main()
