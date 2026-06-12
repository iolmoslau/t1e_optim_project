#!/usr/bin/env python3
"""Constrained normalized-pressure landscapes.

This script mirrors optimal_norm_p.py:

1. Fix epsilon, optimize kappa and delta, then plot kappa-delta.
2. Fix kappa, optimize epsilon and delta, then plot epsilon-delta.
3. Fix delta, optimize epsilon and kappa, then plot epsilon-kappa.

The change is the optimization problem.  The pressure objective is the local
JAX version of normalized psi pressure, and the optimizer also enforces a
volume upper bound.  The volume limit is the elliptical toroid with

    epsilon = 0.45, kappa = 1.9, delta = 0

The inequality-constrained problem is

    maximize pressure(shape)
    subject to volume(shape) <= target_volume

Shared JAX helpers score pressure and volume. This file keeps the landscape,
plotting, and optimizer plumbing local.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
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
from matplotlib.lines import Line2D
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimal_JAX.utils_JAX import (  # noqa: E402
    beta_t_alternative_jax as utils_beta_t_alternative_jax,
    beta_toroidal_jax as utils_beta_toroidal_jax,
    normalized_psi_pressure_masking_jax,
    volume_jax as utils_volume_jax,
)


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

TARGET_SHAPE = np.array([0.45, 1.9, 0.0], dtype=float)

DEFAULT_A = -0.05
DEFAULT_N = 500
DEFAULT_GRID_SIZE = 40
DEFAULT_VOLUME_POINTS = 512
DEFAULT_MAXITER = 200
DEFAULT_OUTPUT_DIR = Path("optimal_JAX/output/landscape_vol_constrainted_output")
BAD_OBJECTIVE_VALUE = 1e100
DEFAULT_Q = 2.0
NORMALIZED_OBJECTIVE = "normalized_pressure"
BETA_T_OBJECTIVE = "beta_toroidal"
BETA_T_ALT_OBJECTIVE = "beta_t_alternative"
VALID_PARAMETER_BOUNDS = {
    "epsilon": (0.020001, 0.949),
    "kappa": (0.050001, 12.0),
    "delta": (-0.949, 0.949),
}

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


def shape_is_valid(shape):
    """Reject values that do not make a usable toroidal shape."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and VALID_PARAMETER_BOUNDS["epsilon"][0] <= epsilon <= VALID_PARAMETER_BOUNDS["epsilon"][1]
        and VALID_PARAMETER_BOUNDS["kappa"][0] <= kappa <= VALID_PARAMETER_BOUNDS["kappa"][1]
        and VALID_PARAMETER_BOUNDS["delta"][0] <= delta <= VALID_PARAMETER_BOUNDS["delta"][1]
    )


def default_plot_ranges():
    """Return the per-script forced landscape ranges."""
    MIN_KAPPA = 0.5
    MAX_KAPPA = 4.1
    MIN_EPSILON = 0.1
    MAX_EPSILON = 0.90
    MIN_DELTA = -0.70
    MAX_DELTA = 0.80
    return {
        "epsilon": (MIN_EPSILON, MAX_EPSILON),
        "kappa": (MIN_KAPPA, MAX_KAPPA),
        "delta": (MIN_DELTA, MAX_DELTA),
    }


def objective_label(objective_name):
    """Readable objective name for printed output and plots."""
    if objective_name == BETA_T_ALT_OBJECTIVE:
        return "beta_t_alternative"
    if objective_name == BETA_T_OBJECTIVE:
        return "beta_toroidal"
    return "normalized pressure"


def objective_value_jax(shape, objective_name, A=DEFAULT_A, N=DEFAULT_N):
    """Evaluate the selected objective for JAX optimization."""
    if objective_name == BETA_T_ALT_OBJECTIVE:
        return utils_beta_t_alternative_jax(shape, A=A, N=N)
    if objective_name == BETA_T_OBJECTIVE:
        return utils_beta_toroidal_jax(shape, A=A, q=DEFAULT_Q, N=N)
    return normalized_psi_pressure_masking_jax(shape, A=A, N=N)


def objective_from_shape(shape, objective_name, A=DEFAULT_A, N=DEFAULT_N):
    """Evaluate the selected objective from ordinary NumPy values."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = objective_value_jax(
            jnp.asarray(shape, dtype=jnp.float64),
            objective_name,
            A=float(A),
            N=int(N),
        )
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def volume_from_shape(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Host wrapper for the volume quantity."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = utils_volume_jax(jnp.asarray(shape, dtype=jnp.float64), int(point_count))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def shape_from_free_values(free_values, free_names, fixed_name, fixed_value):
    """Build full (epsilon, kappa, delta) values from two free values."""
    shape = np.array(STARTING_POINT, dtype=float)
    shape[PARAMETER_INDEX[fixed_name]] = float(fixed_value)
    for name, value in zip(free_names, free_values):
        shape[PARAMETER_INDEX[name]] = float(value)
    return shape


def shape_from_free_values_jax(free_values, free_names, fixed_name, fixed_value):
    """JAX version of shape_from_free_values."""
    values_by_name = {fixed_name: jnp.asarray(fixed_value, dtype=jnp.float64)}
    for name, value in zip(free_names, free_values):
        values_by_name[name] = value
    return jnp.stack([values_by_name[name] for name in PARAMETER_NAMES])


def free_values_from_shape(shape, free_names):
    """Read the two free values from a full shape."""
    return np.array([shape[PARAMETER_INDEX[name]] for name in free_names], dtype=float)


def objective_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
    objective_name,
    A,
    N,
):
    """Selected objective as a function of only the two free variables."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return objective_value_jax(shape, objective_name, A=float(A), N=int(N))


def volume_margin_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
    target_volume,
    volume_points,
):
    """Positive means the shape volume is below the allowed maximum."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return target_volume - utils_volume_jax(shape, point_count=int(volume_points))


def maximize_two_parameters_with_volume_limit(
    test,
    target_volume,
    objective_name=NORMALIZED_OBJECTIVE,
    A=DEFAULT_A,
    N=DEFAULT_N,
    maxiter=DEFAULT_MAXITER,
    volume_points=DEFAULT_VOLUME_POINTS,
):
    """Maximize pressure while keeping volume at or below the target."""
    fixed_name = test["fixed"]
    free_names = test["free"]
    fixed_value = STARTING_POINT[PARAMETER_INDEX[fixed_name]]
    start_free_values = free_values_from_shape(STARTING_POINT, free_names)
    path = [start_free_values.copy()]

    initial_shape = shape_from_free_values(
        start_free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    initial_objective = objective_from_shape(initial_shape, objective_name, A=A, N=N)
    initial_volume = volume_from_shape(initial_shape, point_count=volume_points)

    value_and_gradient = jax.value_and_grad(
        lambda free_values: -objective_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
            objective_name=objective_name,
            A=float(A),
            N=int(N),
        )
    )
    margin_and_gradient = jax.value_and_grad(
        lambda free_values: volume_margin_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
            target_volume=target_volume,
            volume_points=int(volume_points),
        )
    )

    def loss_and_gradient(free_values):
        trial_shape = shape_from_free_values(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        if not shape_is_valid(trial_shape):
            return BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)

        try:
            value, gradient = value_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)
        return value, gradient

    def volume_constraint(free_values):
        trial_shape = shape_from_free_values(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        if not shape_is_valid(trial_shape):
            return -BAD_OBJECTIVE_VALUE
        try:
            margin, _ = margin_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            return float(margin)
        except Exception:
            return -BAD_OBJECTIVE_VALUE

    def volume_constraint_gradient(free_values):
        try:
            _, gradient = margin_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.zeros(2, dtype=float)
        if not np.all(np.isfinite(gradient)):
            return np.zeros(2, dtype=float)
        return gradient

    def remember_step(free_values):
        path.append(np.asarray(free_values, dtype=float).copy())

    result = minimize(
        loss_and_gradient,
        start_free_values,
        method="SLSQP",
        jac=True,
        bounds=[VALID_PARAMETER_BOUNDS[name] for name in free_names],
        constraints=[
            {
                "type": "ineq",
                "fun": volume_constraint,
                "jac": volume_constraint_gradient,
            }
        ],
        callback=remember_step,
        options={"ftol": 1e-8, "maxiter": int(maxiter)},
    )
    if not np.allclose(path[-1], result.x):
        path.append(np.asarray(result.x, dtype=float).copy())

    final_shape = shape_from_free_values(
        result.x,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    final_objective = objective_from_shape(final_shape, objective_name, A=A, N=N)
    final_volume = volume_from_shape(final_shape, point_count=volume_points)
    final_volume_margin = target_volume - final_volume

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "fixed_name": fixed_name,
        "fixed_value": fixed_value,
        "free_names": free_names,
        "objective_name": objective_name,
        "initial_objective": initial_objective,
        "initial_volume": initial_volume,
        "final_shape": final_shape,
        "final_objective": final_objective,
        "final_volume": final_volume,
        "final_volume_margin": final_volume_margin,
    }


def draw_path(ax, path, label="constrained solve path", color="white"):
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


def should_use_log_axis(name, value_range):
    """Use log spacing when a positive parameter spans a large range."""
    low, high = value_range
    return name in ("epsilon", "kappa") and low > 0 and high / low > 20.0


def values_for_axis(name, value_range, count):
    """Build landscape sample points for one plot axis."""
    if should_use_log_axis(name, value_range):
        return np.geomspace(value_range[0], value_range[1], int(count))
    return np.linspace(*value_range, int(count))


def validate_plot_range(name, value_range):
    """Return a finite increasing plot range."""
    low, high = np.asarray(value_range, dtype=float)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"--{name}-range must be two finite values with LOW < HIGH.")
    return (float(low), float(high))


def landscape_values(
    test,
    fixed_value,
    x_range,
    y_range,
    grid_size,
    objective_name,
    A,
    N,
    volume_points,
):
    """Evaluate the selected objective and volume over the plotted ranges."""
    x_name, y_name = test["free"]
    x_values = values_for_axis(x_name, x_range, grid_size)
    y_values = values_for_axis(y_name, y_range, grid_size)
    objective_values = np.empty((len(y_values), len(x_values)), dtype=float)
    volume_error = np.empty_like(objective_values)

    target_volume = volume_from_shape(TARGET_SHAPE, point_count=volume_points)

    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            shape = shape_from_free_values(
                (x_value, y_value),
                free_names=(x_name, y_name),
                fixed_name=test["fixed"],
                fixed_value=fixed_value,
            )
            objective_values[row, column] = objective_from_shape(
                shape,
                objective_name,
                A=A,
                N=N,
            )
            volume_error[row, column] = (
                volume_from_shape(shape, point_count=volume_points) - target_volume
            )

    return x_values, y_values, objective_values, volume_error


def plot_landscape(
    test_name,
    run,
    grid_size,
    output_dir,
    plot_ranges,
    objective_name,
    A,
    N,
    volume_points,
):
    """Plot one objective landscape, volume-limit contour, and solve path."""
    test = TESTS[test_name]
    x_name, y_name = run["free_names"]
    path = run["path"]
    x_range = plot_ranges[x_name]
    y_range = plot_ranges[y_name]

    x_values, y_values, objective_values, volume_error = landscape_values(
        test,
        fixed_value=run["fixed_value"],
        x_range=x_range,
        y_range=y_range,
        grid_size=grid_size,
        objective_name=objective_name,
        A=A,
        N=N,
        volume_points=volume_points,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / test["output"]
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    finite_objective = objective_values[np.isfinite(objective_values)]
    if finite_objective.size == 0:
        ax.text(0.5, 0.5, "No finite objective values", ha="center", va="center")
    else:
        filled = ax.contourf(x_values, y_values, objective_values, levels=35, cmap="viridis")
        lines = ax.contour(
            x_values,
            y_values,
            objective_values,
            levels=12,
            colors="black",
            alpha=0.25,
        )
        ax.clabel(lines, inline=True, fontsize=8)
        fig.colorbar(filled, ax=ax, label=objective_label(objective_name))

    finite_volume_error = volume_error[np.isfinite(volume_error)]
    if finite_volume_error.size and np.min(finite_volume_error) <= 0.0 <= np.max(finite_volume_error):
        ax.contour(
            x_values,
            y_values,
            volume_error,
            levels=[0.0],
            colors="crimson",
            linewidths=2.5,
        )

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

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="crimson", linewidth=2.5))
    labels.append("volume limit")
    ax.legend(handles, labels, loc="best")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def print_run_summary(test_name, run, output_path, target_volume):
    """Print the optimization result in a compact, readable form."""
    result = run["result"]
    epsilon, kappa, delta = run["final_shape"]
    label = objective_label(run["objective_name"])
    print(f"\n{test_name}")
    print(f"  fixed {run['fixed_name']}: {run['fixed_value']:.8g}")
    print(f"  optimizer success: {bool(result.success)}")
    print(f"  optimizer message: {result.message}")
    print(f"  final epsilon: {epsilon:.8g}")
    print(f"  final kappa:   {kappa:.8g}")
    print(f"  final delta:   {delta:.8g}")
    print(f"  starting {label}: {run['initial_objective']:.8g}")
    print(f"  final {label}: {run['final_objective']:.8g}")
    print(f"  maximum allowed volume: {target_volume:.8g}")
    print(f"  starting volume: {run['initial_volume']:.8g}")
    print(f"  final volume: {run['final_volume']:.8g}")
    print(f"  final volume margin: {run['final_volume_margin']:.8g}")
    print(f"  volume constraint satisfied: {run['final_volume'] <= target_volume + 1e-7}")
    print(f"  plot: {output_path}")


def parse_args():
    plot_ranges = default_plot_ranges()
    parser = argparse.ArgumentParser(
        description="Plot optimal_vol_constrainted.py objective landscapes."
    )
    parser.add_argument(
        "--test",
        choices=("all", *TESTS.keys()),
        default="all",
        help="Which fixed-parameter test to run.",
    )
    objective_group = parser.add_mutually_exclusive_group()
    objective_group.add_argument(
        "--beta_t",
        action="store_true",
        help="Maximize beta_toroidal instead of normalized pressure.",
    )
    objective_group.add_argument(
        "--beta_t_alt",
        action="store_true",
        help="Maximize beta_t_alternative instead of normalized pressure.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Number of landscape samples along each plotted axis.",
    )
    parser.add_argument(
        "--epsilon-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=plot_ranges["epsilon"],
        help="Plot range to use whenever epsilon is an axis.",
    )
    parser.add_argument(
        "--kappa-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=plot_ranges["kappa"],
        help="Plot range to use whenever kappa is an axis.",
    )
    parser.add_argument(
        "--delta-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=plot_ranges["delta"],
        help="Plot range to use whenever delta is an axis.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Grid resolution for the local JAX normalized-psi pressure.",
    )
    parser.add_argument(
        "--volume-points",
        type=int,
        default=DEFAULT_VOLUME_POINTS,
        help="Number of boundary points used for the volume calculation.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help="Maximum SLSQP iterations for each constrained solve.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=DEFAULT_A,
        help="A parameter used by the local normalized-psi pressure.",
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
    if args.N < 3:
        raise ValueError("--N must be at least 3.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.volume_points < 16:
        raise ValueError("--volume-points must be at least 16.")

    plot_ranges = {
        "epsilon": validate_plot_range("epsilon", args.epsilon_range),
        "kappa": validate_plot_range("kappa", args.kappa_range),
        "delta": validate_plot_range("delta", args.delta_range),
    }
    if args.beta_t_alt:
        objective_name = BETA_T_ALT_OBJECTIVE
    elif args.beta_t:
        objective_name = BETA_T_OBJECTIVE
    else:
        objective_name = NORMALIZED_OBJECTIVE

    target_volume = volume_from_shape(TARGET_SHAPE, point_count=args.volume_points)
    test_names = list(TESTS) if args.test == "all" else [args.test]

    print("volume upper bound")
    print(f"  epsilon: {TARGET_SHAPE[0]:.8g}")
    print(f"  kappa:   {TARGET_SHAPE[1]:.8g}")
    print(f"  delta:   {TARGET_SHAPE[2]:.8g}")
    print(f"  volume:  {target_volume:.8g}")
    print(f"objective: {objective_label(objective_name)}")

    for test_name in test_names:
        run = maximize_two_parameters_with_volume_limit(
            TESTS[test_name],
            target_volume=target_volume,
            objective_name=objective_name,
            A=args.A,
            N=args.N,
            maxiter=args.maxiter,
            volume_points=args.volume_points,
        )
        output_path = plot_landscape(
            test_name,
            run,
            grid_size=args.grid_size,
            output_dir=args.output_dir,
            plot_ranges=plot_ranges,
            objective_name=objective_name,
            A=args.A,
            N=args.N,
            volume_points=args.volume_points,
        )
        print_run_summary(test_name, run, output_path, target_volume)


if __name__ == "__main__":
    main()
