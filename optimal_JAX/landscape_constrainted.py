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

Only this file defines the physics helper functions used here; it does not
import helper functions from the rest of the project.
"""

from __future__ import annotations

import argparse
import os
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

DEFAULT_A = -0.5
DEFAULT_N = 60
DEFAULT_GRID_SIZE = 20
DEFAULT_VOLUME_POINTS = 512
DEFAULT_MAXITER = 80
DEFAULT_OUTPUT_DIR = Path("landscape_constrainted_output")
BAD_OBJECTIVE_VALUE = 1e100
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


def safe_log(x):
    """Logarithm used by the local Solov'ev basis."""
    return jnp.log(jnp.maximum(x, 1e-12))


def basis_values(x, y):
    """Seven homogeneous Solov'ev basis functions."""
    log_x = safe_log(x)
    x2 = x * x
    y2 = y * y

    return jnp.stack(
        [
            jnp.ones_like(x),
            x2,
            y2 - x2 * log_x,
            x2 * x2 - 4.0 * x2 * y2,
            2.0 * y2 * y2
            - 9.0 * x2 * y2
            + 3.0 * x2 * x2 * log_x
            - 12.0 * x2 * y2 * log_x,
            x2 * x2 * x2 - 12.0 * x2 * x2 * y2 + 8.0 * x2 * y2 * y2,
            8.0 * y2 * y2 * y2
            - 140.0 * x2 * y2 * y2
            + 75.0 * x2 * x2 * y2
            - 15.0 * x2 * x2 * x2 * log_x
            + 180.0 * x2 * x2 * y2 * log_x
            - 120.0 * x2 * y2 * y2 * log_x,
        ],
        axis=0,
    )


def particular_value(x, y, A):
    """Particular Solov'ev solution for source A + (1 - A) * x^2."""
    del y
    return A * (0.5 * x * x * safe_log(x)) + (1.0 - A) * (x**4 / 8.0)


def basis_x(x, y):
    return jax.jacfwd(lambda value: basis_values(value, y))(x)


def basis_y(x, y):
    return jax.jacfwd(lambda value: basis_values(x, value))(y)


def basis_xx(x, y):
    return jax.jacfwd(lambda value: basis_x(value, y))(x)


def basis_yy(x, y):
    return jax.jacfwd(lambda value: basis_y(x, value))(y)


def particular_x(x, y, A):
    return jax.grad(lambda value: particular_value(value, y, A))(x)


def particular_y(x, y, A):
    return jax.grad(lambda value: particular_value(x, value, A))(y)


def particular_xx(x, y, A):
    return jax.grad(lambda value: particular_x(value, y, A))(x)


def particular_yy(x, y, A):
    return jax.grad(lambda value: particular_y(x, value, A))(y)


def solve_coefficients(epsilon, kappa, delta, A):
    """Solve the seven boundary equations for the Solov'ev coefficients."""
    alpha = jnp.arcsin(delta)
    curv1 = -((1.0 + alpha) ** 2) / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * jnp.cos(alpha) ** 2)
    curv3 = ((1.0 - alpha) ** 2) / (epsilon * kappa**2)

    x_outer = 1.0 + epsilon
    x_inner = 1.0 - epsilon
    x_high = 1.0 - epsilon * delta
    y_high = kappa * epsilon
    zero = jnp.array(0.0, dtype=jnp.float64)

    rows = [
        basis_values(x_outer, zero),
        basis_values(x_inner, zero),
        basis_values(x_high, y_high),
        basis_x(x_high, y_high),
        curv1 * basis_x(x_outer, zero) + basis_yy(x_outer, zero),
        curv3 * basis_x(x_inner, zero) + basis_yy(x_inner, zero),
        curv2 * basis_y(x_high, y_high) + basis_xx(x_high, y_high),
    ]
    matrix = jnp.stack(rows)

    rhs = -jnp.stack(
        [
            particular_value(x_outer, zero, A),
            particular_value(x_inner, zero, A),
            particular_value(x_high, y_high, A),
            particular_x(x_high, y_high, A),
            curv1 * particular_x(x_outer, zero, A) + particular_yy(x_outer, zero, A),
            curv3 * particular_x(x_inner, zero, A) + particular_yy(x_inner, zero, A),
            curv2 * particular_y(x_high, y_high, A) + particular_xx(x_high, y_high, A),
        ]
    )

    return jnp.linalg.solve(matrix, rhs)


def psi_value(x, y, epsilon, kappa, delta, A):
    """Evaluate the local JAX flux function."""
    coefficients = solve_coefficients(epsilon, kappa, delta, A)
    return jnp.tensordot(coefficients, basis_values(x, y), axes=(0, 0)) + particular_value(
        x, y, A
    )


def normalized_psi_pressure_jax(shape, A=DEFAULT_A, N=DEFAULT_N):
    """Return positive physical normalized psi pressure by a JAX masking rule."""
    epsilon, kappa, delta = shape
    x = jnp.linspace(1.0 - epsilon, 1.0 + epsilon, int(N))
    y = jnp.linspace(-kappa * epsilon, kappa * epsilon, int(N))
    X, Y = jnp.meshgrid(x, y, indexing="xy")

    PSI = psi_value(X, Y, epsilon, kappa, delta, A)
    inside = PSI <= 0.0
    inside_float = inside.astype(jnp.float64)

    psi_min = jnp.min(jnp.where(inside, PSI, jnp.inf))
    abs_psi_min = jnp.abs(psi_min)

    dx = (x[-1] - x[0]) / (int(N) - 1)
    dy = (y[-1] - y[0]) / (int(N) - 1)
    dA = dx * dy

    normalized_psi = PSI / abs_psi_min
    numerator = dA * jnp.sum(X * normalized_psi * inside_float)
    denominator = dA * jnp.sum(X * inside_float)

    return -numerator / denominator


def miller_boundary(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Points on the shape boundary used for the volume constraint."""
    epsilon, kappa, delta = shape
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(point_count) + 1)
    alpha = jnp.arcsin(delta)
    x = 1.0 + epsilon * jnp.cos(theta + alpha * jnp.sin(theta))
    y = kappa * epsilon * jnp.sin(theta)
    return x, y


def volume_jax(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Dimensionless toroidal volume factor int x dA for the shape."""
    x, y = miller_boundary(shape, point_count=point_count)
    x_mid = 0.5 * (x[:-1] + x[1:])
    y_mid = 0.5 * (y[:-1] + y[1:])
    dx = x[1:] - x[:-1]
    return -jnp.sum(x_mid * y_mid * dx)


def shape_is_valid(shape):
    """Reject values that do not make a usable toroidal shape."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and 0.02 < epsilon < 0.95
        and 0.05 < kappa < 12.0
        and abs(delta) < 0.95
    )


def pressure_from_shape(shape, A=DEFAULT_A, N=DEFAULT_N):
    """Host wrapper for the positive normalized pressure."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = normalized_psi_pressure_jax(jnp.asarray(shape, dtype=jnp.float64), float(A), int(N))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def volume_from_shape(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Host wrapper for the volume quantity."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = volume_jax(jnp.asarray(shape, dtype=jnp.float64), int(point_count))
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


def pressure_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
    A,
    N,
):
    """Pressure as a function of only the two free variables."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return normalized_psi_pressure_jax(shape, A=float(A), N=int(N))


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
    return target_volume - volume_jax(shape, point_count=int(volume_points))


def maximize_two_parameters_with_volume_limit(
    test,
    target_volume,
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
    initial_pressure = pressure_from_shape(initial_shape, A=A, N=N)
    initial_volume = volume_from_shape(initial_shape, point_count=volume_points)

    value_and_gradient = jax.value_and_grad(
        lambda free_values: -pressure_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
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
    final_pressure = pressure_from_shape(final_shape, A=A, N=N)
    final_volume = volume_from_shape(final_shape, point_count=volume_points)
    final_volume_margin = target_volume - final_volume

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "fixed_name": fixed_name,
        "fixed_value": fixed_value,
        "free_names": free_names,
        "initial_pressure": initial_pressure,
        "initial_volume": initial_volume,
        "final_shape": final_shape,
        "final_pressure": final_pressure,
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


def landscape_values(test, fixed_value, x_range, y_range, grid_size, A, N, volume_points):
    """Evaluate pressure and volume over the plotted ranges."""
    x_name, y_name = test["free"]
    x_values = values_for_axis(x_name, x_range, grid_size)
    y_values = values_for_axis(y_name, y_range, grid_size)
    pressure = np.empty((len(y_values), len(x_values)), dtype=float)
    volume_error = np.empty_like(pressure)

    target_volume = volume_from_shape(TARGET_SHAPE, point_count=volume_points)

    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            shape = shape_from_free_values(
                (x_value, y_value),
                free_names=(x_name, y_name),
                fixed_name=test["fixed"],
                fixed_value=fixed_value,
            )
            pressure[row, column] = pressure_from_shape(shape, A=A, N=N)
            volume_error[row, column] = (
                volume_from_shape(shape, point_count=volume_points) - target_volume
            )

    return x_values, y_values, pressure, volume_error


def plot_landscape(test_name, run, grid_size, output_dir, A, N, volume_points):
    """Plot one pressure landscape, volume-limit contour, and solve path."""
    test = TESTS[test_name]
    x_name, y_name = run["free_names"]
    path = run["path"]
    x_range = plot_range_for_path(x_name, path[:, 0])
    y_range = plot_range_for_path(y_name, path[:, 1])

    x_values, y_values, pressure, volume_error = landscape_values(
        test,
        fixed_value=run["fixed_value"],
        x_range=x_range,
        y_range=y_range,
        grid_size=grid_size,
        A=A,
        N=N,
        volume_points=volume_points,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / test["output"]
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    finite_pressure = pressure[np.isfinite(pressure)]
    if finite_pressure.size == 0:
        ax.text(0.5, 0.5, "No finite pressure values", ha="center", va="center")
    else:
        filled = ax.contourf(x_values, y_values, pressure, levels=35, cmap="viridis")
        lines = ax.contour(x_values, y_values, pressure, levels=12, colors="black", alpha=0.25)
        ax.clabel(lines, inline=True, fontsize=8)
        fig.colorbar(filled, ax=ax, label="physical normalized pressure")

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
    print(f"\n{test_name}")
    print(f"  fixed {run['fixed_name']}: {run['fixed_value']:.8g}")
    print(f"  optimizer success: {bool(result.success)}")
    print(f"  optimizer message: {result.message}")
    print(f"  final epsilon: {epsilon:.8g}")
    print(f"  final kappa:   {kappa:.8g}")
    print(f"  final delta:   {delta:.8g}")
    print(f"  starting physical normalized pressure: {run['initial_pressure']:.8g}")
    print(f"  final physical normalized pressure: {run['final_pressure']:.8g}")
    print(f"  maximum allowed volume: {target_volume:.8g}")
    print(f"  starting volume: {run['initial_volume']:.8g}")
    print(f"  final volume: {run['final_volume']:.8g}")
    print(f"  final volume margin: {run['final_volume_margin']:.8g}")
    print(f"  volume constraint satisfied: {run['final_volume'] <= target_volume + 1e-7}")
    print(f"  plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maximize physical normalized pressure with a volume upper bound."
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

    target_volume = volume_from_shape(TARGET_SHAPE, point_count=args.volume_points)
    test_names = list(TESTS) if args.test == "all" else [args.test]

    print("volume upper bound")
    print(f"  epsilon: {TARGET_SHAPE[0]:.8g}")
    print(f"  kappa:   {TARGET_SHAPE[1]:.8g}")
    print(f"  delta:   {TARGET_SHAPE[2]:.8g}")
    print(f"  volume:  {target_volume:.8g}")

    for test_name in test_names:
        run = maximize_two_parameters_with_volume_limit(
            TESTS[test_name],
            target_volume=target_volume,
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
            A=args.A,
            N=args.N,
            volume_points=args.volume_points,
        )
        print_run_summary(test_name, run, output_path, target_volume)


if __name__ == "__main__":
    main()
