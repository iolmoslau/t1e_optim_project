#!/usr/bin/env python3
"""Unconstrained objective landscapes matching optimal_unconstrainted.py."""

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
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for source_dir in (REPO_ROOT, REPO_ROOT / "pressure_integral", REPO_ROOT / "ITER_Equilibria"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from optimal_JAX.optimal_unconstrainted import (  # noqa: E402
    BAD_OBJECTIVE_VALUE,
    DEFAULT_A,
    DEFAULT_FINITE_DIFF_STEP,
    DEFAULT_FTOL,
    DEFAULT_GTOL,
    DEFAULT_MAXITER,
    DEFAULT_METHOD,
    DEFAULT_N,
    DEFAULT_POSITIVE_FLOOR,
    DEFAULT_Q,
    OBJECTIVE_CHOICES,
    format_objective,
    objective_value,
    objective_value_jax,
    optimizer_bounds,
)
from optimal_JAX.optimal_unconstrainted_random import objective_label  # noqa: E402


PARAMETER_NAMES = ("epsilon", "kappa", "delta")
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}
PARAMETER_LABEL = {
    "epsilon": "epsilon",
    "kappa": "kappa",
    "delta": "delta",
}

STARTING_POINT = np.array([0.275, 1.35, 0.0], dtype=float)
DEFAULT_GRID_SIZE = 40
DEFAULT_OUTPUT_DIR = Path("optimal_JAX/output/landscape_unconstrainted_output")
DEFAULT_OUTPUT_FILE = "landscape_unconstrainted.png"

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


def objective_from_shape(
    shape,
    objective,
    A=DEFAULT_A,
    method=DEFAULT_METHOD,
    N=DEFAULT_N,
    q=DEFAULT_Q,
):
    """Evaluate the selected objective from ordinary NumPy values."""
    try:
        value = objective_value(shape, objective, A=A, method=method, N=N, q=q)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def objective_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
    objective,
    A,
    method,
    N,
    q,
    finite_diff_step,
    epsilon_floor,
    kappa_floor,
):
    """Selected objective as a function of only the two free variables."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return objective_value_jax(
        shape,
        objective,
        float(A),
        method,
        int(N),
        float(q),
        float(finite_diff_step),
        float(epsilon_floor),
        float(kappa_floor),
    )


def free_bounds(free_names, epsilon_floor, kappa_floor):
    """Return optimal_unconstrainted.py bounds for the selected free variables."""
    bounds_by_name = dict(
        zip(PARAMETER_NAMES, optimizer_bounds(epsilon_floor, kappa_floor))
    )
    return [bounds_by_name[name] for name in free_names]


def maximize_two_parameters(
    test,
    objective,
    A=DEFAULT_A,
    method=DEFAULT_METHOD,
    N=DEFAULT_N,
    q=DEFAULT_Q,
    maxiter=DEFAULT_MAXITER,
    ftol=DEFAULT_FTOL,
    gtol=DEFAULT_GTOL,
    finite_diff_step=DEFAULT_FINITE_DIFF_STEP,
    epsilon_floor=DEFAULT_POSITIVE_FLOOR,
    kappa_floor=DEFAULT_POSITIVE_FLOOR,
):
    """Maximize the selected objective with one fixed shape parameter."""
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
    initial_objective = objective_from_shape(
        initial_shape, objective, A=A, method=method, N=N, q=q
    )

    value_and_gradient = jax.value_and_grad(
        lambda free_values: objective_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
            objective=objective,
            A=float(A),
            method=method,
            N=int(N),
            q=float(q),
            finite_diff_step=float(finite_diff_step),
            epsilon_floor=float(epsilon_floor),
            kappa_floor=float(kappa_floor),
        )
    )

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
        start_free_values,
        method="L-BFGS-B",
        jac=True,
        bounds=free_bounds(free_names, epsilon_floor, kappa_floor),
        callback=remember_step,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
        },
    )
    final_free_values = np.asarray(result.x, dtype=float)
    if not np.allclose(path[-1], final_free_values):
        path.append(final_free_values.copy())

    final_shape = shape_from_free_values(
        final_free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    final_objective = objective_from_shape(
        final_shape, objective, A=A, method=method, N=N, q=q
    )

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "fixed_name": fixed_name,
        "fixed_value": fixed_value,
        "free_names": free_names,
        "initial_objective": initial_objective,
        "final_shape": final_shape,
        "final_objective": final_objective,
    }


def validate_plot_range(name, value_range):
    """Return a finite increasing plot range."""
    low, high = np.asarray(value_range, dtype=float)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"--{name}-range must be two finite values with LOW < HIGH.")
    return (float(low), float(high))


def values_for_axis(value_range, count):
    """Build landscape sample points for one plot axis."""
    return np.linspace(*value_range, int(count))


def landscape_values(test, fixed_value, x_range, y_range, grid_size, objective, A, method, N, q):
    """Evaluate the selected objective over the plotted ranges."""
    x_name, y_name = test["free"]
    x_values = values_for_axis(x_range, grid_size)
    y_values = values_for_axis(y_range, grid_size)
    values = np.empty((len(y_values), len(x_values)), dtype=float)

    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            shape = shape_from_free_values(
                (x_value, y_value),
                free_names=(x_name, y_name),
                fixed_name=test["fixed"],
                fixed_value=fixed_value,
            )
            values[row, column] = objective_from_shape(
                shape, objective, A=A, method=method, N=N, q=q
            )

    return x_values, y_values, values


def draw_path(ax, path):
    """Draw the optimizer path with start and finish markers."""
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="white",
        marker="o",
        markeredgecolor="black",
        linewidth=2.0,
        label="optimizer path",
        zorder=4,
    )
    ax.scatter(path[0, 0], path[0, 1], color="lime", edgecolor="black", s=90, label="start", zorder=6)
    ax.scatter(path[-1, 0], path[-1, 1], color="red", edgecolor="black", s=90, label="finish", zorder=7)


def draw_landscape_panel(fig, ax, test_name, run, grid_size, plot_ranges, objective, A, method, N, q):
    """Draw one objective landscape and solve path on an axis."""
    test = TESTS[test_name]
    x_name, y_name = run["free_names"]
    x_range = plot_ranges[x_name]
    y_range = plot_ranges[y_name]

    x_values, y_values, values = landscape_values(
        test,
        fixed_value=run["fixed_value"],
        x_range=x_range,
        y_range=y_range,
        grid_size=grid_size,
        objective=objective,
        A=A,
        method=method,
        N=N,
        q=q,
    )

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        ax.text(0.5, 0.5, "No finite objective values", ha="center", va="center")
    else:
        filled = ax.contourf(x_values, y_values, values, levels=35, cmap="viridis")
        lines = ax.contour(x_values, y_values, values, levels=12, colors="black", alpha=0.25)
        ax.clabel(lines, inline=True, fontsize=8)
        fig.colorbar(filled, ax=ax, label=objective_label(objective))

    draw_path(ax, run["path"])
    ax.set_xlabel(PARAMETER_LABEL[x_name])
    ax.set_ylabel(PARAMETER_LABEL[y_name])
    ax.set_title(
        f"{PARAMETER_LABEL[x_name]} vs {PARAMETER_LABEL[y_name]} "
        f"with fixed {PARAMETER_LABEL[run['fixed_name']]}={run['fixed_value']:.4g}"
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, color="white", alpha=0.15)
    ax.legend(loc="best")


def plot_landscapes(runs_by_test_name, grid_size, output_dir, plot_ranges, objective, A, method, N, q):
    """Save selected objective landscapes as subplots in one PNG."""
    test_names = list(runs_by_test_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_FILE
    fig, axes = plt.subplots(
        1,
        len(test_names),
        figsize=(max(8.0, 6.0 * len(test_names)), 5.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, test_name in zip(axes, test_names):
        draw_landscape_panel(
            fig,
            ax,
            test_name,
            runs_by_test_name[test_name],
            grid_size,
            plot_ranges,
            objective,
            A,
            method,
            N,
            q,
        )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_landscape(run, test_name, grid_size, output_dir, plot_ranges, objective, A, method, N, q):
    """Save one objective landscape and solve path as its own PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / TESTS[test_name]["output"]
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    draw_landscape_panel(fig, ax, test_name, run, grid_size, plot_ranges, objective, A, method, N, q)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def print_run_summary(test_name, run, output_path, objective):
    """Print the optimization result in a compact form."""
    result = run["result"]
    epsilon, kappa, delta = run["final_shape"]
    label = objective_label(objective)
    print(f"\n{test_name}")
    print(f"  fixed {run['fixed_name']}: {run['fixed_value']:.8g}")
    print(f"  optimizer success: {bool(result.success)}")
    print(f"  optimizer message: {result.message}")
    print(f"  final epsilon: {epsilon:.8g}")
    print(f"  final kappa:   {kappa:.8g}")
    print(f"  final delta:   {delta:.8g}")
    print(f"  starting {label}: {format_objective(run['initial_objective'])}")
    print(f"  final {label}: {format_objective(run['final_objective'])}")
    print(f"  plot: {output_path}")


def parse_args():
    plot_ranges = default_plot_ranges()
    parser = argparse.ArgumentParser(
        description="Plot selected optimal_unconstrainted.py objective landscapes."
    )
    parser.add_argument("--test", choices=("all", *TESTS.keys()), default="all")
    parser.add_argument("--objective", choices=OBJECTIVE_CHOICES, default=OBJECTIVE_CHOICES[0])
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--epsilon-range", type=float, nargs=2, default=plot_ranges["epsilon"])
    parser.add_argument("--kappa-range", type=float, nargs=2, default=plot_ranges["kappa"])
    parser.add_argument("--delta-range", type=float, nargs=2, default=plot_ranges["delta"])
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--method", choices=("contour", "masking"), default=DEFAULT_METHOD)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--ftol", type=float, default=DEFAULT_FTOL)
    parser.add_argument("--gtol", type=float, default=DEFAULT_GTOL)
    parser.add_argument("--finite-diff-step", type=float, default=DEFAULT_FINITE_DIFF_STEP)
    parser.add_argument("--epsilon-floor", type=float, default=DEFAULT_POSITIVE_FLOOR)
    parser.add_argument("--kappa-floor", type=float, default=DEFAULT_POSITIVE_FLOOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-layout",
        choices=("subplots", "separate"),
        default="subplots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.N < 3:
        raise ValueError("--N must be at least 3.")
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.finite_diff_step <= 0.0:
        raise ValueError("--finite-diff-step must be positive.")

    plot_ranges = {
        "epsilon": validate_plot_range("epsilon", args.epsilon_range),
        "kappa": validate_plot_range("kappa", args.kappa_range),
        "delta": validate_plot_range("delta", args.delta_range),
    }
    test_names = list(TESTS) if args.test == "all" else [args.test]

    print(f"objective: {args.objective}")
    print(f"fixed starting point: {STARTING_POINT}")

    runs_by_test_name = {}
    for test_name in test_names:
        run = maximize_two_parameters(
            TESTS[test_name],
            objective=args.objective,
            A=args.A,
            method=args.method,
            N=args.N,
            q=args.q,
            maxiter=args.maxiter,
            ftol=args.ftol,
            gtol=args.gtol,
            finite_diff_step=args.finite_diff_step,
            epsilon_floor=args.epsilon_floor,
            kappa_floor=args.kappa_floor,
        )
        runs_by_test_name[test_name] = run

    if args.plot_layout == "subplots":
        output_path = plot_landscapes(
            runs_by_test_name,
            args.grid_size,
            args.output_dir,
            plot_ranges,
            args.objective,
            args.A,
            args.method,
            args.N,
            args.q,
        )
        for test_name, run in runs_by_test_name.items():
            print_run_summary(test_name, run, output_path, args.objective)
    else:
        for test_name, run in runs_by_test_name.items():
            output_path = plot_landscape(
                run,
                test_name,
                args.grid_size,
                args.output_dir,
                plot_ranges,
                args.objective,
                args.A,
                args.method,
                args.N,
                args.q,
            )
            print_run_summary(test_name, run, output_path, args.objective)


if __name__ == "__main__":
    main()
