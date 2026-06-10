#!/usr/bin/env python3
"""Constrained beta_t landscapes for the new beta_t objective.

This mirrors landscape_constrainted.py:

1. Fix epsilon, optimize kappa and delta, then plot kappa-delta.
2. Fix kappa, optimize epsilon and delta, then plot epsilon-delta.
3. Fix delta, optimize epsilon and kappa, then plot epsilon-kappa.

The optimization problem is

    maximize beta_t(shape)
    subject to volume(shape) = V_sep
               q_*(shape) >= 2
               kappa <= 2.1

where V_sep is the volume of (epsilon_sep, kappa_sep, delta_sep) from
optimal_new_beta_t.py.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np

try:
    from optimal_JAX import optimal_new_beta_t as beta_opt
except ModuleNotFoundError:
    import optimal_new_beta_t as beta_opt

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize


PARAMETER_NAMES = beta_opt.PARAMETER_NAMES
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}
PARAMETER_LABEL = {
    "epsilon": "epsilon",
    "kappa": "kappa",
    "delta": "delta",
}

# STARTING_POINT = beta_opt.SEP_SHAPE.copy()
STARTING_POINT = np.array([0.275, 1.35, 0.00], dtype=float)
DEFAULT_PLOT_RANGES = {
    "epsilon": (0.10, 0.90),
    "kappa": (0.50, 4.50),
    "delta": (-0.70, 0.80),
}

DEFAULT_N = 500
DEFAULT_GRID_SIZE = 20
DEFAULT_OUTPUT_DIR = Path("optimal_JAX/output/landscape_new_beta_t_output")
DEFAULT_OUTPUT_FILE = "landscape_new_beta_t.png"
SLSQP_OBJECTIVE_SCALE = 1

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


def clip_free_values_to_bounds(free_values, free_names):
    """Clamp SciPy boundary roundoff back inside the declared parameter bounds."""
    clipped = np.asarray(free_values, dtype=float).copy()
    for index, name in enumerate(free_names):
        low, high = beta_opt.PARAMETER_BOUNDS[name]
        clipped[index] = np.clip(clipped[index], low, high)
    return clipped


def landscape_shape_is_valid(shape):
    """Return True when a shape is valid for plotting the objective surface."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    bounds = beta_opt.BASE_PARAMETER_BOUNDS
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and beta_opt.MIN_EPSILON <= epsilon <= beta_opt.MAX_EPSILON
        and bounds["kappa"][0] <= kappa <= bounds["kappa"][1]
        and beta_opt.MIN_DELTA <= delta <= beta_opt.MAX_DELTA
    )


def beta_t_landscape_from_shape(shape, p_0=beta_opt.DEFAULT_P_0, A=beta_opt.DEFAULT_A, N=DEFAULT_N):
    """Evaluate beta_t for plotting without applying optimizer-only constraints."""
    shape = np.asarray(shape, dtype=float)
    if not landscape_shape_is_valid(shape):
        return np.nan
    try:
        value = beta_opt.beta_t_jax(
            jnp.asarray(shape, dtype=jnp.float64),
            p_0=float(p_0),
            A=float(A),
            N=int(N),
        )
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def q_star_landscape_from_shape(shape):
    """Evaluate q_* for plotting without applying optimizer-only constraints."""
    shape = np.asarray(shape, dtype=float)
    if not landscape_shape_is_valid(shape):
        return np.nan
    try:
        value = beta_opt.q_star_jax(jnp.asarray(shape, dtype=jnp.float64))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def volume_landscape_from_shape(shape, point_count=beta_opt.DEFAULT_VOLUME_POINTS):
    """Evaluate volume with the updated beta_t parameter bounds."""
    shape = np.asarray(shape, dtype=float)
    if not landscape_shape_is_valid(shape):
        return np.nan
    try:
        value = beta_opt.volume_jax(jnp.asarray(shape, dtype=jnp.float64), int(point_count))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def beta_t_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
    p_0,
    A,
    N,
):
    """beta_t as a function of only the two free variables."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return beta_opt.beta_t_jax(shape, p_0=float(p_0), A=float(A), N=int(N))


def volume_margin_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
    target_volume,
    volume_points,
):
    """Zero when the shape volume equals V_sep."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return beta_opt.volume_margin_jax(
        shape,
        target_volume=target_volume,
        volume_points=int(volume_points),
    )


def q_star_margin_for_free_values_jax(
    free_values,
    free_names,
    fixed_name,
    fixed_value,
):
    """Positive when q_* satisfies the lower bound."""
    shape = shape_from_free_values_jax(
        free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    return beta_opt.q_star_margin_jax(shape)


def maximize_two_parameters_with_constraints(
    test,
    target_volume,
    p_0=beta_opt.DEFAULT_P_0,
    A=beta_opt.DEFAULT_A,
    N=DEFAULT_N,
    maxiter=beta_opt.DEFAULT_MAXITER,
    volume_points=beta_opt.DEFAULT_VOLUME_POINTS,
):
    """Maximize beta_t while enforcing the new volume and q_* constraints."""
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
    initial_beta_t = beta_opt.beta_t_from_shape(initial_shape, p_0=p_0, A=A, N=N)
    initial_volume = volume_landscape_from_shape(initial_shape, point_count=volume_points)
    initial_q_star = beta_opt.q_star_from_shape(initial_shape)

    value_and_gradient = jax.value_and_grad(
        lambda free_values: -SLSQP_OBJECTIVE_SCALE
        * beta_t_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
            p_0=float(p_0),
            A=float(A),
            N=int(N),
        )
    )
    volume_and_gradient = jax.value_and_grad(
        lambda free_values: volume_margin_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
            target_volume=target_volume,
            volume_points=int(volume_points),
        )
    )
    q_star_and_gradient = jax.value_and_grad(
        lambda free_values: q_star_margin_for_free_values_jax(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
    )

    def loss_and_gradient(free_values):
        trial_shape = shape_from_free_values(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        if not beta_opt.shape_is_valid(trial_shape):
            return beta_opt.BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)

        try:
            value, gradient = value_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return beta_opt.BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return beta_opt.BAD_OBJECTIVE_VALUE, np.zeros(2, dtype=float)
        return value, gradient

    def volume_constraint(free_values):
        trial_shape = shape_from_free_values(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        if not beta_opt.shape_is_valid(trial_shape):
            return -beta_opt.BAD_OBJECTIVE_VALUE
        try:
            margin, _ = volume_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            return float(margin)
        except Exception:
            return -beta_opt.BAD_OBJECTIVE_VALUE

    def volume_constraint_gradient(free_values):
        try:
            _, gradient = volume_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.zeros(2, dtype=float)
        if not np.all(np.isfinite(gradient)):
            return np.zeros(2, dtype=float)
        return gradient

    def q_star_constraint(free_values):
        trial_shape = shape_from_free_values(
            free_values,
            free_names=free_names,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        if not beta_opt.shape_is_valid(trial_shape):
            return -beta_opt.BAD_OBJECTIVE_VALUE
        try:
            margin, _ = q_star_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            return float(margin)
        except Exception:
            return -beta_opt.BAD_OBJECTIVE_VALUE

    def q_star_constraint_gradient(free_values):
        try:
            _, gradient = q_star_and_gradient(jnp.asarray(free_values, dtype=jnp.float64))
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.zeros(2, dtype=float)
        if not np.all(np.isfinite(gradient)):
            return np.zeros(2, dtype=float)
        return gradient

    def remember_step(free_values):
        path.append(clip_free_values_to_bounds(free_values, free_names))

    result = minimize(
        loss_and_gradient,
        start_free_values,
        method="SLSQP",
        jac=True,
        bounds=[beta_opt.PARAMETER_BOUNDS[name] for name in free_names],
        constraints=[
            {
                "type": "eq",
                "fun": volume_constraint,
                "jac": volume_constraint_gradient,
            },
            {
                "type": "ineq",
                "fun": q_star_constraint,
                "jac": q_star_constraint_gradient,
            },
        ],
        callback=remember_step,
        options={"ftol": 1e-8, "maxiter": int(maxiter)},
    )
    final_free_values = clip_free_values_to_bounds(result.x, free_names)
    result.x = final_free_values
    if not np.allclose(path[-1], final_free_values):
        path.append(final_free_values.copy())

    final_shape = shape_from_free_values(
        final_free_values,
        free_names=free_names,
        fixed_name=fixed_name,
        fixed_value=fixed_value,
    )
    final_beta_t = beta_opt.beta_t_from_shape(final_shape, p_0=p_0, A=A, N=N)
    final_volume = volume_landscape_from_shape(final_shape, point_count=volume_points)
    final_q_star = beta_opt.q_star_from_shape(final_shape)

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "fixed_name": fixed_name,
        "fixed_value": fixed_value,
        "free_names": free_names,
        "initial_beta_t": initial_beta_t,
        "initial_volume": initial_volume,
        "initial_q_star": initial_q_star,
        "final_shape": final_shape,
        "final_beta_t": final_beta_t,
        "final_volume": final_volume,
        "final_volume_margin": target_volume - final_volume,
        "final_q_star": final_q_star,
        "final_q_star_margin": final_q_star - beta_opt.MIN_Q_STAR,
    }


def draw_path(ax, path, label="constrained solve path", color="white"):
    """Draw the optimizer path with start and finish markers."""
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        marker="o",
        markeredgecolor="black",
        markersize=4,
        linewidth=2.0,
        label=label,
        zorder=4,
    )
    ax.scatter(
        path[0, 0],
        path[0, 1],
        color="lime",
        edgecolor="black",
        linewidth=1.5,
        s=120,
        label="start",
        zorder=6,
    )
    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        marker="X",
        color="red",
        edgecolor="black",
        linewidth=1.2,
        s=110,
        label="finish",
        zorder=7,
    )


def validate_plot_range(name, value_range):
    """Return a finite increasing plot range."""
    low, high = np.asarray(value_range, dtype=float)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"--{name}-range must be two finite values with LOW < HIGH.")
    return (float(low), float(high))


def should_use_log_axis(name, value_range):
    """Use log spacing when a positive parameter spans a large range."""
    low, high = value_range
    return name in ("epsilon", "kappa") and low > 0 and high / low > 20.0


def values_for_axis(name, value_range, count):
    """Build landscape sample points for one plot axis."""
    if should_use_log_axis(name, value_range):
        return np.geomspace(value_range[0], value_range[1], int(count))
    return np.linspace(*value_range, int(count))


def landscape_values(
    test,
    fixed_value,
    x_range,
    y_range,
    grid_size,
    p_0,
    A,
    N,
    volume_points,
):
    """Evaluate beta_t, volume error, and q_* margin over the plotted ranges."""
    x_name, y_name = test["free"]
    x_values = values_for_axis(x_name, x_range, grid_size)
    y_values = values_for_axis(y_name, y_range, grid_size)
    beta_t = np.empty((len(y_values), len(x_values)), dtype=float)
    volume_error = np.empty_like(beta_t)
    q_star_margin = np.empty_like(beta_t)

    target_volume = beta_opt.sep_volume(point_count=volume_points)

    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            shape = shape_from_free_values(
                (x_value, y_value),
                free_names=(x_name, y_name),
                fixed_name=test["fixed"],
                fixed_value=fixed_value,
            )
            beta_t[row, column] = beta_t_landscape_from_shape(shape, p_0=p_0, A=A, N=N)
            volume_error[row, column] = (
                volume_landscape_from_shape(shape, point_count=volume_points) - target_volume
            )
            q_star_margin[row, column] = (
                q_star_landscape_from_shape(shape) - beta_opt.MIN_Q_STAR
            )

    return x_values, y_values, beta_t, volume_error, q_star_margin


def contour_crosses_zero(values):
    """Return True when a finite array spans zero."""
    finite_values = values[np.isfinite(values)]
    return finite_values.size and np.min(finite_values) <= 0.0 <= np.max(finite_values)


def draw_landscape_panel(
    fig,
    ax,
    test_name,
    run,
    grid_size,
    plot_ranges,
    p_0,
    A,
    N,
    volume_points,
):
    """Draw one beta_t landscape, constraints, and solve path on an axis."""
    test = TESTS[test_name]
    x_name, y_name = run["free_names"]
    path = run["path"]
    x_range = plot_ranges[x_name]
    y_range = plot_ranges[y_name]

    x_values, y_values, beta_t, volume_error, q_star_margin = landscape_values(
        test,
        fixed_value=run["fixed_value"],
        x_range=x_range,
        y_range=y_range,
        grid_size=grid_size,
        p_0=p_0,
        A=A,
        N=N,
        volume_points=volume_points,
    )

    finite_beta_t = beta_t[np.isfinite(beta_t)]
    if finite_beta_t.size == 0:
        ax.text(0.5, 0.5, "No finite beta_t values", ha="center", va="center")
    else:
        filled = ax.contourf(x_values, y_values, beta_t, levels=35, cmap="viridis")
        lines = ax.contour(x_values, y_values, beta_t, levels=12, colors="black", alpha=0.25)
        ax.clabel(lines, inline=True, fontsize=8)
        fig.colorbar(filled, ax=ax, label="beta_t")

    legend_handles = []
    legend_labels = []

    if contour_crosses_zero(volume_error):
        ax.contour(
            x_values,
            y_values,
            volume_error,
            levels=[0.0],
            colors="crimson",
            linewidths=2.5,
        )
        legend_handles.append(Line2D([0], [0], color="crimson", linewidth=2.5))
        legend_labels.append("V = V_sep")

    if contour_crosses_zero(q_star_margin):
        ax.contour(
            x_values,
            y_values,
            q_star_margin,
            levels=[0.0],
            colors="orange",
            linewidths=2.5,
            linestyles="--",
        )
        legend_handles.append(Line2D([0], [0], color="orange", linewidth=2.5, linestyle="--"))
        legend_labels.append("q_* = 2")

    if x_name == "kappa" and x_range[0] <= beta_opt.MAX_KAPPA <= x_range[1]:
        ax.axvline(beta_opt.MAX_KAPPA, color="magenta", linewidth=2.0, linestyle=":")
        legend_handles.append(Line2D([0], [0], color="magenta", linewidth=2.0, linestyle=":"))
        legend_labels.append("kappa = 2.1")
    if y_name == "kappa" and y_range[0] <= beta_opt.MAX_KAPPA <= y_range[1]:
        ax.axhline(beta_opt.MAX_KAPPA, color="magenta", linewidth=2.0, linestyle=":")
        legend_handles.append(Line2D([0], [0], color="magenta", linewidth=2.0, linestyle=":"))
        legend_labels.append("kappa = 2.1")

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
    handles.extend(legend_handles)
    labels.extend(legend_labels)
    ax.legend(handles, labels, loc="best")


def plot_landscapes(
    runs_by_test_name,
    grid_size,
    output_dir,
    plot_ranges,
    p_0,
    A,
    N,
    volume_points,
):
    """Save selected beta_t landscapes as subplots in one PNG."""
    test_names = list(runs_by_test_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_OUTPUT_FILE
    figure_width = max(8.0, 6.0 * len(test_names))
    fig, axes = plt.subplots(
        1,
        len(test_names),
        figsize=(figure_width, 5.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, test_name in zip(axes, test_names):
        draw_landscape_panel(
            fig,
            ax,
            test_name,
            runs_by_test_name[test_name],
            grid_size=grid_size,
            plot_ranges=plot_ranges,
            p_0=p_0,
            A=A,
            N=N,
            volume_points=volume_points,
        )

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
    print(f"  starting beta_t: {run['initial_beta_t']:.8g}")
    print(f"  final beta_t: {run['final_beta_t']:.8g}")
    print(f"  V_sep: {target_volume:.8g}")
    print(f"  starting volume: {run['initial_volume']:.8g}")
    print(f"  final volume: {run['final_volume']:.8g}")
    print(f"  final volume margin: {run['final_volume_margin']:.8g}")
    print(f"  starting q_*: {run['initial_q_star']:.8g}")
    print(f"  final q_*: {run['final_q_star']:.8g}")
    print(f"  final q_* margin: {run['final_q_star_margin']:.8g}")
    print(f"  volume constraint satisfied: {abs(run['final_volume_margin']) <= 1e-7}")
    print(f"  q_* constraint satisfied: {run['final_q_star'] >= beta_opt.MIN_Q_STAR - 1e-7}")
    print(f"  kappa constraint satisfied: {kappa <= beta_opt.MAX_KAPPA + 1e-7}")
    print(f"  plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot constrained beta_t landscapes and optimizer paths."
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
        "--epsilon-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_PLOT_RANGES["epsilon"],
        help="Plot range to use whenever epsilon is an axis.",
    )
    parser.add_argument(
        "--kappa-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_PLOT_RANGES["kappa"],
        help="Plot range to use whenever kappa is an axis.",
    )
    parser.add_argument(
        "--delta-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_PLOT_RANGES["delta"],
        help="Plot range to use whenever delta is an axis.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Grid resolution for the local JAX beta_t pressure average.",
    )
    parser.add_argument(
        "--volume-points",
        type=int,
        default=beta_opt.DEFAULT_VOLUME_POINTS,
        help="Number of boundary points used for the volume calculation.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=beta_opt.DEFAULT_MAXITER,
        help="Maximum SLSQP iterations for each constrained solve.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=beta_opt.DEFAULT_A,
        help="A parameter used by the local flux calculation.",
    )
    parser.add_argument(
        "--p-0",
        type=float,
        default=beta_opt.DEFAULT_P_0,
        help="Pressure scale p_0 used in <p> = p_0 Psi / Psi_min.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the combined landscape plot is saved.",
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
    target_volume = beta_opt.sep_volume(point_count=args.volume_points)
    test_names = list(TESTS) if args.test == "all" else [args.test]

    print("fixed volume target")
    print(f"  epsilon_sep: {beta_opt.epsilon_sep:.8g}")
    print(f"  kappa_sep:   {beta_opt.kappa_sep:.8g}")
    print(f"  delta_sep:   {beta_opt.delta_sep:.8g}")
    print(f"  V_sep:       {target_volume:.8g}")
    print(f"  q_* minimum: {beta_opt.MIN_Q_STAR:.8g}")
    print(f"  kappa max:   {beta_opt.MAX_KAPPA:.8g}")

    runs_by_test_name = {}
    for test_name in test_names:
        run = maximize_two_parameters_with_constraints(
            TESTS[test_name],
            target_volume=target_volume,
            p_0=args.p_0,
            A=args.A,
            N=args.N,
            maxiter=args.maxiter,
            volume_points=args.volume_points,
        )
        runs_by_test_name[test_name] = run

    output_path = plot_landscapes(
        runs_by_test_name,
        grid_size=args.grid_size,
        output_dir=args.output_dir,
        plot_ranges=plot_ranges,
        p_0=args.p_0,
        A=args.A,
        N=args.N,
        volume_points=args.volume_points,
    )

    for test_name, run in runs_by_test_name.items():
        print_run_summary(test_name, run, output_path, target_volume)


if __name__ == "__main__":
    main()
