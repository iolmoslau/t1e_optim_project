#!/usr/bin/env python3
"""Optimize one pressure objective with a volume upper bound.

A shape is the triple [epsilon, kappa, delta].  This script asks:

    maximize objective(epsilon, kappa, delta)
    subject to volume(epsilon, kappa, delta) = target_volume

The target volume comes from this reference elliptical toroid:

    epsilon = 0.45
    kappa   = 1.9
    delta   = 0

The file has three layers:

1. Solov'ev formulas turn a shape into flux, pressure, and volume.
2. Small wrapper functions choose which objective to score.
3. optimize_shape asks SciPy SLSQP to improve the shape without exceeding
   the target volume.

The long algebraic formulas intentionally keep their equation-like symbols.
The optimizer code below uses plainer names because that is the main workflow.
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
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

PARAMETER_NAMES = ("epsilon", "kappa", "delta")

# Default run: start from a smaller toroid and cap volume at TARGET_SHAPE.
DEFAULT_STARTING_SHAPE = np.array([0.275, 1.35, 0.0], dtype=float)
TARGET_SHAPE = np.array([0.45, 1.9, 0.0], dtype=float)

DEFAULT_A = -0.05
DEFAULT_N = 500
DEFAULT_VOLUME_POINTS = 512
DEFAULT_MAXITER = 80
DEFAULT_PLOT = Path("optimal_JAX/output/optimal_constrainted_norm_p_flux_contours.png")
DEFAULT_PLOT_GRID_SIZE = 600
DEFAULT_CONTOUR_COUNT = 20
DEFAULT_Q = 2.0

# If SciPy tries an invalid shape, return a huge value so it backs away.
BAD_OBJECTIVE_VALUE = 1e100
NORMALIZED_OBJECTIVE = "normalized_pressure"
BETA_T_OBJECTIVE = "beta_toroidal"

# Physical parameter limits used by the optimizer.
PARAMETER_BOUNDS = {
    "epsilon": (0.020001, 0.949),
    "kappa": (0.050001, 12.0),
    "delta": (-0.949, 0.949),
}


# ---------------------------------------------------------------------------
# Solov'ev flux formulas
# ---------------------------------------------------------------------------

def safe_log(x):
    """Logarithm used by the Solov'ev basis."""
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


# These helpers ask JAX for the derivatives used in the boundary equations.
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
    """Find coefficients that make the flux match the requested boundary."""
    alpha = jnp.arcsin(delta)
    curv1 = -((1.0 + alpha) ** 2) / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * jnp.cos(alpha) ** 2)
    curv3 = ((1.0 - alpha) ** 2) / (epsilon * kappa**2)

    x_outer = 1.0 + epsilon
    x_inner = 1.0 - epsilon
    x_high = 1.0 - epsilon * delta
    y_high = kappa * epsilon
    zero = jnp.array(0.0, dtype=jnp.float64)

    matrix = jnp.stack(
        [
            basis_values(x_outer, zero),
            basis_values(x_inner, zero),
            basis_values(x_high, y_high),
            basis_x(x_high, y_high),
            curv1 * basis_x(x_outer, zero) + basis_yy(x_outer, zero),
            curv3 * basis_x(x_inner, zero) + basis_yy(x_inner, zero),
            curv2 * basis_y(x_high, y_high) + basis_xx(x_high, y_high),
        ]
    )

    right_hand_side = -jnp.stack(
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

    return jnp.linalg.solve(matrix, right_hand_side)


# ---------------------------------------------------------------------------
# Objective and volume calculations
# ---------------------------------------------------------------------------

def psi_value(x, y, epsilon, kappa, delta, A):
    """Evaluate the local JAX flux function."""
    coefficients = solve_coefficients(epsilon, kappa, delta, A)
    return jnp.tensordot(coefficients, basis_values(x, y), axes=(0, 0)) + particular_value(
        x, y, A
    )


def normalized_pressure_jax(shape, A=DEFAULT_A, N=DEFAULT_N):
    """Score one shape by physical normalized pressure.

    Higher is better.  The grid is a simple numerical approximation: build a
    rectangle around the shape, keep points inside the plasma, then average the
    normalized flux over those points.
    """
    epsilon, kappa, delta = shape
    x = jnp.linspace(1.0 - epsilon, 1.0 + epsilon, int(N))
    y = jnp.linspace(-kappa * epsilon, kappa * epsilon, int(N))
    X, Y = jnp.meshgrid(x, y, indexing="xy")

    flux = psi_value(X, Y, epsilon, kappa, delta, A)
    inside_plasma = flux <= 0.0
    inside_weight = inside_plasma.astype(jnp.float64)

    lowest_flux = jnp.min(jnp.where(inside_plasma, flux, jnp.inf))
    normalizing_flux = jnp.abs(lowest_flux)

    dx = (x[-1] - x[0]) / (int(N) - 1)
    dy = (y[-1] - y[0]) / (int(N) - 1)
    cell_area = dx * dy

    normalized_flux = flux / normalizing_flux
    numerator = cell_area * jnp.sum(X * normalized_flux * inside_weight)
    denominator = cell_area * jnp.sum(X * inside_weight)

    return -numerator 

#/ denominator
def int_contour_boundary_jax(values, x_points):
    """Green's theorem line integral used by pressure_utils.py."""
    dx = x_points[1:] - x_points[:-1]
    return -jnp.sum(values * dx)


def poloidal_circum_jax(x_points, y_points):
    """Approximate the poloidal circumference of a closed boundary."""
    dx = x_points[1:] - x_points[:-1]
    dy = y_points[1:] - y_points[:-1]
    return jnp.sum(jnp.sqrt(dx * dx + dy * dy))


def G_total_jax(x, y, A, coefficients):
    """Antiderivative formula used in the toroidal-beta calculation."""
    log_x = safe_log(x)
    x2 = x * x
    y2 = y * y

    G_base = x**5 * y / 8.0
    G_A = x**3 * y * (-x2 + 4.0 * log_x) / 8.0
    G_1 = x * y
    G_2 = x**3 * y
    G_3 = x * y * (-x2 * log_x + y2 / 3.0)
    G_4 = x**3 * y * (x2 - 4.0 * y2 / 3.0)
    G_5 = x * y * (
        15.0 * x2 * x2 * log_x
        + x2 * y2 * (-20.0 * log_x - 15.0)
        + 2.0 * y2 * y2
    ) / 5.0
    G_6 = x**3 * y * (x2 * x2 - 4.0 * x2 * y2 + 8.0 * y2 * y2 / 5.0)
    G_7 = x * y * (
        -105.0 * x2 * x2 * x2 * log_x
        + x2 * x2 * y2 * (420.0 * log_x + 175.0)
        + x2 * y2 * y2 * (-168.0 * log_x - 196.0)
        + 8.0 * y2 * y2 * y2
    ) / 7.0

    return (
        G_base
        + A * G_A
        + coefficients[0] * G_1
        + coefficients[1] * G_2
        + coefficients[2] * G_3
        + coefficients[3] * G_4
        + coefficients[4] * G_5
        + coefficients[5] * G_6
        + coefficients[6] * G_7
    )


def miller_boundary(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Return points along the closed Miller boundary for one shape."""
    epsilon, kappa, delta = shape
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(point_count) + 1)
    alpha = jnp.arcsin(delta)
    x = 1.0 + epsilon * jnp.cos(theta + alpha * jnp.sin(theta))
    y = kappa * epsilon * jnp.sin(theta)
    return x, y


def volume_jax(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Estimate the dimensionless toroidal volume from the boundary curve."""
    x, y = miller_boundary(shape, point_count=point_count)
    x_mid = 0.5 * (x[:-1] + x[1:])
    y_mid = 0.5 * (y[:-1] + y[1:])
    dx = x[1:] - x[:-1]
    return -jnp.sum(x_mid * y_mid * dx)


def beta_toroidal_jax(shape, A=DEFAULT_A, q=DEFAULT_Q, N=DEFAULT_VOLUME_POINTS):
    """Score one shape by toroidal beta."""
    epsilon, kappa, delta = shape
    x_points, y_points = miller_boundary(shape, point_count=int(N))
    x_mid = 0.5 * (x_points[:-1] + x_points[1:])
    y_mid = 0.5 * (y_points[:-1] + y_points[1:])

    coefficients = solve_coefficients(epsilon, kappa, delta, A)
    circum = poloidal_circum_jax(x_points, y_points)
    volume = int_contour_boundary_jax(x_mid * y_mid, x_points)
    psi_integral = int_contour_boundary_jax(
        G_total_jax(x_mid, y_mid, A, coefficients),
        x_points,
    )
    factor = int_contour_boundary_jax(
        y_mid * (A / x_mid + (1.0 - A) * x_mid),
        x_points,
    )

    beta_poloidal = -2.0 * (1.0 - A) * (circum**2 / volume) * psi_integral * factor**-2
    return epsilon**2 * beta_poloidal / q**2


# ---------------------------------------------------------------------------
# Plain NumPy wrappers used outside JAX tracing
# ---------------------------------------------------------------------------

def shape_is_valid(shape):
    """Return True when the shape has finite values inside the allowed limits."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and PARAMETER_BOUNDS["epsilon"][0] <= epsilon <= PARAMETER_BOUNDS["epsilon"][1]
        and PARAMETER_BOUNDS["kappa"][0] <= kappa <= PARAMETER_BOUNDS["kappa"][1]
        and PARAMETER_BOUNDS["delta"][0] <= delta <= PARAMETER_BOUNDS["delta"][1]
    )


def objective_label(objective_name):
    """Readable objective name for printed output and plots."""
    if objective_name == BETA_T_OBJECTIVE:
        return "beta_toroidal"
    return "normalized pressure"


def objective_slug(objective_name):
    """Filename-safe objective name."""
    if objective_name == BETA_T_OBJECTIVE:
        return "beta_toroidal"
    return "normalized_pressure"


def plot_path_for_objective(path, objective_name):
    """Add the objective name to the output PNG filename."""
    path = Path(path)
    slug = objective_slug(objective_name)
    suffix = path.suffix or ".png"
    if slug in path.stem:
        return path.with_suffix(suffix)
    return path.with_name(f"{path.stem}_{slug}{suffix}")


def objective_value_jax(shape, objective_name, A=DEFAULT_A, N=DEFAULT_N):
    """Evaluate the selected objective for JAX optimization."""
    if objective_name == BETA_T_OBJECTIVE:
        return beta_toroidal_jax(shape, A=A, q=DEFAULT_Q, N=N)
    return normalized_pressure_jax(shape, A=A, N=N)


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
    """Evaluate volume from ordinary NumPy values."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = volume_jax(jnp.asarray(shape, dtype=jnp.float64), int(point_count))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def volume_margin_jax(shape, target_volume, volume_points):
    """Return the remaining volume room; positive means under the cap."""
    return target_volume - volume_jax(shape, point_count=int(volume_points))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_flux_contours(
    shape,
    objective_name=NORMALIZED_OBJECTIVE,
    A=DEFAULT_A,
    output_path=DEFAULT_PLOT,
    grid_size=DEFAULT_PLOT_GRID_SIZE,
    contour_count=DEFAULT_CONTOUR_COUNT,
    show=False,
):
    """Save a contour plot of the final optimized flux."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)

    x_min = max(np.finfo(float).tiny, 1.0 - epsilon - 0.05)
    x = np.linspace(x_min, 1.0 + epsilon + 0.1, int(grid_size))
    y = np.linspace(-kappa * epsilon - 0.05, kappa * epsilon + 0.025, int(grid_size))
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(
        psi_value(
            jnp.asarray(X, dtype=jnp.float64),
            jnp.asarray(Y, dtype=jnp.float64),
            float(epsilon),
            float(kappa),
            float(delta),
            float(A),
        ),
        dtype=float,
    )

    z_min = float(np.nanmin(Z))
    if z_min < 0:
        contour_levels = np.linspace(z_min, 0.0, int(contour_count))
    else:
        contour_levels = int(contour_count)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.contour(X, Y, Z, levels=contour_levels, cmap="jet")
    ax.axvline(x=0.0, linestyle="--", color="black")
    ax.set_xlabel("$R/R_{0}$", fontsize=14)
    ax.set_ylabel("$Z/R_{0}$", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1.0 + epsilon + 0.25)
    ax.set_ylim(-kappa * epsilon - 0.2, kappa * epsilon + 0.2)
    ax.set_title(
        f"{objective_label(objective_name)}: "
        f"epsilon={epsilon:.4g}, kappa={kappa:.4g}, delta={delta:.4g}, A={A:.4g}"
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize_shape(
    start_shape,
    target_volume,
    objective_name=NORMALIZED_OBJECTIVE,
    A=DEFAULT_A,
    N=DEFAULT_N,
    maxiter=DEFAULT_MAXITER,
    volume_points=DEFAULT_VOLUME_POINTS,
):
    """Find the best shape while keeping the volume at or below target_volume.

    SciPy minimizes functions, so this code minimizes the negative objective.
    That is the same as maximizing the physical objective reported to the user.
    The volume rule is an upper bound: target_volume - current_volume must stay
    non-negative.
    """
    start_shape = np.asarray(start_shape, dtype=float)
    path = [start_shape.copy()]

    # SciPy needs both each score and its local slope.  JAX supplies the slope.
    negative_objective_with_gradient = jax.value_and_grad(
        lambda shape: -objective_value_jax(
            shape,
            objective_name,
            A=float(A),
            N=int(N),
        )
    )
    volume_room_with_gradient = jax.value_and_grad(
        lambda shape: volume_margin_jax(
            shape,
            target_volume=target_volume,
            volume_points=int(volume_points),
        )
    )

    def negative_objective_and_gradient(shape):
        if not shape_is_valid(shape):
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)
        try:
            value, gradient = negative_objective_with_gradient(
                jnp.asarray(shape, dtype=jnp.float64)
            )
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)
        return value, gradient

    def volume_room(shape):
        if not shape_is_valid(shape):
            return -BAD_OBJECTIVE_VALUE
        try:
            room, _ = volume_room_with_gradient(jnp.asarray(shape, dtype=jnp.float64))
            return float(room)
        except Exception:
            return -BAD_OBJECTIVE_VALUE

    def volume_room_gradient(shape):
        try:
            _, gradient = volume_room_with_gradient(jnp.asarray(shape, dtype=jnp.float64))
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.zeros(3, dtype=float)
        if not np.all(np.isfinite(gradient)):
            return np.zeros(3, dtype=float)
        return gradient

    def save_optimizer_step(shape):
        path.append(np.asarray(shape, dtype=float).copy())

    result = minimize(
        negative_objective_and_gradient,
        start_shape,
        method="SLSQP",
        jac=True,
        bounds=[PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES],
        constraints=[
            {
                # SLSQP requires equality constraints to be = 0.
                "type": "eq",
                "fun": volume_room,
                "jac": volume_room_gradient,
            }
        ],
        callback=save_optimizer_step,
        options={"ftol": 1e-8, "maxiter": int(maxiter)},
    )

    if not np.allclose(path[-1], result.x):
        path.append(np.asarray(result.x, dtype=float).copy())

    final_objective = objective_from_shape(result.x, objective_name, A=A, N=N)
    final_volume = volume_from_shape(result.x, point_count=volume_points)

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "objective_name": objective_name,
        "initial_shape": start_shape,
        "initial_objective": objective_from_shape(start_shape, objective_name, A=A, N=N),
        "initial_volume": volume_from_shape(start_shape, point_count=volume_points),
        "final_shape": np.asarray(result.x, dtype=float),
        "final_objective": final_objective,
        "final_volume": final_volume,
        "final_volume_margin": target_volume - final_volume,
    }


# ---------------------------------------------------------------------------
# Command-line output
# ---------------------------------------------------------------------------

def print_shape(label, shape):
    """Print one shape in a readable three-line block."""
    epsilon, kappa, delta = shape
    print(label)
    print(f"  epsilon: {epsilon:.8g}")
    print(f"  kappa:   {kappa:.8g}")
    print(f"  delta:   {delta:.8g}")


def print_summary(run, target_volume):
    """Print a compact optimization summary."""
    result = run["result"]
    label = objective_label(run["objective_name"])
    print()
    print(f"objective: {label}")
    print(f"optimizer success: {bool(result.success)}")
    print(f"optimizer message: {result.message}")
    print(f"iterations: {result.nit}")
    print_shape("starting shape", run["initial_shape"])
    print_shape("final shape", run["final_shape"])
    print(f"starting {label}: {run['initial_objective']:.8g}")
    print(f"final {label}: {run['final_objective']:.8g}")
    print(f"maximum allowed volume: {target_volume:.8g}")
    print(f"starting volume: {run['initial_volume']:.8g}")
    print(f"final volume: {run['final_volume']:.8g}")
    print(f"final volume margin: {run['final_volume_margin']:.8g}")
    print(f"volume constraint satisfied: {run['final_volume'] <= target_volume + 1e-7}")
    print(f"path length: {len(run['path'])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize all three shape parameters with a volume upper bound."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Number of grid points per direction for normalized-pressure scoring.",
    )
    parser.add_argument(
        "--volume-points",
        type=int,
        default=DEFAULT_VOLUME_POINTS,
        help="Number of points used to trace the boundary for volume scoring.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help="Maximum SLSQP iterations.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=DEFAULT_A,
        help="A parameter used by the local normalized-pressure calculation.",
    )
    parser.add_argument(
        "--beta_t",
        action="store_true",
        help="Maximize beta_toroidal instead of normalized pressure.",
    )
    parser.add_argument(
        "--start-epsilon",
        type=float,
        default=DEFAULT_STARTING_SHAPE[0],
        help="Starting epsilon.",
    )
    parser.add_argument(
        "--start-kappa",
        type=float,
        default=DEFAULT_STARTING_SHAPE[1],
        help="Starting kappa.",
    )
    parser.add_argument(
        "--start-delta",
        type=float,
        default=DEFAULT_STARTING_SHAPE[2],
        help="Starting delta.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="PNG path for the optimized flux contours.",
    )
    parser.add_argument(
        "--plot-grid-size",
        type=int,
        default=DEFAULT_PLOT_GRID_SIZE,
        help="Grid size for the optimized flux contour plot.",
    )
    parser.add_argument(
        "--contour-count",
        type=int,
        default=DEFAULT_CONTOUR_COUNT,
        help="Number of optimized flux contour levels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.N < 3:
        raise ValueError("--N must be at least 3.")
    if args.volume_points < 16:
        raise ValueError("--volume-points must be at least 16.")
    if args.plot_grid_size < 3:
        raise ValueError("--plot-grid-size must be at least 3.")
    if args.contour_count < 1:
        raise ValueError("--contour-count must be at least 1.")

    start_shape = np.array(
        [args.start_epsilon, args.start_kappa, args.start_delta],
        dtype=float,
    )
    if not shape_is_valid(start_shape):
        raise ValueError("The starting shape is outside the allowed parameter bounds.")

    objective_name = BETA_T_OBJECTIVE if args.beta_t else NORMALIZED_OBJECTIVE
    plot_path = plot_path_for_objective(args.plot, objective_name)
    target_volume = volume_from_shape(TARGET_SHAPE, point_count=args.volume_points)

    print("volume upper bound")
    print_shape("target shape", TARGET_SHAPE)
    print(f"  volume: {target_volume:.8g}")
    print(f"objective: {objective_label(objective_name)}")

    run = optimize_shape(
        start_shape=start_shape,
        target_volume=target_volume,
        objective_name=objective_name,
        A=args.A,
        N=args.N,
        maxiter=args.maxiter,
        volume_points=args.volume_points,
    )
    print_summary(run, target_volume)
    plot_flux_contours(
        run["final_shape"],
        objective_name=objective_name,
        A=args.A,
        output_path=plot_path,
        grid_size=args.plot_grid_size,
        contour_count=args.contour_count,
    )
    print(f"saved optimized flux contours: {plot_path}")


if __name__ == "__main__":
    main()
