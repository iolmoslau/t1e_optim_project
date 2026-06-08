#!/usr/bin/env python3
"""Optimize normalized pressure with a volume upper bound.

This script solves the inequality-constrained problem using all three shape
parameters:

    maximize physical normalized pressure(epsilon, kappa, delta)
    subject to volume(epsilon, kappa, delta) <= target_volume

The target volume is the elliptical toroid with

    epsilon = 0.45
    kappa   = 1.9
    delta   = 0

The pressure and volume calculations are written locally in JAX.
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


PARAMETER_NAMES = ("epsilon", "kappa", "delta")

DEFAULT_STARTING_SHAPE = np.array([0.275, 1.35, 0.0], dtype=float)
TARGET_SHAPE = np.array([0.45, 1.9, 0.0], dtype=float)

DEFAULT_A = -0.05
DEFAULT_N = 60
DEFAULT_VOLUME_POINTS = 512
DEFAULT_MAXITER = 80
DEFAULT_PLOT = Path("optimal_JAX/output/optimal_constrainted_norm_p_flux_contours.png")
DEFAULT_PLOT_GRID_SIZE = 600
DEFAULT_CONTOUR_COUNT = 20
BAD_OBJECTIVE_VALUE = 1e100

PARAMETER_BOUNDS = {
    "epsilon": (0.020001, 0.949),
    "kappa": (0.050001, 12.0),
    "delta": (-0.949, 0.949),
}


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


def psi_value(x, y, epsilon, kappa, delta, A):
    """Evaluate the local JAX flux function."""
    coefficients = solve_coefficients(epsilon, kappa, delta, A)
    return jnp.tensordot(coefficients, basis_values(x, y), axes=(0, 0)) + particular_value(
        x, y, A
    )


def normalized_pressure_jax(shape, A=DEFAULT_A, N=DEFAULT_N):
    """Positive physical normalized pressure on a simple masking grid."""
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
    """Boundary points for the volume calculation."""
    epsilon, kappa, delta = shape
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(point_count) + 1)
    alpha = jnp.arcsin(delta)
    x = 1.0 + epsilon * jnp.cos(theta + alpha * jnp.sin(theta))
    y = kappa * epsilon * jnp.sin(theta)
    return x, y


def volume_jax(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Dimensionless toroidal volume factor int x dA."""
    x, y = miller_boundary(shape, point_count=point_count)
    x_mid = 0.5 * (x[:-1] + x[1:])
    y_mid = 0.5 * (y[:-1] + y[1:])
    dx = x[1:] - x[:-1]
    return -jnp.sum(x_mid * y_mid * dx)


def shape_is_valid(shape):
    """Check that the three shape parameters are usable."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and PARAMETER_BOUNDS["epsilon"][0] <= epsilon <= PARAMETER_BOUNDS["epsilon"][1]
        and PARAMETER_BOUNDS["kappa"][0] <= kappa <= PARAMETER_BOUNDS["kappa"][1]
        and PARAMETER_BOUNDS["delta"][0] <= delta <= PARAMETER_BOUNDS["delta"][1]
    )


def pressure_from_shape(shape, A=DEFAULT_A, N=DEFAULT_N):
    """Evaluate pressure from ordinary NumPy values."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = normalized_pressure_jax(jnp.asarray(shape, dtype=jnp.float64), float(A), int(N))
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
    """Positive means the volume is below the allowed maximum."""
    return target_volume - volume_jax(shape, point_count=int(volume_points))


def plot_flux_contours(
    shape,
    A=DEFAULT_A,
    output_path=DEFAULT_PLOT,
    grid_size=DEFAULT_PLOT_GRID_SIZE,
    contour_count=DEFAULT_CONTOUR_COUNT,
    show=False,
):
    """Plot Solov'ev flux contours with the same style as main.py."""
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


def optimize_shape(
    start_shape,
    target_volume,
    A=DEFAULT_A,
    N=DEFAULT_N,
    maxiter=DEFAULT_MAXITER,
    volume_points=DEFAULT_VOLUME_POINTS,
):
    """Optimize epsilon, kappa, and delta together."""
    start_shape = np.asarray(start_shape, dtype=float)
    path = [start_shape.copy()]

    value_and_gradient = jax.value_and_grad(
        lambda shape: -normalized_pressure_jax(shape, A=float(A), N=int(N))
    )
    margin_and_gradient = jax.value_and_grad(
        lambda shape: volume_margin_jax(
            shape,
            target_volume=target_volume,
            volume_points=int(volume_points),
        )
    )

    def loss_and_gradient(shape):
        if not shape_is_valid(shape):
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)
        try:
            value, gradient = value_and_gradient(jnp.asarray(shape, dtype=jnp.float64))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)
        return value, gradient

    def volume_constraint(shape):
        if not shape_is_valid(shape):
            return -BAD_OBJECTIVE_VALUE
        try:
            margin, _ = margin_and_gradient(jnp.asarray(shape, dtype=jnp.float64))
            return float(margin)
        except Exception:
            return -BAD_OBJECTIVE_VALUE

    def volume_constraint_gradient(shape):
        try:
            _, gradient = margin_and_gradient(jnp.asarray(shape, dtype=jnp.float64))
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.zeros(3, dtype=float)
        if not np.all(np.isfinite(gradient)):
            return np.zeros(3, dtype=float)
        return gradient

    def remember_step(shape):
        path.append(np.asarray(shape, dtype=float).copy())

    result = minimize(
        loss_and_gradient,
        start_shape,
        method="SLSQP",
        jac=True,
        bounds=[PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES],
        constraints=[
            {
                "type": "eq",
                "fun": volume_constraint,
                "jac": volume_constraint_gradient,
            }
        ],
        callback=remember_step,
        options={"ftol": 1e-8, "maxiter": int(maxiter)},
    )

    if not np.allclose(path[-1], result.x):
        path.append(np.asarray(result.x, dtype=float).copy())

    final_pressure = pressure_from_shape(result.x, A=A, N=N)
    final_volume = volume_from_shape(result.x, point_count=volume_points)

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "initial_shape": start_shape,
        "initial_pressure": pressure_from_shape(start_shape, A=A, N=N),
        "initial_volume": volume_from_shape(start_shape, point_count=volume_points),
        "final_shape": np.asarray(result.x, dtype=float),
        "final_pressure": final_pressure,
        "final_volume": final_volume,
        "final_volume_margin": target_volume - final_volume,
    }


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
    print()
    print(f"optimizer success: {bool(result.success)}")
    print(f"optimizer message: {result.message}")
    print(f"iterations: {result.nit}")
    print_shape("starting shape", run["initial_shape"])
    print_shape("final shape", run["final_shape"])
    print(f"starting physical normalized pressure: {run['initial_pressure']:.8g}")
    print(f"final physical normalized pressure: {run['final_pressure']:.8g}")
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
        help="Grid resolution for the local JAX normalized-pressure calculation.",
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
        help="Maximum SLSQP iterations.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=DEFAULT_A,
        help="A parameter used by the local normalized-pressure calculation.",
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

    target_volume = volume_from_shape(TARGET_SHAPE, point_count=args.volume_points)

    print("volume upper bound")
    print_shape("target shape", TARGET_SHAPE)
    print(f"  volume: {target_volume:.8g}")

    run = optimize_shape(
        start_shape=start_shape,
        target_volume=target_volume,
        A=args.A,
        N=args.N,
        maxiter=args.maxiter,
        volume_points=args.volume_points,
    )
    print_summary(run, target_volume)
    plot_flux_contours(
        run["final_shape"],
        A=args.A,
        output_path=args.plot,
        grid_size=args.plot_grid_size,
        contour_count=args.contour_count,
    )
    print(f"saved optimized flux contours: {args.plot}")


if __name__ == "__main__":
    main()
