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

1. Shared JAX helpers turn a shape into pressure and volume scores.
2. Small wrapper functions choose which objective to score.
3. optimize_shape asks SciPy SLSQP to improve the shape without exceeding
   the target volume.

The long algebraic formulas intentionally keep their equation-like symbols.
The optimizer code below uses plainer names because that is the main workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimal_JAX.plot_contour_from_shape import plot_contour_from_shape  # noqa: E402
from optimal_JAX.utils_JAX import (  # noqa: E402
    beta_t_alternative_jax as utils_beta_t_alternative_jax,
    beta_toroidal_jax as utils_beta_toroidal_jax,
    normalized_psi_pressure_masking_jax,
    volume_jax as utils_volume_jax,
)


# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

PARAMETER_NAMES = ("epsilon", "kappa", "delta")

# Default run: start from a smaller toroid and cap volume at TARGET_SHAPE.
DEFAULT_STARTING_SHAPE = np.array([0.375, 1.7, 0.15], dtype=float)
TARGET_SHAPE = np.array([0.45, 1.9, 0.0], dtype=float)

DEFAULT_A = -0.05
DEFAULT_N = 500
DEFAULT_VOLUME_POINTS = 512
DEFAULT_MAXITER = 200
DEFAULT_PLOT = Path("optimal_JAX/output/optimal_vol_constrainted_norm_p_flux_contours.png")
DEFAULT_PLOT_GRID_SIZE = 600
DEFAULT_CONTOUR_COUNT = 20
DEFAULT_Q = 2.0

# If SciPy tries an invalid shape, return a huge value so it backs away.
BAD_OBJECTIVE_VALUE = 1e100
NORMALIZED_OBJECTIVE = "normalized_pressure"
BETA_T_OBJECTIVE = "beta_toroidal"
BETA_T_ALT_OBJECTIVE = "beta_t_alternative"

# Physical parameter limits used by the optimizer.
PARAMETER_BOUNDS = {
    "epsilon": (0.020001, 0.949),
    "kappa": (0.050001, 12.0),
    "delta": (-0.949, 0.949),
}


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
    if objective_name == BETA_T_ALT_OBJECTIVE:
        return "beta_t_alternative"
    if objective_name == BETA_T_OBJECTIVE:
        return "beta_toroidal"
    return "normalized pressure"


def objective_slug(objective_name):
    """Filename-safe objective name."""
    if objective_name == BETA_T_ALT_OBJECTIVE:
        return "beta_t_alternative"
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
    """Evaluate volume from ordinary NumPy values."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = utils_volume_jax(jnp.asarray(shape, dtype=jnp.float64), int(point_count))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def volume_margin_jax(shape, target_volume, volume_points):
    """Return the remaining volume room; positive means under the cap."""
    return target_volume - utils_volume_jax(shape, point_count=int(volume_points))


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
    if output_path is None:
        output_path = DEFAULT_PLOT
    return plot_contour_from_shape(
        float(epsilon),
        float(kappa),
        float(delta),
        Path(output_path),
        A=float(A),
        grid_size=int(grid_size),
        contour_count=int(contour_count),
        show=show,
    )


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

    if args.beta_t_alt:
        objective_name = BETA_T_ALT_OBJECTIVE
    elif args.beta_t:
        objective_name = BETA_T_OBJECTIVE
    else:
        objective_name = NORMALIZED_OBJECTIVE
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
