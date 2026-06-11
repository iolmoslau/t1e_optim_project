#!/usr/bin/env python3
"""
A shape is the triple [epsilon, kappa, delta].  This script asks:

    maximize objective(epsilon, kappa, delta)
    subject to volume(epsilon, kappa, delta) = target_volume
    q_* >= 2
    kappa <= 2.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from optimal_JAX.optimal_vol_constrainted import (
        BAD_OBJECTIVE_VALUE,
        BETA_T_OBJECTIVE,
        DEFAULT_A,
        DEFAULT_CONTOUR_COUNT,
        DEFAULT_MAXITER,
        DEFAULT_N,
        DEFAULT_PLOT_GRID_SIZE,
        DEFAULT_VOLUME_POINTS,
        PARAMETER_BOUNDS as BASE_PARAMETER_BOUNDS,
        PARAMETER_NAMES,
        plot_flux_contours,
        print_shape,
    )
except ModuleNotFoundError:
    from optimal_vol_constrainted import (
        BAD_OBJECTIVE_VALUE,
        BETA_T_OBJECTIVE,
        DEFAULT_A,
        DEFAULT_CONTOUR_COUNT,
        DEFAULT_MAXITER,
        DEFAULT_N,
        DEFAULT_PLOT_GRID_SIZE,
        DEFAULT_VOLUME_POINTS,
        PARAMETER_BOUNDS as BASE_PARAMETER_BOUNDS,
        PARAMETER_NAMES,
        plot_flux_contours,
        print_shape,
    )

from optimal_JAX.utils_JAX import (
    P_0 as UTILS_P_0,
    average_pressure_updated_jax as utils_average_pressure_updated_jax,
    beta_poloidal_updated_jax as utils_beta_poloidal_updated_jax,
    beta_toroidal_updated_jax as utils_beta_toroidal_updated_jax,
    q_star_updated_jax as utils_q_star_updated_jax,
    volume_jax as utils_volume_jax,
)

volume_jax = utils_volume_jax


import jax
import jax.numpy as jnp


DEFAULT_P_0 = UTILS_P_0
DEFAULT_PLOT = Path("optimal_JAX/output/optimal_new_beta_t_flux_contours.png")
SLSQP_OBJECTIVE_SCALE = 1
B_0 = 12.2  # T
I = 8.7e6  # A
R_0 = 1.85  # m
epsilon_sep = 0.31
kappa_sep = 1.97
delta_sep = 0.54
MU_0 = 4.0 * np.pi * 1e-7

MIN_Q_STAR = 2.0
MAX_KAPPA = 2.1
SEP_SHAPE = np.array([epsilon_sep, kappa_sep, delta_sep], dtype=float)
MIN_EPSILON = 0.1
MAX_EPSILON = 0.90
MIN_DELTA = -0.70
MAX_DELTA = 0.80

PARAMETER_BOUNDS = {
    "epsilon": (MIN_EPSILON, MAX_EPSILON),
    "kappa": (BASE_PARAMETER_BOUNDS["kappa"][0], MAX_KAPPA),
    "delta": (MIN_DELTA, MAX_DELTA),
}


def average_pressure_jax(shape, p_0, A=DEFAULT_A, N=DEFAULT_N):
    """Contour volume average of Psi / Psi_min inside the plasma."""
    return utils_average_pressure_updated_jax(shape, p_0=p_0, A=A, N=N)


def beta_p_jax(shape, average_pressure):
    """Poloidal beta from the provided volume-averaged pressure."""
    return utils_beta_poloidal_updated_jax(shape, average_pressure, R_0=R_0, I=I)


def q_star_jax(shape):
    """Cylindrical safety factor q_* from the requested scaling."""
    return utils_q_star_updated_jax(shape, R_0=R_0, I=I, B_0=B_0)


def beta_t_jax(shape, p_0=DEFAULT_P_0, A=DEFAULT_A, N=DEFAULT_N):
    """Toroidal beta matching pressure_utils.beta_toroidal_updated."""
    return utils_beta_toroidal_updated_jax(
        shape,
        p_0=p_0,
        A=A,
        N=N,
        R_0=R_0,
        I=I,
        B_0=B_0,
    )


beta_t_new_jax = beta_t_jax


def shape_is_valid(shape):
    """Return True when the shape is finite and inside the local bounds."""
    epsilon, kappa, delta = np.asarray(shape, dtype=float)
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and PARAMETER_BOUNDS["epsilon"][0] <= epsilon <= PARAMETER_BOUNDS["epsilon"][1]
        and PARAMETER_BOUNDS["kappa"][0] <= kappa <= PARAMETER_BOUNDS["kappa"][1]
        and PARAMETER_BOUNDS["delta"][0] <= delta <= PARAMETER_BOUNDS["delta"][1]
    )


def clip_shape_to_bounds(shape):
    """Clamp SciPy boundary roundoff back inside the declared parameter bounds."""
    clipped = np.asarray(shape, dtype=float).copy()
    for index, name in enumerate(PARAMETER_NAMES):
        low, high = PARAMETER_BOUNDS[name]
        clipped[index] = np.clip(clipped[index], low, high)
    return clipped


def beta_t_from_shape(shape, p_0=DEFAULT_P_0, A=DEFAULT_A, N=DEFAULT_N):
    """Evaluate the new toroidal beta from ordinary NumPy values."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = beta_t_jax(
            jnp.asarray(shape, dtype=jnp.float64),
            p_0=float(p_0),
            A=float(A),
            N=int(N),
        )
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def volume_from_shape(shape, point_count=DEFAULT_VOLUME_POINTS):
    """Evaluate volume from ordinary NumPy values using the updated bounds."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = utils_volume_jax(jnp.asarray(shape, dtype=jnp.float64), int(point_count))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def q_star_from_shape(shape):
    """Evaluate q_* from ordinary NumPy values."""
    shape = np.asarray(shape, dtype=float)
    if not shape_is_valid(shape):
        return np.nan
    try:
        value = q_star_jax(jnp.asarray(shape, dtype=jnp.float64))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(value)


def sep_volume(point_count=DEFAULT_VOLUME_POINTS):
    """Volume fixed by epsilon_sep, kappa_sep, and delta_sep."""
    return volume_from_shape(SEP_SHAPE, point_count=point_count)


def volume_margin_jax(shape, target_volume, volume_points):
    """Zero when the current shape volume equals V_sep."""
    return target_volume - utils_volume_jax(shape, point_count=int(volume_points))


def q_star_margin_jax(shape):
    """Positive when q_* satisfies the lower bound."""
    return q_star_jax(shape) - MIN_Q_STAR


def optimize_shape(
    start_shape=None,
    target_volume=None,
    p_0=DEFAULT_P_0,
    A=DEFAULT_A,
    N=DEFAULT_N,
    maxiter=DEFAULT_MAXITER,
    volume_points=DEFAULT_VOLUME_POINTS,
):
    """Maximize beta_t with fixed V_sep, q_* >= 2, and kappa <= 2.1."""
    if start_shape is None:
        start_shape = SEP_SHAPE
    if target_volume is None:
        target_volume = sep_volume(point_count=volume_points)

    start_shape = np.asarray(start_shape, dtype=float)
    path = [start_shape.copy()]

    negative_objective_with_gradient = jax.value_and_grad(
        lambda shape: -SLSQP_OBJECTIVE_SCALE
        * beta_t_jax(
            shape,
            p_0=float(p_0),
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
    q_star_room_with_gradient = jax.value_and_grad(q_star_margin_jax)

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

    def q_star_room(shape):
        if not shape_is_valid(shape):
            return -BAD_OBJECTIVE_VALUE
        try:
            room, _ = q_star_room_with_gradient(jnp.asarray(shape, dtype=jnp.float64))
            return float(room)
        except Exception:
            return -BAD_OBJECTIVE_VALUE

    def q_star_room_gradient(shape):
        try:
            _, gradient = q_star_room_with_gradient(jnp.asarray(shape, dtype=jnp.float64))
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.zeros(3, dtype=float)
        if not np.all(np.isfinite(gradient)):
            return np.zeros(3, dtype=float)
        return gradient

    def save_optimizer_step(shape):
        path.append(clip_shape_to_bounds(shape))

    result = minimize(
        negative_objective_and_gradient,
        start_shape,
        method="SLSQP",
        jac=True,
        bounds=[PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES],
        constraints=[
            {
                "type": "eq",
                "fun": volume_room,
                "jac": volume_room_gradient,
            },
            {
                "type": "ineq",
                "fun": q_star_room,
                "jac": q_star_room_gradient,
            },
        ],
        callback=save_optimizer_step,
        options={"ftol": 1e-8, "maxiter": int(maxiter)},
    )

    final_shape = clip_shape_to_bounds(result.x)
    result.x = final_shape
    if not np.allclose(path[-1], final_shape):
        path.append(final_shape.copy())

    final_volume = volume_from_shape(final_shape, point_count=volume_points)
    final_q_star = q_star_from_shape(final_shape)

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "initial_shape": start_shape,
        "initial_beta_t": beta_t_from_shape(start_shape, p_0=p_0, A=A, N=N),
        "initial_volume": volume_from_shape(start_shape, point_count=volume_points),
        "initial_q_star": q_star_from_shape(start_shape),
        "target_volume": target_volume,
        "final_shape": final_shape,
        "final_beta_t": beta_t_from_shape(final_shape, p_0=p_0, A=A, N=N),
        "final_volume": final_volume,
        "final_volume_margin": target_volume - final_volume,
        "final_q_star": final_q_star,
        "final_q_star_margin": final_q_star - MIN_Q_STAR,
    }


def print_summary(run):
    """Print a compact optimization summary."""
    result = run["result"]
    print()
    print("objective: beta_t")
    print(f"optimizer success: {bool(result.success)}")
    print(f"optimizer message: {result.message}")
    print(f"iterations: {result.nit}")
    print_shape("separatrix shape", SEP_SHAPE)
    print_shape("starting shape", run["initial_shape"])
    print_shape("final shape", run["final_shape"])
    print(f"starting beta_t: {run['initial_beta_t']:.8g}")
    print(f"final beta_t: {run['final_beta_t']:.8g}")
    print(f"V_sep: {run['target_volume']:.8g}")
    print(f"starting volume: {run['initial_volume']:.8g}")
    print(f"final volume: {run['final_volume']:.8g}")
    print(f"final volume margin: {run['final_volume_margin']:.8g}")
    print(f"starting q_*: {run['initial_q_star']:.8g}")
    print(f"final q_*: {run['final_q_star']:.8g}")
    print(f"final q_* margin: {run['final_q_star_margin']:.8g}")
    print(f"final kappa <= {MAX_KAPPA}: {run['final_shape'][1] <= MAX_KAPPA + 1e-7}")
    print(f"q_* constraint satisfied: {run['final_q_star'] >= MIN_Q_STAR - 1e-7}")
    print(f"volume constraint satisfied: {abs(run['final_volume_margin']) <= 1e-7}")
    print(f"path length: {len(run['path'])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maximize beta_t with fixed V_sep, q_* >= 2, and kappa <= 2.1."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Number of grid points per direction for pressure averaging.",
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
        help="A parameter used by the local flux calculation.",
    )
    parser.add_argument(
        "--p-0",
        type=float,
        default=DEFAULT_P_0,
        help="Pressure scale used by the updated beta_t objective.",
    )
    parser.add_argument(
        "--start-epsilon",
        type=float,
        default=SEP_SHAPE[0],
        help="Starting epsilon.",
    )
    parser.add_argument(
        "--start-kappa",
        type=float,
        default=SEP_SHAPE[1],
        help="Starting kappa.",
    )
    parser.add_argument(
        "--start-delta",
        type=float,
        default=SEP_SHAPE[2],
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

    target_volume = sep_volume(point_count=args.volume_points)
    run = optimize_shape(
        start_shape=start_shape,
        target_volume=target_volume,
        p_0=args.p_0,
        A=args.A,
        N=args.N,
        maxiter=args.maxiter,
        volume_points=args.volume_points,
    )
    print_summary(run)
    plot_flux_contours(
        run["final_shape"],
        objective_name=BETA_T_OBJECTIVE,
        A=args.A,
        output_path=args.plot,
        grid_size=args.plot_grid_size,
        contour_count=args.contour_count,
    )
    print(f"saved optimized flux contours: {args.plot}")


if __name__ == "__main__":
    main()
