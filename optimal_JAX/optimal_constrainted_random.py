#!/usr/bin/env python3
"""Run repeated constrained optimizations from random initial guesses."""

from __future__ import annotations

import argparse

import numpy as np

from optimal_constrainted_norm_p import (
    BETA_T_OBJECTIVE,
    DEFAULT_A,
    DEFAULT_MAXITER,
    DEFAULT_N,
    DEFAULT_VOLUME_POINTS,
    NORMALIZED_OBJECTIVE,
    PARAMETER_BOUNDS,
    TARGET_SHAPE,
    objective_label,
    optimize_shape,
    volume_from_shape,
)


def random_shape(rng, parameter_ranges):
    """Draw one random initial guess inside the selected parameter ranges."""
    return np.array(
        [
            rng.uniform(*parameter_ranges["epsilon"]),
            rng.uniform(*parameter_ranges["kappa"]),
            rng.uniform(*parameter_ranges["delta"]),
        ],
        dtype=float,
    )


def format_shape(shape):
    """Format epsilon, kappa, delta for compact output."""
    epsilon, kappa, delta = shape
    return f"({epsilon:.8g}, {kappa:.8g}, {delta:.8g})"


def print_run_summary(run_number, run, objective_name):
    """Print one run in the requested final-shape, pressure form."""
    label = objective_label(objective_name)
    result = run["result"]
    print(
        f"{run_number}, "
        f"final_shape={format_shape(run['final_shape'])}, "
        f"{label}={run['final_objective']:.8g}, "
        f"success={bool(result.success)}"
    )


def print_best_summary(runs, objective_name):
    """Print the best result across all runs."""
    label = objective_label(objective_name)
    finite_runs = [
        run for run in runs
        if np.isfinite(run["final_objective"])
    ]
    if not finite_runs:
        print("\nbest run: none with finite objective")
        return

    best = max(finite_runs, key=lambda run: run["final_objective"])
    print("\nbest run")
    print(f"  final shape parameters: {format_shape(best['final_shape'])}")
    print(f"  {label}: {best['final_objective']:.8g}")
    print(f"  final volume: {best['final_volume']:.8g}")
    print(f"  final volume margin: {best['final_volume_margin']:.8g}")
    print(f"  optimizer success: {bool(best['result'].success)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated constrained optimizations from random initial guesses."
    )
    parser.add_argument(
        "--runs",
        "-x",
        type=int,
        default=5,
        help="Number of random initial guesses to run.",
    )
    parser.add_argument(
        "--beta_t",
        action="store_true",
        help="Maximize beta_toroidal instead of normalized pressure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible initial guesses.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Grid resolution passed to the objective.",
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
        help="Maximum SLSQP iterations for each run.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=DEFAULT_A,
        help="A parameter passed to the objective.",
    )
    parser.add_argument("--epsilon-min", type=float, default=PARAMETER_BOUNDS["epsilon"][0])
    parser.add_argument("--epsilon-max", type=float, default=PARAMETER_BOUNDS["epsilon"][1])
    parser.add_argument("--kappa-min", type=float, default=PARAMETER_BOUNDS["kappa"][0])
    parser.add_argument("--kappa-max", type=float, default=PARAMETER_BOUNDS["kappa"][1])
    parser.add_argument("--delta-min", type=float, default=PARAMETER_BOUNDS["delta"][0])
    parser.add_argument("--delta-max", type=float, default=PARAMETER_BOUNDS["delta"][1])
    return parser.parse_args()


def shape_ranges_from_args(args):
    """Build and check the random initial-guess ranges."""
    ranges = {
        "epsilon": (args.epsilon_min, args.epsilon_max),
        "kappa": (args.kappa_min, args.kappa_max),
        "delta": (args.delta_min, args.delta_max),
    }
    for name, (low, high) in ranges.items():
        bound_low, bound_high = PARAMETER_BOUNDS[name]
        if low < bound_low or high > bound_high or low >= high:
            raise ValueError(
                f"{name} range must be inside {PARAMETER_BOUNDS[name]} with min < max."
            )
    return ranges


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.N < 3:
        raise ValueError("--N must be at least 3.")
    if args.volume_points < 16:
        raise ValueError("--volume-points must be at least 16.")

    objective_name = BETA_T_OBJECTIVE if args.beta_t else NORMALIZED_OBJECTIVE
    parameter_ranges = shape_ranges_from_args(args)
    rng = np.random.default_rng(args.seed)
    target_volume = volume_from_shape(TARGET_SHAPE, point_count=args.volume_points)

    print(f"objective: {objective_label(objective_name)}")
    print(f"runs: {args.runs}")
    print(f"target volume: {target_volume:.8g}")
    print("run_number, final_shape_parameters, selected_objective, success")

    runs = []
    for run_number in range(1, args.runs + 1):
        start_shape = random_shape(rng, parameter_ranges)
        run = optimize_shape(
            start_shape=start_shape,
            target_volume=target_volume,
            objective_name=objective_name,
            A=args.A,
            N=args.N,
            maxiter=args.maxiter,
            volume_points=args.volume_points,
        )
        runs.append(run)
        print_run_summary(run_number, run, objective_name)

    print_best_summary(runs, objective_name)


if __name__ == "__main__":
    main()
