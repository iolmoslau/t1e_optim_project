#!/usr/bin/env python3
"""Run repeated unconstrained optimizations from random initial guesses."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimal_JAX.optimal_unconstrainted import (
    DEFAULT_A,
    DEFAULT_FINITE_DIFF_STEP,
    DEFAULT_FTOL,
    DEFAULT_GTOL,
    DEFAULT_MAXITER,
    DEFAULT_METHOD,
    DEFAULT_N,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSITIVE_FLOOR,
    DEFAULT_Q,
    OBJECTIVE_BETA_TOROIDAL,
    OBJECTIVE_CHOICES,
    OBJECTIVE_NORMALIZED_PRESSURE,
    OBJECTIVE_SLUGS,
    OBJECTIVE_VOLUME_PRESSURE,
    finite_float,
    format_objective,
    int_at_least,
    optimize_shape,
    positive_float,
)
from optimal_JAX.plot_contour_from_shape import plot_contour_from_shape


DEFAULT_RANDOM_RANGES = {
    "epsilon": (0.020001, 0.949),
    "kappa": (0.050001, 12.0),
    "delta": (-0.949, 0.949),
}


def objective_label(objective: str) -> str:
    """Readable objective name for printed output and plots."""
    if objective == OBJECTIVE_VOLUME_PRESSURE:
        return "volume pressure"
    if objective == OBJECTIVE_NORMALIZED_PRESSURE:
        return "normalized psi pressure"
    if objective == OBJECTIVE_BETA_TOROIDAL:
        return "beta toroidal"
    return objective


def default_best_contour_path(objective: str) -> Path:
    """Return the default contour PNG path for the best random-start result."""
    return (
        DEFAULT_OUTPUT_DIR
        / f"optimal_unconstrained_random_{OBJECTIVE_SLUGS[objective]}_best_contour.png"
    )


def default_objective_history_path(objective: str) -> Path:
    """Return the default objective-history PNG path for random-start results."""
    return (
        DEFAULT_OUTPUT_DIR
        / f"optimal_unconstrained_random_{OBJECTIVE_SLUGS[objective]}_objective_history.png"
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
    epsilon, kappa, delta = (float(value) for value in shape)
    return f"({epsilon:.8g}, {kappa:.8g}, {delta:.8g})"


def print_run_summary(run_number: int, run) -> None:
    """Print one random-start result."""
    result = run["result"]
    print(
        f"{run_number}, "
        f"initial_shape={format_shape(run['initial_shape'])}, "
        f"final_shape={format_shape(run['final_shape'])}, "
        f"final_objective={format_objective(run['final_objective'])}, "
        f"success={bool(result.success)}"
    )


def best_run(runs):
    """Return the run with the highest finite final objective."""
    finite_runs = [run for run in runs if np.isfinite(run["final_objective"])]
    if not finite_runs:
        return None
    return max(finite_runs, key=lambda run: run["final_objective"])


def print_best_summary(runs, objective: str) -> None:
    """Print the best result across all random-start runs."""
    best = best_run(runs)
    if best is None:
        print("\nbest run: none with finite objective")
        return

    result = best["result"]
    print("\nbest run")
    print(f"  run number: {best['run_number']}")
    print(f"  initial shape parameters: {format_shape(best['initial_shape'])}")
    print(f"  final shape parameters: {format_shape(best['final_shape'])}")
    print(f"  {objective_label(objective)}: {format_objective(best['final_objective'])}")
    print(f"  optimizer success: {bool(result.success)}")
    print(f"  optimizer message: {result.message}")


def shape_ranges_from_args(args):
    """Build and check the random initial-guess ranges."""
    ranges = {
        "epsilon": (args.epsilon_min, args.epsilon_max),
        "kappa": (args.kappa_min, args.kappa_max),
        "delta": (args.delta_min, args.delta_max),
    }
    bounds = dict(DEFAULT_RANDOM_RANGES)
    bounds["epsilon"] = (
        max(bounds["epsilon"][0], args.epsilon_floor),
        bounds["epsilon"][1],
    )
    bounds["kappa"] = (
        max(bounds["kappa"][0], args.kappa_floor),
        bounds["kappa"][1],
    )

    for name, (low, high) in ranges.items():
        bound_low, bound_high = bounds[name]
        if low is None:
            low = bound_low
        if high is None:
            high = bound_high
        if low < bound_low or high > bound_high or low >= high:
            raise ValueError(
                f"{name} range must be inside {bounds[name]} with min < max."
            )
        ranges[name] = (float(low), float(high))
    return ranges


def plot_objective_history(runs, objective: str, output_path: Path) -> Path:
    """Save a PNG showing final objective value by random-start run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_numbers = np.array([run["run_number"] for run in runs], dtype=int)
    objective_values = np.array(
        [float(run["final_objective"]) for run in runs],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(run_numbers, objective_values, marker="o", linewidth=1.5)
    ax.set_xlabel("run iteration")
    ax.set_ylabel(objective_label(objective))
    ax.set_title(f"{objective_label(objective)} by random-start run")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(run_numbers)

    finite_mask = np.isfinite(objective_values)
    if finite_mask.any():
        best_index = int(np.nanargmax(objective_values))
        ax.scatter(
            [run_numbers[best_index]],
            [objective_values[best_index]],
            color="tab:red",
            zorder=3,
            label="best",
        )
        ax.legend()

    fig.savefig(output_path, dpi=200, format="png")
    plt.close(fig)
    return output_path


def save_best_contour(run, output_path: Path, A: float) -> Path:
    """Save a contour plot for the best final shape."""
    epsilon, kappa, delta = (float(value) for value in run["final_shape"])
    return plot_contour_from_shape(epsilon, kappa, delta, output_path, A=A)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated unconstrained optimizations from random initial guesses."
    )
    parser.add_argument(
        "--runs",
        "-x",
        type=int_at_least("runs", 1),
        default=5,
        help="Number of random initial guesses to run.",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_CHOICES,
        default=OBJECTIVE_VOLUME_PRESSURE,
        help="Objective function to maximize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible initial guesses.",
    )
    parser.add_argument(
        "--A",
        type=finite_float("A"),
        default=DEFAULT_A,
        help="Solov'ev profile parameter.",
    )
    parser.add_argument(
        "--N",
        type=int_at_least("N", 2),
        default=DEFAULT_N,
        help="Grid or boundary resolution passed to the selected objective.",
    )
    parser.add_argument(
        "--method",
        choices=("contour", "masking"),
        default=DEFAULT_METHOD,
        help="Objective integration method for pressure objectives.",
    )
    parser.add_argument(
        "--q",
        type=positive_float("q"),
        default=DEFAULT_Q,
        help="Safety factor used by beta_toroidal.",
    )
    parser.add_argument(
        "--maxiter",
        type=int_at_least("maxiter", 0),
        default=DEFAULT_MAXITER,
        help="Maximum L-BFGS-B iterations for each run.",
    )
    parser.add_argument(
        "--ftol",
        type=positive_float("ftol"),
        default=DEFAULT_FTOL,
        help="Relative function tolerance for L-BFGS-B.",
    )
    parser.add_argument(
        "--gtol",
        type=positive_float("gtol"),
        default=DEFAULT_GTOL,
        help="Projected-gradient tolerance for L-BFGS-B.",
    )
    parser.add_argument(
        "--finite-diff-step",
        type=positive_float("finite-diff step"),
        default=DEFAULT_FINITE_DIFF_STEP,
        help="Absolute finite-difference step used by the JAX custom JVP.",
    )
    parser.add_argument(
        "--epsilon-floor",
        type=positive_float("epsilon floor"),
        default=DEFAULT_POSITIVE_FLOOR,
        help="Positive lower bound for epsilon.",
    )
    parser.add_argument(
        "--kappa-floor",
        type=positive_float("kappa floor"),
        default=DEFAULT_POSITIVE_FLOOR,
        help="Positive lower bound for kappa.",
    )
    parser.add_argument("--epsilon-min", type=finite_float("epsilon min"), default=None)
    parser.add_argument("--epsilon-max", type=finite_float("epsilon max"), default=None)
    parser.add_argument("--kappa-min", type=finite_float("kappa min"), default=None)
    parser.add_argument("--kappa-max", type=finite_float("kappa max"), default=None)
    parser.add_argument("--delta-min", type=finite_float("delta min"), default=None)
    parser.add_argument("--delta-max", type=finite_float("delta max"), default=None)
    parser.add_argument(
        "--best-contour-plot",
        type=Path,
        default=None,
        help=(
            "PNG path for the best final-shape contour plot. Defaults to an "
            "objective-specific name in optimal_JAX/output."
        ),
    )
    parser.add_argument(
        "--objective-history-plot",
        type=Path,
        default=None,
        help=(
            "PNG path for the objective value vs run iteration plot. Defaults "
            "to an objective-specific name in optimal_JAX/output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameter_ranges = shape_ranges_from_args(args)
    rng = np.random.default_rng(args.seed)

    print(f"objective: {args.objective}")
    print(f"runs: {args.runs}")
    print(f"random epsilon range: {parameter_ranges['epsilon']}")
    print(f"random kappa range: {parameter_ranges['kappa']}")
    print(f"random delta range: {parameter_ranges['delta']}")
    print("run_number, initial_shape, final_shape, final_objective, success")

    runs = []
    for run_number in range(1, args.runs + 1):
        initial_shape = random_shape(rng, parameter_ranges)
        run = optimize_shape(
            initial_shape=initial_shape,
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
        run = dict(run)
        run["run_number"] = run_number
        runs.append(run)
        print_run_summary(run_number, run)

    print_best_summary(runs, args.objective)

    objective_history_path = (
        args.objective_history_plot
        if args.objective_history_plot is not None
        else default_objective_history_path(args.objective)
    )
    saved_history_path = plot_objective_history(
        runs,
        args.objective,
        objective_history_path,
    )
    print(f"saved objective history plot: {saved_history_path}")

    best = best_run(runs)
    if best is None:
        raise RuntimeError("No finite final objective found; no best contour was saved.")

    best_contour_path = (
        args.best_contour_plot
        if args.best_contour_plot is not None
        else default_best_contour_path(args.objective)
    )
    saved_contour_path = save_best_contour(best, best_contour_path, A=args.A)
    print(f"saved best final shape contour plot: {saved_contour_path}")


if __name__ == "__main__":
    main()
