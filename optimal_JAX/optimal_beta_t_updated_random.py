#!/usr/bin/env python3
"""Run repeated updated beta_t optimizations from random initial guesses."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = Path("optimal_JAX/output/optimal_beta_t")
DEFAULT_OBJECTIVE_HISTORY_NAME = "optimal_beta_t_updated_random_objective_history.png"
DEFAULT_BEST_CONTOUR_NAME = "optimal_beta_t_updated_random_best_contour.png"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((REPO_ROOT / DEFAULT_OUTPUT_DIR / ".matplotlib").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from optimal_JAX import optimal_beta_t_updated as beta_t_optimizer
except ModuleNotFoundError:
    import optimal_beta_t_updated as beta_t_optimizer


def objective_label() -> str:
    """Readable objective name for printed output and plots."""
    return "updated beta_t"


def default_best_contour_path() -> Path:
    """Return the default contour PNG path for the best random-start result."""
    return DEFAULT_OUTPUT_DIR / DEFAULT_BEST_CONTOUR_NAME


def default_objective_history_path() -> Path:
    """Return the default objective-history PNG path for random-start results."""
    return DEFAULT_OUTPUT_DIR / DEFAULT_OBJECTIVE_HISTORY_NAME


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


def format_objective(value):
    """Format objective values for compact output."""
    return f"{float(value):.8g}"


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


def print_best_summary(runs) -> None:
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
    print(f"  {objective_label()}: {format_objective(best['final_objective'])}")
    print(f"  optimizer success: {bool(result.success)}")
    print(f"  optimizer message: {result.message}")


def shape_ranges_from_args(args):
    """Build and check the random initial-guess ranges."""
    ranges = {
        "epsilon": (args.epsilon_min, args.epsilon_max),
        "kappa": (args.kappa_min, args.kappa_max),
        "delta": (args.delta_min, args.delta_max),
    }

    for name, (low, high) in ranges.items():
        bound_low, bound_high = beta_t_optimizer.PARAMETER_BOUNDS[name]
        if low is None:
            low = bound_low
        if high is None:
            high = bound_high
        if low < bound_low or high > bound_high or low >= high:
            raise ValueError(
                f"{name} range must be inside "
                f"{beta_t_optimizer.PARAMETER_BOUNDS[name]} with min < max."
            )
        ranges[name] = (float(low), float(high))
    return ranges


def plot_objective_history(runs, output_path: Path) -> Path:
    """Save a PNG showing objective value by optimizer step for the best run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run = best_run(runs)
    if run is None:
        raise RuntimeError("No finite final objective found; no history was saved.")

    path = np.asarray(run["path"], dtype=float)
    objective_values = np.array(
        [
            beta_t_optimizer.beta_t_from_shape(
                shape,
                p_0=run["p_0"],
                A=run["A"],
                N=run["N"],
            )
            for shape in path
        ],
        dtype=float,
    )
    steps = np.arange(len(objective_values), dtype=int)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(steps, objective_values, marker="o", linewidth=1.5)
    ax.set_xlabel("optimization step")
    ax.set_ylabel(objective_label())
    ax.set_title(f"{objective_label()} history for best run {run['run_number']}")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    fig.savefig(output_path, dpi=200, format="png")
    plt.close(fig)
    return output_path


def save_best_contour(
    run,
    output_path: Path,
    A: float,
    plot_grid_size: int,
    contour_count: int,
) -> Path:
    """Save a contour plot for the best final shape."""
    return beta_t_optimizer.plot_flux_contours(
        run["final_shape"],
        objective_name=beta_t_optimizer.BETA_T_OBJECTIVE,
        A=A,
        output_path=output_path,
        grid_size=plot_grid_size,
        contour_count=contour_count,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated updated beta_t optimizations from random initial guesses."
    )
    parser.add_argument(
        "--runs",
        "-x",
        type=int,
        default=5,
        help="Number of random initial guesses to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible initial guesses.",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=beta_t_optimizer.DEFAULT_A,
        help="A parameter passed to the objective.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=beta_t_optimizer.DEFAULT_N,
        help="Grid resolution passed to the objective.",
    )
    parser.add_argument(
        "--p-0",
        type=float,
        default=beta_t_optimizer.DEFAULT_P_0,
        help="Pressure scale used by the updated beta_t objective.",
    )
    parser.add_argument(
        "--volume-points",
        type=int,
        default=beta_t_optimizer.DEFAULT_VOLUME_POINTS,
        help="Number of boundary points used for the volume calculation.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=beta_t_optimizer.DEFAULT_MAXITER,
        help="Maximum SLSQP iterations for each run.",
    )
    parser.add_argument("--epsilon-min", type=float, default=None)
    parser.add_argument("--epsilon-max", type=float, default=None)
    parser.add_argument("--kappa-min", type=float, default=None)
    parser.add_argument("--kappa-max", type=float, default=None)
    parser.add_argument("--delta-min", type=float, default=None)
    parser.add_argument("--delta-max", type=float, default=None)
    parser.add_argument(
        "--plot-grid-size",
        type=int,
        default=beta_t_optimizer.DEFAULT_PLOT_GRID_SIZE,
        help="Grid size for the best final-shape contour plot.",
    )
    parser.add_argument(
        "--contour-count",
        type=int,
        default=beta_t_optimizer.DEFAULT_CONTOUR_COUNT,
        help="Number of best final-shape flux contour levels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.N < 3:
        raise ValueError("--N must be at least 3.")
    if args.volume_points < 16:
        raise ValueError("--volume-points must be at least 16.")
    if args.plot_grid_size < 3:
        raise ValueError("--plot-grid-size must be at least 3.")
    if args.contour_count < 1:
        raise ValueError("--contour-count must be at least 1.")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parameter_ranges = shape_ranges_from_args(args)
    rng = np.random.default_rng(args.seed)
    target_volume = beta_t_optimizer.sep_volume(point_count=args.volume_points)

    print(f"objective: {objective_label()}")
    print(f"runs: {args.runs}")
    print(f"random epsilon range: {parameter_ranges['epsilon']}")
    print(f"random kappa range: {parameter_ranges['kappa']}")
    print(f"random delta range: {parameter_ranges['delta']}")
    print("run_number, initial_shape, final_shape, final_objective, success")

    runs = []
    for run_number in range(1, args.runs + 1):
        initial_shape = random_shape(rng, parameter_ranges)
        run = beta_t_optimizer.optimize_shape(
            start_shape=initial_shape,
            target_volume=target_volume,
            p_0=args.p_0,
            A=args.A,
            N=args.N,
            maxiter=args.maxiter,
            volume_points=args.volume_points,
        )
        run = dict(run)
        run["run_number"] = run_number
        run["final_objective"] = run["final_beta_t"]
        run["p_0"] = args.p_0
        run["A"] = args.A
        run["N"] = args.N
        runs.append(run)
        print_run_summary(run_number, run)

    print_best_summary(runs)

    objective_history_path = default_objective_history_path()
    saved_history_path = plot_objective_history(runs, objective_history_path)
    print(f"saved objective history plot: {saved_history_path}")

    best = best_run(runs)
    if best is None:
        raise RuntimeError("No finite final objective found; no best contour was saved.")

    best_contour_path = default_best_contour_path()
    saved_contour_path = save_best_contour(
        best,
        best_contour_path,
        A=args.A,
        plot_grid_size=args.plot_grid_size,
        contour_count=args.contour_count,
    )
    print(f"saved best final shape contour plot: {saved_contour_path}")


if __name__ == "__main__":
    main()
