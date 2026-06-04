"""Scan A values and plot the best pressure from random starts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from main import multiple_random_runs, _format_shape


DEFAULT_A_VALUES = tuple(-0.05 * index for index in range(11))
DEFAULT_OUTPUT = Path(__file__).with_name("scan_A_highest_pressure.png")


def scan_A_values(
    A_values=DEFAULT_A_VALUES,
    count=10,
    seed=0,
    n_quad=2048,
    maxiter=200,
    gtol=1e-6,
    use_bounds=True,
):
    rows = []
    for A in A_values:
        results = multiple_random_runs(
            count=count,
            seed=seed,
            A=A,
            n_quad=n_quad,
            maxiter=maxiter,
            gtol=gtol,
            use_bounds=use_bounds,
        )
        best = max(results, key=lambda result: float(result["pressure"]))
        rows.append(
            {
                "A": float(A),
                "initial_shape": best["initial_shape"],
                "final_shape": best["shape"],
                "final_pressure": float(best["pressure"]),
            }
        )
    return rows


def plot_highest_pressure(rows, output_path=DEFAULT_OUTPUT):
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(
        [row["A"] for row in rows],
        [row["final_pressure"] for row in rows],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("A")
    ax.set_ylabel("Highest final pressure")
    ax.set_title("Best pressure from 10 random starts")
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def print_summary(rows):
    print(f"{'A':>8}  {'Initial value':<58}  {'Final shape':<58}  {'Final pressure':>16}")
    for row in rows:
        print(
            f"{row['A']:>8.2f}  "
            f"{_format_shape(row['initial_shape']):<58}  "
            f"{_format_shape(row['final_shape']):<58}  "
            f"{row['final_pressure']:>16.12g}"
        )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-quad", type=int, default=2048)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--no-bounds", action="store_true", help="disable configured bounds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    rows = scan_A_values(
        count=args.count,
        seed=args.seed,
        n_quad=args.n_quad,
        maxiter=args.maxiter,
        gtol=args.gtol,
        use_bounds=not args.no_bounds,
    )
    print_summary(rows)
    output_path = plot_highest_pressure(rows, args.output)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
