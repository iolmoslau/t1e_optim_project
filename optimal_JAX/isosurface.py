#!/usr/bin/env python3
"""Plot the V = V_sep surface colored by the updated beta_t objective."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from scipy.optimize import brentq

try:
    from optimal_JAX import optimal_new_beta_t as beta_opt
except ModuleNotFoundError:
    import optimal_new_beta_t as beta_opt

import jax.numpy as jnp


DEFAULT_OUTPUT = Path("optimal_JAX/output/optimal_new_beta_t_isosurface.png")
DEFAULT_GRID_SIZE = 40
DEFAULT_ROOT_SAMPLES = 40
DEFAULT_N = 500
DEFAULT_VOLUME_POINTS = 128
DEFAULT_ELEVATION = 24.0
DEFAULT_AZIMUTH = -58.0
DEFAULT_EPSILON_RANGE = (0.10, 0.60)
DEFAULT_KAPPA_RANGE = (0.50, 4.50)
DEFAULT_DELTA_RANGE = (-0.70, 0.80)
SINGULAR_TOLERANCE = 1e-9
PARAMETER_LABEL = {
    "epsilon": "epsilon",
    "kappa": "kappa",
    "delta": "delta",
}


def validate_range(name, value_range):
    """Return a finite increasing plotting range."""
    low, high = np.asarray(value_range, dtype=float)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"--{name}-range must be two finite values with LOW < HIGH.")
    return (float(low), float(high))


def shape_has_defined_volume(epsilon, kappa, delta):
    """Return True when the Miller volume formula is defined for plotting."""
    return (
        np.isfinite(epsilon)
        and np.isfinite(kappa)
        and np.isfinite(delta)
        and abs(epsilon) > SINGULAR_TOLERANCE
        and kappa >= 0.0
        and abs(delta) <= 1.0
    )


def shape_has_defined_beta_t(epsilon, kappa, delta):
    """Return True when the updated beta_t formula avoids known singularities."""
    return (
        shape_has_defined_volume(epsilon, kappa, delta)
        and kappa > SINGULAR_TOLERANCE
        and abs(delta) < 1.0 - SINGULAR_TOLERANCE
    )


def volume_from_values(epsilon, kappa, delta, volume_points):
    """Evaluate volume for one ordinary NumPy shape."""
    if not shape_has_defined_volume(epsilon, kappa, delta):
        return np.nan
    try:
        shape = jnp.asarray([epsilon, kappa, delta], dtype=jnp.float64)
        value = float(beta_opt.volume_jax(shape, int(volume_points)))
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def volume_error_for_kappa(kappa, epsilon, delta, target_volume, volume_points):
    """Positive or negative error from V_sep at one kappa value."""
    return volume_from_values(epsilon, kappa, delta, volume_points) - target_volume


def root_candidates_for_shape(
    epsilon,
    delta,
    kappa_range,
    target_volume,
    volume_points,
    root_samples,
):
    """Find candidate kappa roots where V(epsilon, kappa, delta) = V_sep."""
    if (
        not np.isfinite(epsilon)
        or not np.isfinite(delta)
        or abs(epsilon) <= SINGULAR_TOLERANCE
        or abs(delta) > 1.0
        or kappa_range[1] <= 0.0
    ):
        return []

    kappa_values = np.linspace(kappa_range[0], kappa_range[1], int(root_samples))
    errors = np.array(
        [
            volume_error_for_kappa(
                kappa,
                epsilon=epsilon,
                delta=delta,
                target_volume=target_volume,
                volume_points=volume_points,
            )
            for kappa in kappa_values
        ],
        dtype=float,
    )

    roots = []
    for index in range(len(kappa_values) - 1):
        left_kappa = kappa_values[index]
        right_kappa = kappa_values[index + 1]
        left_error = errors[index]
        right_error = errors[index + 1]
        if not np.isfinite(left_error) or not np.isfinite(right_error):
            continue
        if np.isclose(left_error, 0.0, atol=1e-10):
            roots.append(float(left_kappa))
            continue
        if left_error * right_error > 0.0:
            continue
        try:
            roots.append(
                float(
                    brentq(
                        volume_error_for_kappa,
                        left_kappa,
                        right_kappa,
                        args=(epsilon, delta, target_volume, volume_points),
                        xtol=1e-10,
                        rtol=1e-10,
                        maxiter=100,
                    )
                )
            )
        except ValueError:
            continue

    if np.isclose(errors[-1], 0.0, atol=1e-10):
        roots.append(float(kappa_values[-1]))
    return roots


def kappa_on_volume_surface(
    epsilon,
    delta,
    kappa_range,
    target_volume,
    volume_points,
    root_samples,
    preferred_kappa,
):
    """Return the root closest to the separatrix kappa, or NaN if no sheet exists."""
    roots = root_candidates_for_shape(
        epsilon=epsilon,
        delta=delta,
        kappa_range=kappa_range,
        target_volume=target_volume,
        volume_points=volume_points,
        root_samples=root_samples,
    )
    if not roots:
        return np.nan
    return min(roots, key=lambda root: abs(root - preferred_kappa))


def beta_t_from_values(epsilon, kappa, delta, p_0, A, N):
    """Evaluate updated beta_t for one ordinary NumPy shape."""
    if not shape_has_defined_beta_t(epsilon, kappa, delta):
        return np.nan
    try:
        value = beta_opt.beta_t_jax(
            jnp.asarray([epsilon, kappa, delta], dtype=jnp.float64),
            p_0=float(p_0),
            A=float(A),
            N=int(N),
        )
    except Exception:
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


def surface_values(
    epsilon_range,
    kappa_range,
    delta_range,
    grid_size,
    root_samples,
    target_volume,
    volume_points,
    p_0,
    A,
    N,
):
    """Sample the V = V_sep surface and beta_t values on that surface."""
    epsilon_values = np.linspace(*epsilon_range, int(grid_size))
    delta_values = np.linspace(*delta_range, int(grid_size))
    epsilon_grid, delta_grid = np.meshgrid(epsilon_values, delta_values, indexing="xy")
    kappa_grid = np.full_like(epsilon_grid, np.nan, dtype=float)
    beta_t_grid = np.full_like(epsilon_grid, np.nan, dtype=float)

    for row in range(epsilon_grid.shape[0]):
        for column in range(epsilon_grid.shape[1]):
            epsilon = float(epsilon_grid[row, column])
            delta = float(delta_grid[row, column])
            kappa = kappa_on_volume_surface(
                epsilon=epsilon,
                delta=delta,
                kappa_range=kappa_range,
                target_volume=target_volume,
                volume_points=volume_points,
                root_samples=root_samples,
                preferred_kappa=beta_opt.kappa_sep,
            )
            if not np.isfinite(kappa):
                continue
            beta_t = beta_t_from_values(
                epsilon=epsilon,
                kappa=kappa,
                delta=delta,
                p_0=p_0,
                A=A,
                N=N,
            )
            if not np.isfinite(beta_t):
                continue
            kappa_grid[row, column] = kappa
            beta_t_grid[row, column] = beta_t

    return epsilon_grid, kappa_grid, delta_grid, beta_t_grid


def plot_isosurface(
    epsilon_grid,
    kappa_grid,
    delta_grid,
    beta_t_grid,
    output_path,
    elevation,
    azimuth,
    epsilon_range,
    kappa_range,
    delta_range,
    show_sep_point=True,
):
    """Save the colored V = V_sep surface as a Matplotlib PNG file."""
    finite_beta_t = beta_t_grid[np.isfinite(beta_t_grid)]
    if finite_beta_t.size == 0:
        raise ValueError("No finite V = V_sep surface points were found in the selected ranges.")

    finite_surface = (
        np.isfinite(epsilon_grid)
        & np.isfinite(kappa_grid)
        & np.isfinite(delta_grid)
        & np.isfinite(beta_t_grid)
    )
    masked_kappa = np.ma.masked_where(~finite_surface, kappa_grid)
    masked_delta = np.ma.masked_where(~finite_surface, delta_grid)
    masked_epsilon = np.ma.masked_where(~finite_surface, epsilon_grid)
    masked_beta_t = np.ma.masked_where(~finite_surface, beta_t_grid)

    norm = colors.Normalize(vmin=float(np.min(finite_beta_t)), vmax=float(np.max(finite_beta_t)))
    cmap = matplotlib.colormaps["viridis"]

    fig = plt.figure(figsize=(10.0, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        masked_kappa,
        masked_delta,
        masked_epsilon,
        facecolors=cmap(norm(masked_beta_t)),
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )

    if show_sep_point:
        ax.scatter(
            beta_opt.kappa_sep,
            beta_opt.delta_sep,
            beta_opt.epsilon_sep,
            color="red",
            edgecolor="black",
            s=80,
            label="separatrix shape",
        )

    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(finite_beta_t)
    fig.colorbar(scalar_mappable, ax=ax, shrink=0.65, pad=0.08, label="beta_t")

    ax.set_title("V = V_sep surface colored by updated beta_t")
    ax.set_xlabel(PARAMETER_LABEL["kappa"])
    ax.set_ylabel(PARAMETER_LABEL["delta"])
    ax.set_zlabel(PARAMETER_LABEL["epsilon"])
    ax.set_xlim(*kappa_range)
    ax.set_ylim(*delta_range)
    ax.set_zlim(*epsilon_range)
    ax.view_init(elev=float(elevation), azim=float(azimuth))
    if show_sep_point:
        ax.legend(loc="upper left")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the V = V_sep isosurface colored by updated beta_t."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Number of epsilon and delta samples along each surface axis.",
    )
    parser.add_argument(
        "--root-samples",
        type=int,
        default=DEFAULT_ROOT_SAMPLES,
        help="Number of kappa samples used to bracket V = V_sep roots.",
    )
    parser.add_argument(
        "--epsilon-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_EPSILON_RANGE,
        help="Epsilon range sampled for the surface.",
    )
    parser.add_argument(
        "--kappa-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_KAPPA_RANGE,
        help="Kappa range searched for V = V_sep roots.",
    )
    parser.add_argument(
        "--delta-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_DELTA_RANGE,
        help="Delta range sampled for the surface.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=DEFAULT_N,
        help="Grid resolution for beta_t pressure averaging.",
    )
    parser.add_argument(
        "--volume-points",
        type=int,
        default=DEFAULT_VOLUME_POINTS,
        help="Boundary points used for V = V_sep calculations.",
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
        help="Accepted for CLI compatibility; ignored by the updated beta_t objective.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG path for the isosurface plot.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=DEFAULT_ELEVATION,
        help="3D view elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=DEFAULT_AZIMUTH,
        help="3D view azimuth angle.",
    )
    parser.add_argument(
        "--hide-sep-point",
        action="store_true",
        help="Do not mark the separatrix shape on the surface.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")
    if args.root_samples < 2:
        raise ValueError("--root-samples must be at least 2.")
    if args.N < 3:
        raise ValueError("--N must be at least 3.")
    if args.volume_points < 16:
        raise ValueError("--volume-points must be at least 16.")

    epsilon_range = validate_range("epsilon", args.epsilon_range)
    kappa_range = validate_range("kappa", args.kappa_range)
    delta_range = validate_range("delta", args.delta_range)
    target_volume = beta_opt.sep_volume(point_count=args.volume_points)

    epsilon_grid, kappa_grid, delta_grid, beta_t_grid = surface_values(
        epsilon_range=epsilon_range,
        kappa_range=kappa_range,
        delta_range=delta_range,
        grid_size=args.grid_size,
        root_samples=args.root_samples,
        target_volume=target_volume,
        volume_points=args.volume_points,
        p_0=args.p_0,
        A=args.A,
        N=args.N,
    )

    output_path = plot_isosurface(
        epsilon_grid=epsilon_grid,
        kappa_grid=kappa_grid,
        delta_grid=delta_grid,
        beta_t_grid=beta_t_grid,
        output_path=args.output,
        elevation=args.elev,
        azimuth=args.azim,
        epsilon_range=epsilon_range,
        kappa_range=kappa_range,
        delta_range=delta_range,
        show_sep_point=not args.hide_sep_point,
    )

    finite_count = int(np.count_nonzero(np.isfinite(beta_t_grid)))
    finite_beta_t = beta_t_grid[np.isfinite(beta_t_grid)]
    print(f"V_sep: {target_volume:.8g}")
    print(f"surface points: {finite_count}/{beta_t_grid.size}")
    print(f"beta_t min: {np.min(finite_beta_t):.8g}")
    print(f"beta_t max: {np.max(finite_beta_t):.8g}")
    print(f"saved isosurface: {output_path}")


if __name__ == "__main__":
    main()
