"""JAX/BFGS optimizer for the symmetric Cerfon-Freidberg Solov'ev shape."""

from __future__ import annotations

import argparse
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize


DEFAULT_A = -0.05
DEFAULT_BOUNDS = ((0.05, 0.95), (0.5, 3.0), (-0.75, 0.75)) # epsilon,kappa,delta
DEFAULT_INITIAL_SHAPE = (0.32, 1.7, 0.33) # epsilon,kappa,delta

def basis_values(x, y):
    logx = jnp.log(x)
    return jnp.stack(
        [
            jnp.ones_like(x + y),
            x**2,
            y**2 - x**2 * logx,
            x**4 - 4 * x**2 * y**2,
            2 * y**4 - 9 * y**2 * x**2 + 3 * x**4 * logx - 12 * x**2 * y**2 * logx,
            x**6 - 12 * x**4 * y**2 + 8 * x**2 * y**4,
            (
                8 * y**6
                - 140 * y**4 * x**2
                + 75 * y**2 * x**4
                - 15 * x**6 * logx
                + 180 * x**4 * y**2 * logx
                - 120 * x**2 * y**4 * logx
            ),
        ]
    )


def basis_x(x, y):
    logx = jnp.log(x)
    return jnp.stack(
        [
            jnp.zeros_like(x + y),
            2 * x,
            -2 * x * logx - x,
            4 * x**3 - 8 * x * y**2,
            -30 * x * y**2 + 12 * x**3 * logx + 3 * x**3 - 24 * x * logx * y**2,
            6 * x**5 - 48 * x**3 * y**2 + 16 * x * y**4,
            (
                -400 * x * y**4
                + 480 * x**3 * y**2
                - 90 * x**5 * logx
                + 720 * y**2 * x**3 * logx
                - 15 * x**5
                - 240 * y**4 * x * logx
            ),
        ]
    )


def basis_y(x, y):
    logx = jnp.log(x)
    return jnp.stack(
        [
            jnp.zeros_like(x + y),
            jnp.zeros_like(x + y),
            2 * y,
            -8 * x**2 * y,
            8 * y**3 - 18 * y * x**2 - 24 * x**2 * y * logx,
            -24 * x**4 * y + 32 * x**2 * y**3,
            48 * y**5 - 560 * x**2 * y**3 - 480 * x**2 * logx * y**3 + 360 * x**4 * logx * y + 150 * x**4 * y,
        ]
    )


def basis_xx(x, y):
    logx = jnp.log(x)
    return jnp.stack(
        [
            jnp.zeros_like(x + y),
            jnp.ones_like(x + y) * 2,
            -2 * logx - 3,
            12 * x**2 - 8 * y**2,
            -54 * y**2 + 36 * x**2 * logx + 21 * x**2 - 24 * logx * y**2,
            30 * x**4 - 144 * x**2 * y**2 + 16 * y**4,
            -640 * y**4 + 2160 * x**2 * y**2 - 450 * x**4 * logx - 165 * x**4 + 2160 * y**2 * x**2 * logx - 240 * y**4 * logx,
        ]
    )


def basis_yy(x, y):
    logx = jnp.log(x)
    return jnp.stack(
        [
            jnp.zeros_like(x + y),
            jnp.zeros_like(x + y),
            jnp.ones_like(x + y) * 2,
            -8 * x**2,
            24 * y**2 - 18 * x**2 - 24 * x**2 * logx,
            -24 * x**4 + 96 * x**2 * y**2,
            240 * y**4 - 1680 * x**2 * y**2 - 1440 * x**2 * logx * y**2 + 360 * x**4 * logx + 150 * x**4,
        ]
    )


def particular_value(x, A=DEFAULT_A):
    return A * 0.5 * x**2 * jnp.log(x) + (1 - A) * x**4 / 8


def particular_x(x, A=DEFAULT_A):
    return A * (x * jnp.log(x) + x / 2) + (1 - A) * x**3 / 2


def particular_xx(x, A=DEFAULT_A):
    return A * (jnp.log(x) + 1.5) + (1 - A) * 1.5 * x**2


def solve_coefficients(epsilon, kappa, delta, A=DEFAULT_A):
    """Solve the 7x7 symmetric Solov'ev boundary system for c_1..c_7."""
    alpha = jnp.arcsin(delta)
    curv1 = -(1 + alpha) ** 2 / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * jnp.cos(alpha) ** 2)
    curv3 = (1 - alpha) ** 2 / (epsilon * kappa**2)

    x_out = 1 + epsilon
    x_in = 1 - epsilon
    y_mid = jnp.asarray(0.0, dtype=jnp.result_type(epsilon, kappa, delta, A))
    x_hi = 1 - epsilon * delta
    y_hi = kappa * epsilon

    rows = [
        basis_values(x_out, y_mid),
        basis_values(x_in, y_mid),
        basis_values(x_hi, y_hi),
        basis_x(x_hi, y_hi),
        curv1 * basis_x(x_out, y_mid) + basis_yy(x_out, y_mid),
        curv3 * basis_x(x_in, y_mid) + basis_yy(x_in, y_mid),
        curv2 * basis_y(x_hi, y_hi) + basis_xx(x_hi, y_hi),
    ]
    matrix = jnp.stack(rows)

    rhs = -jnp.stack(
        [
            particular_value(x_out, A),
            particular_value(x_in, A),
            particular_value(x_hi, A),
            particular_x(x_hi, A),
            curv1 * particular_x(x_out, A),
            curv3 * particular_x(x_in, A),
            particular_xx(x_hi, A),
        ]
    )
    return jnp.linalg.solve(matrix, rhs)


def psi_value(x, y, coefficients, A=DEFAULT_A):
    """Evaluate the full symmetric Solov'ev flux function."""
    return jnp.einsum("i,i...->...", coefficients, basis_values(x, y)) + particular_value(x, A)


def g_basis_values(x, y):
    logx = jnp.log(x)
    return jnp.stack(
        [
            x * y,
            x**3 * y,
            x * y * (-x**2 * logx + y**2 / 3),
            x**3 * y * (x**2 - 4 * y**2 / 3),
            x * y * (15 * x**4 * logx + x**2 * y**2 * (-20 * logx - 15) + 2 * y**4) / 5,
            x**3 * y * (x**4 - 4 * x**2 * y**2 + 8 * y**4 / 5),
            x
            * y
            * (
                -105 * x**6 * logx
                + x**4 * y**2 * (420 * logx + 175)
                + x**2 * y**4 * (-168 * logx - 196)
                + 8 * y**6
            )
            / 7,
        ]
    )


def g_total(x, y, A, coefficients):
    """Exact y-antiderivative of x * psi(x, y)."""
    g_base = x**5 * y / 8
    g_A = x**3 * y * (-x**2 + 4 * jnp.log(x)) / 8
    return g_base + A * g_A + jnp.einsum("i,i...->...", coefficients, g_basis_values(x, y))


def boundary_samples(epsilon, kappa, delta, n_quad):
    theta = (jnp.arange(n_quad) + 0.5) * (2 * jnp.pi / n_quad)
    alpha = jnp.arcsin(delta)
    phase = theta + alpha * jnp.sin(theta)
    x = 1 + epsilon * jnp.cos(phase)
    y = epsilon * kappa * jnp.sin(theta)
    dx_dtheta = -epsilon * jnp.sin(phase) * (1 + alpha * jnp.cos(theta))
    return x, y, dx_dtheta, 2 * jnp.pi / n_quad


def green_integral(g_values, dx_dtheta, dtheta):
    return -jnp.sum(g_values * dx_dtheta) * dtheta


def volume_averaged_pressure(epsilon, kappa, delta, A=DEFAULT_A, n_quad=2048):
    """Return -(1-A) * int x*psi dA / int x dA using analytic-boundary Green integrals."""
    coefficients = solve_coefficients(epsilon, kappa, delta, A)
    x, y, dx_dtheta, dtheta = boundary_samples(epsilon, kappa, delta, n_quad)
    numerator = green_integral(g_total(x, y, A, coefficients), dx_dtheta, dtheta)
    denominator = green_integral(x * y, dx_dtheta, dtheta)
    return -(1 - A) * numerator / denominator


def bounded_from_unconstrained(raw, bounds=DEFAULT_BOUNDS):
    bounds = jnp.asarray(bounds)
    low = bounds[:, 0]
    high = bounds[:, 1]
    return low + (high - low) * jax.nn.sigmoid(raw)


def unconstrained_from_bounded(shape, bounds=DEFAULT_BOUNDS):
    shape = jnp.asarray(shape)
    bounds = jnp.asarray(bounds)
    low = bounds[:, 0]
    high = bounds[:, 1]
    ratio = jnp.clip((shape - low) / (high - low), 1e-12, 1 - 1e-12)
    return jnp.log(ratio) - jnp.log1p(-ratio)


def pressure_from_raw(raw, A=DEFAULT_A, bounds=DEFAULT_BOUNDS, n_quad=2048):
    epsilon, kappa, delta = bounded_from_unconstrained(raw, bounds)
    return volume_averaged_pressure(epsilon, kappa, delta, A=A, n_quad=n_quad)


def negative_pressure_from_raw(raw, A=DEFAULT_A, bounds=DEFAULT_BOUNDS, n_quad=2048):
    return -pressure_from_raw(raw, A=A, bounds=bounds, n_quad=n_quad)


def _validate_bounds(bounds):
    parsed = tuple((float(lo), float(hi)) for lo, hi in bounds)
    for lo, hi in parsed:
        if not lo < hi:
            raise ValueError("each lower bound must be less than its upper bound")
    if parsed[0][0] <= 0 or parsed[0][1] >= 1:
        raise ValueError("epsilon bounds must stay inside (0, 1)")
    if parsed[1][0] <= 0:
        raise ValueError("kappa lower bound must be positive")
    if parsed[2][0] <= -1 or parsed[2][1] >= 1:
        raise ValueError("delta bounds must stay inside (-1, 1)")
    return parsed


def _validate_initial_shape(initial_shape, bounds):
    shape = tuple(float(v) for v in initial_shape)
    for value, (lo, hi), name in zip(shape, bounds, ("epsilon", "kappa", "delta")):
        if not lo < value < hi:
            raise ValueError(f"initial {name} must be inside ({lo}, {hi})")
    return shape


def optimize_shape(
    initial_shape=DEFAULT_INITIAL_SHAPE,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    n_quad=2048,
    maxiter=200,
    gtol=1e-6,
):
    """Maximize volume-averaged pressure over epsilon, kappa, delta."""
    bounds = _validate_bounds(bounds)
    initial_shape = _validate_initial_shape(initial_shape, bounds)
    raw0 = unconstrained_from_bounded(jnp.asarray(initial_shape), bounds)

    def objective(raw):
        return negative_pressure_from_raw(raw, A=A, bounds=bounds, n_quad=n_quad)

    options = {"maxiter": int(maxiter)} if maxiter is not None else None
    result = minimize(objective, raw0, method="BFGS", tol=gtol, options=options)
    final_raw = result.x
    final_shape = bounded_from_unconstrained(final_raw, bounds)
    final_grad = jax.grad(objective)(final_raw)

    return {
        "initial_shape": jnp.asarray(initial_shape),
        "initial_pressure": volume_averaged_pressure(*initial_shape, A=A, n_quad=n_quad),
        "raw_initial": raw0,
        "shape": final_shape,
        "pressure": volume_averaged_pressure(*final_shape, A=A, n_quad=n_quad),
        "coefficients": solve_coefficients(*final_shape, A=A),
        "raw": final_raw,
        "gradient": final_grad,
        "gradient_norm": jnp.linalg.norm(final_grad),
        "optimizer": result,
    }


def run_unit_tests(
    result,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    n_quad=2048,
    gradient_tol=1e-5,
    hessian_tol=1e-8,
    perturbation_tol=1e-9,
    perturbation_radius=1e-3,
    perturbation_samples=100,
    perturbation_seed=0,
    active_bound_tol=1e-4,
):
    """Run local optimality checks around the BFGS result."""
    import numpy as np

    bounds = _validate_bounds(bounds)
    bounds_array = jnp.asarray(bounds)
    low = bounds_array[:, 0]
    high = bounds_array[:, 1]
    names = ("epsilon", "kappa", "delta")

    raw = result["raw"]
    shape = result["shape"]
    pressure = result["pressure"]
    pressure_objective = lambda z: pressure_from_raw(z, A=A, bounds=bounds, n_quad=n_quad)

    gradient = jax.grad(pressure_objective)(raw)
    gradient_norm = float(jnp.linalg.norm(gradient))

    hessian = jax.hessian(pressure_objective)(raw)
    all_hessian_eigenvalues = np.linalg.eigvalsh(np.asarray(hessian))

    lower_active = shape <= low + active_bound_tol
    upper_active = shape >= high - active_bound_tol
    free_indices = [
        i for i, is_free in enumerate(np.asarray(~(lower_active | upper_active))) if is_free
    ]
    if free_indices:
        reduced_hessian = np.asarray(hessian)[np.ix_(free_indices, free_indices)]
        reduced_eigenvalues = np.linalg.eigvalsh(reduced_hessian)
        hessian_pass = bool(np.max(reduced_eigenvalues) < -hessian_tol)
    else:
        reduced_eigenvalues = np.array([])
        hessian_pass = True

    key = jax.random.PRNGKey(int(perturbation_seed))
    directions = jax.random.normal(key, (int(perturbation_samples), 3), dtype=shape.dtype)
    directions = jnp.where(upper_active, -jnp.abs(directions), directions)
    directions = jnp.where(lower_active, jnp.abs(directions), directions)
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    perturbations = float(perturbation_radius) * directions
    points = jnp.clip(shape + perturbations, low, high)
    perturbed_pressures = jnp.asarray(
        [volume_averaged_pressure(*point, A=A, n_quad=n_quad) for point in points]
    )
    max_perturbed_pressure = float(jnp.max(perturbed_pressures))

    checks = {
        "gradient": gradient_norm <= gradient_tol,
        "hessian": hessian_pass,
        "perturbation": max_perturbed_pressure <= float(pressure) + perturbation_tol,
    }

    free_names = [names[i] for i in free_indices] or ["none"]
    print("Unit tests:")
    print(f"  Gradient norm: {gradient_norm:.6e} <= {gradient_tol:.6e}")
    print(f"  Full Hessian eigenvalues: {all_hessian_eigenvalues}")
    print(f"  Free Hessian variables: {', '.join(free_names)}")
    print(f"  Free Hessian eigenvalues: {reduced_eigenvalues}")
    print(f"  Max perturbed pressure: {max_perturbed_pressure:.12g}")
    print(f"  Optimum pressure: {float(pressure):.12g}")

    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"failed unit checks: {', '.join(failures)}")
    print("  All unit checks passed.")

    return {
        "gradient_norm": gradient_norm,
        "full_hessian_eigenvalues": all_hessian_eigenvalues,
        "free_hessian_eigenvalues": reduced_eigenvalues,
        "max_perturbed_pressure": max_perturbed_pressure,
        "checks": checks,
    }


def plot_flux_contours(
    shape,
    coefficients=None,
    A=DEFAULT_A,
    output_path=None,
    grid_size=600,
    contour_count=20,
    show=False,
):
    """Plot Solov'ev flux contours with the paper-style jet colormap."""
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon, kappa, delta = [float(v) for v in shape]
    if coefficients is None:
        coefficients = solve_coefficients(epsilon, kappa, delta, A=A)

    x = np.linspace(1 - epsilon - 0.05, 1 + epsilon + 0.1, int(grid_size))
    y = np.linspace(-kappa * epsilon - 0.05, kappa * epsilon + 0.025, int(grid_size))
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(psi_value(jnp.asarray(X), jnp.asarray(Y), coefficients, A=A))

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
    ax.set_xlim(0, 1 + epsilon + 0.25)
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


def _format_shape(shape):
    epsilon, kappa, delta = [float(v) for v in shape]
    return f"epsilon={epsilon:.12g}, kappa={kappa:.12g}, delta={delta:.12g}"


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--initial", nargs=3, type=float, default=DEFAULT_INITIAL_SHAPE, metavar=("EPS", "KAP", "DLT"))
    parser.add_argument("--epsilon-bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[0], metavar=("LOW", "HIGH"))
    parser.add_argument("--kappa-bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[1], metavar=("LOW", "HIGH"))
    parser.add_argument("--delta-bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[2], metavar=("LOW", "HIGH"))
    parser.add_argument("--n-quad", type=int, default=2048)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--plot", type=Path, help="save optimized flux contours to this path")
    parser.add_argument("--plot-grid", type=int, default=600)
    parser.add_argument("--contour-count", type=int, default=20)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--test-perturbation-radius", type=float, default=1e-3)
    parser.add_argument("--test-perturbation-samples", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=0)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    bounds = (tuple(args.epsilon_bounds), tuple(args.kappa_bounds), tuple(args.delta_bounds))
    result = optimize_shape(
        initial_shape=tuple(args.initial),
        A=args.A,
        bounds=bounds,
        n_quad=args.n_quad,
        maxiter=args.maxiter,
        gtol=args.gtol,
    )
    optimizer = result["optimizer"]

    print(f"Initial shape: {_format_shape(result['initial_shape'])}")
    print(f"Initial pressure: {float(result['initial_pressure']):.12g}")
    print(f"Final shape: {_format_shape(result['shape'])}")
    print(f"Final pressure: {float(result['pressure']):.12g}")
    print(f"Gradient norm: {float(result['gradient_norm']):.12g}")
    if hasattr(optimizer, "status"):
        print(f"Optimizer status: {int(optimizer.status)}")
    if hasattr(optimizer, "success"):
        print(f"Optimizer success: {bool(optimizer.success)}")
    if args.run_tests:
        run_unit_tests(
            result,
            A=args.A,
            bounds=bounds,
            n_quad=args.n_quad,
            perturbation_radius=args.test_perturbation_radius,
            perturbation_samples=args.test_perturbation_samples,
            perturbation_seed=args.test_seed,
        )
    if args.plot or args.show_plot:
        plot_flux_contours(
            result["shape"],
            coefficients=result["coefficients"],
            A=args.A,
            output_path=args.plot,
            grid_size=args.plot_grid,
            contour_count=args.contour_count,
            show=args.show_plot,
        )
        if args.plot:
            print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
