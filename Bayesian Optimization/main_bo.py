"""BO development copy of optimal_JAX/main.py.

This file preserves the main program structure and provides a safe place to
add Bayesian optimization logic without modifying the original main.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import numpy as np

from jax import config

config.update("jax_enable_x64", True)

# Match the legacy import layout used by pressure_integral/test_beta.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (
    PROJECT_ROOT,
    PROJECT_ROOT / "ITER_Equilibria",
    PROJECT_ROOT / "pressure_integral",
):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import jax
import jax.numpy as jnp
from sklearn.utils._tags import default_tags
from skopt import dummy_minimize, forest_minimize, gbrt_minimize, gp_minimize
from skopt.learning.gbrt import GradientBoostingQuantileRegressor
from skopt.space import Real
from pressure_integral.pressure_utils import (
    beta_t_alternative,
    beta_toroidal,
    get_vol_av_p_from_params,
    normalized_psi_pressure,
    plot_plasma_profile,
)

# Fix scikit-optimize GBRT compatibility with newer scikit-learn tag handling.
GradientBoostingQuantileRegressor.__sklearn_tags__ = lambda self: default_tags(self)


DEFAULT_A = -0.05
DEFAULT_OBJECTIVE = "beta_toroidal_fixed_q"
DEFAULT_Q = 2.0
DEFAULT_BOUNDS = ((0.1, 0.5), (0.5, 4.0), (-0.5, 0.5))  # epsilon,kappa,delta
#DEFAULT_BOUNDS = ((0.1, 0.45), (1, 1.7), (-0.3, 0.3))  # epsilon,kappa,delta
PARAM_NAMES = ("epsilon", "kappa", "delta")
PARAM_LABELS = {
    "epsilon": r"$\epsilon$",
    "kappa": r"$\kappa$",
    "delta": r"$\delta$",
}
OBJECTIVE_LABELS = {
    "volume_average_pressure": "volume_average_pressure",
    "normalized_psi_pressure": "normalized_psi_pressure",
    "beta_toroidal_fixed_q": "beta_toroidal_fixed_q",
    "beta_toroidal_variable_q": "beta_toroidal_variable_q",
}
LANDSCAPE_CMAP = "plasma"
ERROR_CMAP = "cividis"
CALL_CMAP = "plasma"
FIRST_CALL_COLOR = "#E69F00"
FINAL_BEST_COLOR = "#0072B2"
MARKER_EDGE_COLOR = "#000000"
REFERENCE_LINE_COLOR = "#000000"
DELTA_SCAN_COLORS = (
    "#0072B2",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
)


def _normalize_objective_name(objective_name):
    objective_name = str(objective_name).lower()
    aliases = {
        "pressure": "volume_average_pressure",
        "volume_averaged_pressure": "volume_average_pressure",
        "vol_av_pressure": "volume_average_pressure",
        "psi_normalized_pressure": "normalized_psi_pressure",
        "psi_normalised_pressure": "normalized_psi_pressure",
        "beta_toroidal": "beta_toroidal_fixed_q",
        "beta_t_fixed_q": "beta_toroidal_fixed_q",
        "beta_t_variable_q": "beta_toroidal_variable_q",
        "beta_t_alternative": "beta_toroidal_variable_q",
    }
    objective_name = aliases.get(objective_name, objective_name)
    if objective_name not in OBJECTIVE_LABELS:
        choices = ", ".join(OBJECTIVE_LABELS)
        raise ValueError(f"unknown objective {objective_name!r}; choose one of: {choices}")
    return objective_name


def objective_label(objective_name):
    return OBJECTIVE_LABELS[_normalize_objective_name(objective_name)]


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


def objective_np(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
):
    """Evaluate the selected pressure/beta objective on a shape vector.

    This wrapper is intended for Bayesian optimization.
    """
    objective_name = _normalize_objective_name(objective_name)
    epsilon, kappa, delta = shape
    with np.errstate(divide="ignore", invalid="ignore"):
        if objective_name == "volume_average_pressure":
            return float(get_vol_av_p_from_params(
                epsilon, kappa, delta, A=A, method="contour", N=int(n_quad)
            ))
        if objective_name == "normalized_psi_pressure":
            return float(normalized_psi_pressure(
                epsilon, kappa, delta, A=A, method="contour", N=int(n_quad)
            ))
        if objective_name == "beta_toroidal_fixed_q":
            return float(beta_toroidal(
                epsilon, kappa, delta, A=A, q=float(q), N=int(n_quad)
            ))
        if objective_name == "beta_toroidal_variable_q":
            return float(beta_t_alternative(
                epsilon, kappa, delta, A=A, N=int(n_quad)
            ))
    raise ValueError(f"unsupported objective {objective_name!r}")


def negative_objective_np(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
):
    """Negated objective for minimizers expecting lower-is-better."""
    return -objective_np(
        shape,
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        q=q,
    )


def _build_bo_space(bounds, free_indices=None):
    """Build scikit-optimize search space from bounds.

    `bounds` should be an iterable of (low, high) pairs for epsilon,kappa,delta.
    """
    if free_indices is None:
        free_indices = range(len(PARAM_NAMES))
    return [
        Real(bounds[i][0], bounds[i][1], name=PARAM_NAMES[i])
        for i in free_indices
    ]


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


def _normalize_fixed_values(fixed_values=None, bounds=DEFAULT_BOUNDS):
    bounds = _validate_bounds(bounds)
    if fixed_values is None:
        fixed_values = (None, None, None)
    elif isinstance(fixed_values, dict):
        fixed_values = tuple(fixed_values.get(name) for name in PARAM_NAMES)
    else:
        fixed_values = tuple(fixed_values)

    if len(fixed_values) != len(PARAM_NAMES):
        raise ValueError("fixed_values must contain epsilon, kappa, and delta")

    normalized = []
    for value, (lo, hi), name in zip(fixed_values, bounds, PARAM_NAMES):
        if value is None:
            normalized.append(None)
            continue
        value = float(value)
        if value < lo or value > hi:
            raise ValueError(f"fixed {name}={value:g} is outside bounds [{lo:g}, {hi:g}]")
        normalized.append(value)
    return tuple(normalized)


def _fixed_and_free_indices(fixed_values):
    fixed_indices = tuple(i for i, value in enumerate(fixed_values) if value is not None)
    free_indices = tuple(i for i, value in enumerate(fixed_values) if value is None)
    return fixed_indices, free_indices


def _shape_from_free_values(free_values, fixed_values):
    free_values = tuple(float(value) for value in free_values)
    _, free_indices = _fixed_and_free_indices(fixed_values)
    if len(free_values) != len(free_indices):
        raise ValueError("candidate dimension does not match the free parameter count")

    shape = [None, None, None]
    for i, value in enumerate(fixed_values):
        if value is not None:
            shape[i] = float(value)
    for i, value in zip(free_indices, free_values):
        shape[i] = float(value)
    return tuple(shape)


def _format_fixed_values(fixed_values):
    fixed_items = [
        f"{PARAM_NAMES[i]}={value:.12g}"
        for i, value in enumerate(fixed_values)
        if value is not None
    ]
    return ", ".join(fixed_items) if fixed_items else "none"


def optimize_shape_bayesian(
    bounds=DEFAULT_BOUNDS,
    A=DEFAULT_A,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
    fixed_values=None,
    n_calls=50,
    n_initial_points=10,
    random_state=0,
    n_quad=2048,
    surrogate="gp",
):
    objective_name = _normalize_objective_name(objective_name)
    bounds = _validate_bounds(bounds)
    fixed_values = _normalize_fixed_values(fixed_values, bounds)
    fixed_indices, free_indices = _fixed_and_free_indices(fixed_values)
    if len(fixed_indices) > 1:
        raise ValueError("fix at most one of epsilon, kappa, or delta for this 2D contour workflow")
    if not free_indices:
        raise ValueError("at least one parameter must be free for Bayesian optimization")
    if int(n_initial_points) >= int(n_calls):
        raise ValueError("n_initial_points must be smaller than n_calls")

    # Build search space and run BO using the requested surrogate model.
    # The objective is negated because the optimizer expects lower-is-better.
    search_space = _build_bo_space(bounds, free_indices)
    objective = lambda x: negative_objective_np(
        _shape_from_free_values(x, fixed_values),
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        q=q,
    )
    surrogate = str(surrogate).lower()

    if surrogate == "gp":
        bo_result = gp_minimize(
            func=objective,
            dimensions=search_space,
            acq_func="EI",
            n_calls=int(n_calls),
            n_initial_points=int(n_initial_points),
            random_state=int(random_state),
            verbose=False,
        )
    elif surrogate == "forest":
        bo_result = forest_minimize(
            func=objective,
            dimensions=search_space,
            acq_func="EI",
            n_calls=int(n_calls),
            n_initial_points=int(n_initial_points),
            random_state=int(random_state),
            verbose=False,
        )
    elif surrogate == "gbrt":
        bo_result = gbrt_minimize(
            func=objective,
            dimensions=search_space,
            acq_func="EI",
            n_calls=int(n_calls),
            n_initial_points=int(n_initial_points),
            random_state=int(random_state),
            verbose=False,
        )
    elif surrogate == "dummy":
        bo_result = dummy_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=int(n_calls),
            random_state=int(random_state),
            verbose=False,
        )
    else:
        raise ValueError(
            "unsupported surrogate model: %r. supported models are 'gp', 'forest', 'gbrt', 'dummy'"
            % surrogate
        )

    best_shape = _shape_from_free_values(bo_result.x, fixed_values)
    best_value = objective_np(
        best_shape,
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        q=q,
    )
    best_volume_average_pressure = objective_np(
        best_shape,
        A=A,
        n_quad=n_quad,
        objective_name="volume_average_pressure",
        q=q,
    )
    coefficients = solve_coefficients(*best_shape, A=A)
    raw = unconstrained_from_bounded(jnp.asarray(best_shape), bounds)
    shape_iters = [
        _shape_from_free_values(candidate, fixed_values)
        for candidate in bo_result.x_iters
    ]
    best_call = int(np.argmin(np.asarray(bo_result.func_vals))) + 1

    if shape_iters:
        first_shape = shape_iters[0]
        first_value = objective_np(
            first_shape,
            A=A,
            n_quad=n_quad,
            objective_name=objective_name,
            q=q,
        )
    else:
        first_shape = best_shape
        first_value = best_value

    return {
        "initial_shape": jnp.asarray(first_shape),
        "initial_objective_value": first_value,
        "raw_initial": unconstrained_from_bounded(jnp.asarray(first_shape), bounds),
        "shape": jnp.asarray(best_shape),
        "objective_value": best_value,
        "final_volume_average_pressure": best_volume_average_pressure,
        "objective_name": objective_name,
        "objective_label": objective_label(objective_name),
        "q": float(q),
        "coefficients": coefficients,
        "raw": raw,
        "optimizer": bo_result,
        "calls": int(n_calls),
        "initial_points": int(n_initial_points),
        "best_raw": bo_result.x,
        "best_fun": bo_result.fun,
        "best_call": best_call,
        "x_iters": bo_result.x_iters,
        "shape_iters": shape_iters,
        "func_vals": bo_result.func_vals,
        "fixed_values": fixed_values,
        "fixed_indices": fixed_indices,
        "free_indices": free_indices,
    }


def run_unit_tests(
    result,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    n_quad=2048,
    objective_name=None,
    q=None,
    perturbation_tol=1e-9,
    perturbation_radius=1e-3,
    perturbation_samples=100,
    perturbation_seed=0,
    active_bound_tol=1e-4,
):
    """Run local perturbation checks around the optimizer result."""
    import numpy as np

    bounds = _validate_bounds(bounds)
    bounds_array = np.asarray(bounds, dtype=float)
    low = bounds_array[:, 0]
    high = bounds_array[:, 1]
    names = ("epsilon", "kappa", "delta")

    shape = np.asarray(result["shape"], dtype=float)
    objective_name = _normalize_objective_name(objective_name or result.get("objective_name", DEFAULT_OBJECTIVE))
    q = float(result.get("q", DEFAULT_Q) if q is None else q)
    objective_value = float(result["objective_value"])
    label = objective_label(objective_name)
    fixed_values = result.get("fixed_values", (None, None, None))
    fixed_mask = np.asarray([value is not None for value in fixed_values], dtype=bool)

    lower_active = shape <= low + active_bound_tol
    upper_active = shape >= high - active_bound_tol
    free_indices = [
        i for i, is_free in enumerate(~(lower_active | upper_active | fixed_mask)) if is_free
    ]

    rng = np.random.default_rng(int(perturbation_seed))
    directions = rng.normal(size=(int(perturbation_samples), 3))
    directions[:, fixed_mask] = 0.0
    directions = np.where(upper_active, -np.abs(directions), directions)
    directions = np.where(lower_active, np.abs(directions), directions)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    perturbations = float(perturbation_radius) * directions
    points = np.clip(shape + perturbations, low, high)
    perturbed_values = np.asarray(
        [
            objective_np(
                point,
                A=A,
                n_quad=n_quad,
                objective_name=objective_name,
                q=q,
            )
            for point in points
        ]
    )
    max_perturbed_value = float(np.max(perturbed_values))

    checks = {
        "perturbation": max_perturbed_value <= objective_value + perturbation_tol,
    }

    free_names = [names[i] for i in free_indices] or ["none"]
    print("Unit tests:")
    print(f"  Free variables: {', '.join(free_names)}")
    print(f"  Max perturbed {label}: {max_perturbed_value:.12g}")
    print(f"  Optimum {label}: {objective_value:.12g}")

    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"failed unit checks: {', '.join(failures)}")
    print("  All unit checks passed.")

    return {
        "max_perturbed_objective": max_perturbed_value,
        "checks": checks,
    }


def finite_difference_objective_gradient(
    shape,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
    step=1e-3,
):
    """Estimate objective partial derivatives with bounded finite differences."""
    bounds = _validate_bounds(bounds)
    bounds_array = np.asarray(bounds, dtype=float)
    low = bounds_array[:, 0]
    high = bounds_array[:, 1]
    shape = np.asarray(shape, dtype=float)
    names = ("epsilon", "kappa", "delta")

    if shape.shape != (3,):
        raise ValueError("shape must contain epsilon, kappa, and delta")
    if np.any(shape < low - 1e-12) or np.any(shape > high + 1e-12):
        raise ValueError("shape must be inside the optimization bounds")

    objective_name = _normalize_objective_name(objective_name)
    shape = np.clip(shape, low, high)
    requested_step = float(step)
    if requested_step <= 0:
        raise ValueError("finite-difference step must be positive")

    cache = {}

    def value_at(point):
        key = tuple(float(v) for v in point)
        if key not in cache:
            cache[key] = objective_np(
                key,
                A=A,
                n_quad=n_quad,
                objective_name=objective_name,
                q=q,
            )
        return cache[key]

    gradients = np.zeros(3, dtype=float)
    schemes = []
    steps = []

    for i in range(3):
        lo, hi = low[i], high[i]
        width = hi - lo
        h = min(requested_step, 0.25 * width)
        x0 = shape.copy()

        if x0[i] - h >= lo and x0[i] + h <= hi:
            xm = x0.copy()
            xp = x0.copy()
            xm[i] -= h
            xp[i] += h
            gradients[i] = (value_at(xp) - value_at(xm)) / (2 * h)
            schemes.append("central")
        elif x0[i] + 2 * h <= hi:
            x1 = x0.copy()
            x2 = x0.copy()
            x1[i] += h
            x2[i] += 2 * h
            gradients[i] = (-3 * value_at(x0) + 4 * value_at(x1) - value_at(x2)) / (2 * h)
            schemes.append("forward")
        elif x0[i] - 2 * h >= lo:
            x1 = x0.copy()
            x2 = x0.copy()
            x1[i] -= h
            x2[i] -= 2 * h
            gradients[i] = (3 * value_at(x0) - 4 * value_at(x1) + value_at(x2)) / (2 * h)
            schemes.append("backward")
        else:
            raise ValueError(f"could not build a finite-difference stencil for {names[i]}")
        steps.append(h)

    return {
        "shape": shape,
        "value": value_at(shape),
        "names": names,
        "gradient": gradients,
        "schemes": tuple(schemes),
        "steps": tuple(steps),
    }


def print_final_gradient_diagnostics(
    result,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    n_quad=2048,
    objective_name=None,
    q=None,
    step=1e-3,
):
    objective_name = _normalize_objective_name(objective_name or result.get("objective_name", DEFAULT_OBJECTIVE))
    q = float(result.get("q", DEFAULT_Q) if q is None else q)
    diagnostics = finite_difference_objective_gradient(
        result["shape"],
        A=A,
        bounds=bounds,
        n_quad=n_quad,
        objective_name=objective_name,
        q=q,
        step=step,
    )
    label = objective_label(objective_name)
    print("Numerical gradient at final shape:")
    print(f"  Objective: {label} = {diagnostics['value']:.12g}")
    for name, value, scheme, used_step in zip(
        diagnostics["names"],
        diagnostics["gradient"],
        diagnostics["schemes"],
        diagnostics["steps"],
    ):
        print(f"  d/d{name}: {value:.12g}  ({scheme}, h={used_step:.3g})")
    return diagnostics


def plot_flux_contours(
    shape,
    coefficients=None,
    A=DEFAULT_A,
    output_path=None,
    grid_size=600,
    contour_count=20,
    show=False,
):
    """Plot plasma contours using the shared pressure_utils implementation."""
    import matplotlib.pyplot as plt

    epsilon, kappa, delta = [float(v) for v in shape]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_plasma_profile(
        epsilon,
        kappa,
        delta,
        A=A,
        N=int(grid_size),
        n_levels=int(contour_count),
        colorbar=True,
        title=True,
        ylabel=True,
        ax=ax,
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


def _trajectory_sample_indices(count, every):
    if count <= 0:
        return []
    every = max(1, int(every))
    indices = [0]
    indices.extend(range(every - 1, count, every))
    if count - 1 not in indices:
        indices.append(count - 1)
    return sorted(set(indices))


def plot_fixed_parameter_objective_contours(
    result,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    fixed_values=None,
    output_path=None,
    error_output_path=None,
    grid_size=35,
    contour_count=20,
    trajectory_every=5,
    n_quad=500,
    objective_name=None,
    q=None,
    show=False,
):
    """Plot true objective, surrogate prediction, and surrogate error contours."""
    import matplotlib.pyplot as plt
    import numpy as np

    bounds = _validate_bounds(bounds)
    fixed_values = result.get("fixed_values", fixed_values)
    fixed_values = _normalize_fixed_values(fixed_values, bounds)
    fixed_indices, free_indices = _fixed_and_free_indices(fixed_values)

    if len(fixed_indices) != 1 or len(free_indices) != 2:
        raise ValueError("objective contour plots require exactly one fixed parameter")

    objective_name = _normalize_objective_name(objective_name or result.get("objective_name", DEFAULT_OBJECTIVE))
    q = float(result.get("q", DEFAULT_Q) if q is None else q)
    label = objective_label(objective_name)

    grid_size = int(grid_size)
    contour_count = int(contour_count)
    trajectory_every = int(trajectory_every)
    if grid_size < 2:
        raise ValueError("objective contour grid size must be at least 2")
    if contour_count < 2:
        raise ValueError("objective contour count must be at least 2")
    if trajectory_every < 1:
        raise ValueError("trajectory_every must be positive")

    fixed_index = fixed_indices[0]
    x_index, y_index = free_indices
    fixed_name = PARAM_NAMES[fixed_index]
    x_name = PARAM_NAMES[x_index]
    y_name = PARAM_NAMES[y_index]
    fixed_value = fixed_values[fixed_index]

    x_values = np.linspace(bounds[x_index][0], bounds[x_index][1], grid_size)
    y_values = np.linspace(bounds[y_index][0], bounds[y_index][1], grid_size)
    X, Y = np.meshgrid(x_values, y_values)
    Z_true = np.full_like(X, np.nan, dtype=float)

    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            shape = [None, None, None]
            shape[fixed_index] = fixed_value
            shape[x_index] = float(x_value)
            shape[y_index] = float(y_value)
            try:
                Z_true[row, col] = objective_np(
                    tuple(shape),
                    A=A,
                    n_quad=n_quad,
                    objective_name=objective_name,
                    q=q,
                )
            except Exception:
                Z_true[row, col] = np.nan

    optimizer = result.get("optimizer")
    models = getattr(optimizer, "models", None)
    model = models[-1] if models else None
    Z_surrogate = np.full_like(Z_true, np.nan, dtype=float)

    if model is not None:
        grid_free = np.column_stack([X.ravel(), Y.ravel()])
        try:
            model_points = optimizer.space.transform(grid_free.tolist())
        except Exception:
            model_points = grid_free
        try:
            predicted_minimizer_values = model.predict(model_points)
            Z_surrogate = -np.asarray(predicted_minimizer_values, dtype=float).reshape(X.shape)
        except Exception:
            model = None

    absolute_error = np.abs(Z_surrogate - Z_true)
    relative_denominator = np.abs(Z_true)
    valid_error = (
        np.isfinite(Z_true)
        & np.isfinite(Z_surrogate)
        & (relative_denominator > 1e-12)
    )
    Z_relative_error = np.full_like(Z_true, np.nan, dtype=float)
    Z_relative_error[valid_error] = absolute_error[valid_error] / relative_denominator[valid_error]
    if np.any(valid_error):
        error_values = Z_relative_error[valid_error]
        surrogate_error = {
            "mean_relative": float(np.mean(error_values)),
            "rmse": float(np.sqrt(np.mean(error_values**2))),
            "max_relative": float(np.max(error_values)),
        }
    else:
        surrogate_error = None

    shape_iters = np.asarray(result.get("shape_iters", []), dtype=float)
    final_shape = np.asarray(result["shape"], dtype=float)
    best_call = result.get("best_call")

    def add_evaluation_points(axes):
        if not shape_iters.size:
            return None
        axes = np.atleast_1d(axes)
        calls = np.arange(1, len(shape_iters) + 1)
        scatter = None
        for axis in axes:
            scatter = axis.scatter(
                shape_iters[:, x_index],
                shape_iters[:, y_index],
                c=calls,
                cmap=CALL_CMAP,
                edgecolor=MARKER_EDGE_COLOR,
                linewidth=0.35,
                s=34,
                label="BO evaluations",
                zorder=4,
            )
            axis.scatter(
                shape_iters[0, x_index],
                shape_iters[0, y_index],
                color=FIRST_CALL_COLOR,
                edgecolor=MARKER_EDGE_COLOR,
                s=72,
                marker="s",
                label="first call",
                zorder=5,
            )
            axis.scatter(
                final_shape[x_index],
                final_shape[y_index],
                color=FINAL_BEST_COLOR,
                edgecolor=MARKER_EDGE_COLOR,
                s=110,
                marker="*",
                label=f"final best (call {best_call})" if best_call is not None else "final best",
                zorder=6,
            )
        return scatter

    def format_axes(axis, show_ylabel=True):
        axis.set_xlabel(PARAM_LABELS[x_name])
        if show_ylabel:
            axis.set_ylabel(PARAM_LABELS[y_name])
        else:
            axis.set_ylabel("")
        axis.set_xlim(bounds[x_index][0], bounds[x_index][1])
        axis.set_ylim(bounds[y_index][0], bounds[y_index][1])

    combined_fig, axes = plt.subplots(
        1,
        3,
        figsize=(21.0, 6.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]},
    )
    comparison_values = np.concatenate([
        Z_true[np.isfinite(Z_true)],
        Z_surrogate[np.isfinite(Z_surrogate)],
    ])
    if comparison_values.size:
        levels = np.linspace(float(np.min(comparison_values)), float(np.max(comparison_values)), contour_count)
        if np.allclose(levels[0], levels[-1]):
            levels = contour_count
    else:
        levels = contour_count

    true_filled = None
    if np.isfinite(Z_true).any():
        true_filled = axes[0].contourf(X, Y, Z_true, levels=levels, cmap=LANDSCAPE_CMAP)
        axes[0].contour(X, Y, Z_true, levels=levels, colors="black", linewidths=0.35, alpha=0.45)
    else:
        axes[0].text(0.5, 0.5, "No finite true objective values", transform=axes[0].transAxes, ha="center", va="center")

    surrogate_filled = None
    if model is not None and np.isfinite(Z_surrogate).any():
        surrogate_filled = axes[1].contourf(X, Y, Z_surrogate, levels=levels, cmap=LANDSCAPE_CMAP)
        axes[1].contour(X, Y, Z_surrogate, levels=levels, colors="black", linewidths=0.35, alpha=0.45)
    else:
        axes[1].text(0.5, 0.5, "No surrogate model available", transform=axes[1].transAxes, ha="center", va="center")

    comparison_mappable = true_filled if true_filled is not None else surrogate_filled
    if comparison_mappable is not None:
        combined_fig.colorbar(
            comparison_mappable,
            ax=axes[:2],
            label=label,
            fraction=0.035,
            pad=0.02,
        )

    axes[0].set_title(f"True objective, fixed {fixed_name}={fixed_value:.4g}")
    axes[1].set_title("Surrogate model")
    if surrogate_error is not None:
        max_relative_error = max(surrogate_error["max_relative"], 1e-12)
        error_levels = np.linspace(0.0, max_relative_error, contour_count)
        error_filled = axes[2].contourf(X, Y, Z_relative_error, levels=error_levels, cmap=ERROR_CMAP, extend="max")
        axes[2].contour(X, Y, Z_relative_error, levels=error_levels, colors="black", linewidths=0.3, alpha=0.35)
        combined_fig.colorbar(
            error_filled,
            ax=axes[2],
            label="relative error",
            fraction=0.045,
            pad=0.025,
        )
        axes[2].text(
            0.02,
            0.98,
            (
                f"Mean={surrogate_error['mean_relative']:.3g}\n"
                f"RMS={surrogate_error['rmse']:.3g}\n"
                f"Max={surrogate_error['max_relative']:.3g}"
            ),
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
        )
    else:
        axes[2].text(0.5, 0.5, "No relative error available", transform=axes[2].transAxes, ha="center", va="center")
    axes[2].set_title("Relative surrogate error")

    for i, axis in enumerate(axes):
        format_axes(axis, show_ylabel=(i == 0))
    bo_scatter = add_evaluation_points(axes)
    if bo_scatter is not None:
        call_colorbar = combined_fig.colorbar(
            bo_scatter,
            ax=axes,
            orientation="horizontal",
            fraction=0.055,
            pad=0.12,
            aspect=55,
        )
        call_colorbar.set_label("BO call")
    axes[0].legend(loc="best", fontsize=8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_fig.savefig(output_path, dpi=200)
    if error_output_path is not None:
        error_output_path = Path(error_output_path)
        error_output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_fig.savefig(error_output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(combined_fig)
    return {
        "combined_fig": combined_fig,
        "comparison_fig": combined_fig,
        "error_fig": combined_fig,
        "surrogate_error": surrogate_error,
    }


def _objective_value_or_nan(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
):
    try:
        value = objective_np(
            shape,
            A=A,
            n_quad=n_quad,
            objective_name=objective_name,
            q=q,
        )
    except Exception:
        return np.nan
    return float(value) if np.isfinite(value) else np.nan


def evaluate_fixed_delta_objective_grid(
    fixed_delta,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    grid_size=61,
    n_quad=500,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
):
    """Evaluate the objective on an epsilon-kappa grid at one fixed delta."""
    bounds = _validate_bounds(bounds)
    fixed_delta = _normalize_fixed_values((None, None, fixed_delta), bounds)[2]
    objective_name = _normalize_objective_name(objective_name)
    grid_size = int(grid_size)
    if grid_size < 3:
        raise ValueError("sharp-point grid size must be at least 3")

    epsilon_values = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    kappa_values = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    Epsilon, Kappa = np.meshgrid(epsilon_values, kappa_values)
    Z = np.full_like(Epsilon, np.nan, dtype=float)

    for row, kappa in enumerate(kappa_values):
        for col, epsilon in enumerate(epsilon_values):
            Z[row, col] = _objective_value_or_nan(
                (float(epsilon), float(kappa), fixed_delta),
                A=A,
                n_quad=n_quad,
                objective_name=objective_name,
                q=q,
            )

    return epsilon_values, kappa_values, Epsilon, Kappa, Z


def find_sharp_fixed_delta_points(
    fixed_delta,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    count=3,
    grid_size=61,
    min_distance=0.08,
    n_quad=500,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
):
    """Find high-curvature candidates on the fixed-delta objective grid."""
    count = int(count)
    if count <= 0:
        return []

    bounds = _validate_bounds(bounds)
    epsilon_values, kappa_values, _, _, Z = evaluate_fixed_delta_objective_grid(
        fixed_delta,
        A=A,
        bounds=bounds,
        grid_size=grid_size,
        n_quad=n_quad,
        objective_name=objective_name,
        q=q,
    )
    finite = np.isfinite(Z)
    if np.count_nonzero(finite) < 9:
        return []

    filled = np.where(finite, Z, np.nanmedian(Z[finite]))
    dx = float(np.mean(np.diff(epsilon_values)))
    dy = float(np.mean(np.diff(kappa_values)))
    dZ_dkappa, dZ_depsilon = np.gradient(filled, dy, dx)
    d2Z_depsilon2 = np.gradient(dZ_depsilon, dx, axis=1)
    d2Z_dkappa2 = np.gradient(dZ_dkappa, dy, axis=0)
    d2Z_cross = np.gradient(dZ_depsilon, dy, axis=0)
    sharpness = np.sqrt(
        d2Z_depsilon2**2 + d2Z_dkappa2**2 + 2.0 * d2Z_cross**2
    )
    sharpness[~finite] = np.nan
    sharpness[0, :] = np.nan
    sharpness[-1, :] = np.nan
    sharpness[:, 0] = np.nan
    sharpness[:, -1] = np.nan

    scale_epsilon = bounds[0][1] - bounds[0][0]
    scale_kappa = bounds[1][1] - bounds[1][0]
    min_distance = max(0.0, float(min_distance))
    order = np.argsort(np.nan_to_num(sharpness.ravel(), nan=-np.inf))[::-1]
    selected = []
    for flat_index in order:
        score = float(sharpness.ravel()[flat_index])
        if not np.isfinite(score):
            break
        row, col = np.unravel_index(flat_index, sharpness.shape)
        epsilon = float(epsilon_values[col])
        kappa = float(kappa_values[row])
        far_enough = True
        for point in selected:
            distance = np.hypot(
                (epsilon - point["epsilon"]) / scale_epsilon,
                (kappa - point["kappa"]) / scale_kappa,
            )
            if distance < min_distance:
                far_enough = False
                break
        if not far_enough:
            continue
        selected.append(
            {
                "epsilon": epsilon,
                "kappa": kappa,
                "delta": float(fixed_delta),
                "objective": float(Z[row, col]),
                "sharpness": score,
                "gradient_norm": float(np.hypot(dZ_depsilon[row, col], dZ_dkappa[row, col])),
            }
        )
        if len(selected) >= count:
            break
    return selected


def _normalize_delta_scan_points(points, bounds):
    normalized = []
    for point in points:
        if isinstance(point, dict):
            epsilon = point["epsilon"]
            kappa = point["kappa"]
        else:
            epsilon, kappa = point[:2]
        epsilon = float(epsilon)
        kappa = float(kappa)
        if epsilon < bounds[0][0] or epsilon > bounds[0][1]:
            raise ValueError(
                f"delta scan epsilon={epsilon:g} is outside bounds "
                f"[{bounds[0][0]:g}, {bounds[0][1]:g}]"
            )
        if kappa < bounds[1][0] or kappa > bounds[1][1]:
            raise ValueError(
                f"delta scan kappa={kappa:g} is outside bounds "
                f"[{bounds[1][0]:g}, {bounds[1][1]:g}]"
            )
        normalized.append((epsilon, kappa))
    return normalized


def plot_delta_objective_scans(
    points,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    output_path=None,
    delta_count=201,
    n_quad=500,
    objective_name=DEFAULT_OBJECTIVE,
    q=DEFAULT_Q,
    reference_delta=None,
    show=False,
):
    """Plot objective(delta) while holding epsilon and kappa fixed."""
    import matplotlib.pyplot as plt

    bounds = _validate_bounds(bounds)
    objective_name = _normalize_objective_name(objective_name)
    label = objective_label(objective_name)
    points = _normalize_delta_scan_points(points, bounds)
    if not points:
        raise ValueError("at least one delta scan point is required")

    delta_count = int(delta_count)
    if delta_count < 2:
        raise ValueError("delta scan count must be at least 2")
    delta_values = np.linspace(bounds[2][0], bounds[2][1], delta_count)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    curves = []

    for index, (epsilon, kappa) in enumerate(points, start=1):
        values = np.asarray(
            [
                _objective_value_or_nan(
                    (epsilon, kappa, float(delta)),
                    A=A,
                    n_quad=n_quad,
                    objective_name=objective_name,
                    q=q,
                )
                for delta in delta_values
            ],
            dtype=float,
        )
        color = DELTA_SCAN_COLORS[(index - 1) % len(DELTA_SCAN_COLORS)]
        curve_label = rf"{index}: $\epsilon$={epsilon:.4g}, $\kappa$={kappa:.4g}"
        ax.plot(delta_values, values, color=color, linewidth=1.8, label=curve_label)

        finite = np.isfinite(values)
        summary = {
            "epsilon": epsilon,
            "kappa": kappa,
            "finite_count": int(np.count_nonzero(finite)),
            "reference_delta": reference_delta,
            "reference_value": np.nan,
            "best_delta": np.nan,
            "best_value": np.nan,
            "min_value": np.nan,
        }
        if finite.any():
            best_index = int(np.nanargmax(values))
            summary["best_delta"] = float(delta_values[best_index])
            summary["best_value"] = float(values[best_index])
            summary["min_value"] = float(np.nanmin(values))
            ax.scatter(
                [summary["best_delta"]],
                [summary["best_value"]],
                color=color,
                edgecolor=MARKER_EDGE_COLOR,
                linewidth=0.4,
                marker="*",
                s=90,
                zorder=4,
            )
        if reference_delta is not None and bounds[2][0] <= reference_delta <= bounds[2][1]:
            reference_value = _objective_value_or_nan(
                (epsilon, kappa, float(reference_delta)),
                A=A,
                n_quad=n_quad,
                objective_name=objective_name,
                q=q,
            )
            summary["reference_value"] = reference_value
            if np.isfinite(reference_value):
                ax.scatter(
                    [float(reference_delta)],
                    [reference_value],
                    color=color,
                    edgecolor=MARKER_EDGE_COLOR,
                    linewidth=0.4,
                    s=36,
                    zorder=5,
                )
        curves.append(summary)

    if reference_delta is not None and bounds[2][0] <= reference_delta <= bounds[2][1]:
        ax.axvline(float(reference_delta), color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(PARAM_LABELS["delta"])
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs delta at fixed epsilon,kappa")
    ax.set_xlim(bounds[2][0], bounds[2][1])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "fig": fig,
        "curves": curves,
        "delta_values": delta_values,
    }


def _format_shape(shape):
    epsilon, kappa, delta = [float(v) for v in shape]
    return f"epsilon={epsilon:.12g}, kappa={kappa:.12g}, delta={delta:.12g}"


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--fix-epsilon", type=float, help="fix epsilon and optimize kappa,delta")
    parser.add_argument("--fix-kappa", type=float, help="fix kappa and optimize epsilon,delta")
    parser.add_argument("--fix-delta", type=float, help="fix delta and optimize epsilon,kappa")
    parser.add_argument("--epsilon-bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[0], metavar=("LOW", "HIGH"))
    parser.add_argument("--kappa-bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[1], metavar=("LOW", "HIGH"))
    parser.add_argument("--delta-bounds", nargs=2, type=float, default=DEFAULT_BOUNDS[2], metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--objective",
        choices=tuple(OBJECTIVE_LABELS),
        default=DEFAULT_OBJECTIVE,
        help=(
            "objective to maximize: volume_average_pressure, normalized_psi_pressure, "
            "beta_toroidal_fixed_q, or beta_toroidal_variable_q"
        ),
    )
    parser.add_argument("--q", type=float, default=DEFAULT_Q, help="fixed q used by beta_toroidal_fixed_q")
    parser.add_argument("--n-quad", type=int, default=2048)
    parser.add_argument("--n-calls", type=int, default=50)
    parser.add_argument("--n-initial-points", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--surrogate",
        choices=("gp", "forest", "gbrt", "dummy"),
        default="gp",
        help=(
            "surrogate model for BO: gp=Gaussian Process, "
            "forest=Random Forest, gbrt=Gradient Boosting, dummy=Random Search"
        ),
    )
    parser.add_argument("--plot", type=Path, help="save optimized flux contours to this path")
    parser.add_argument("--plot-grid", type=int, default=600)
    parser.add_argument("--contour-count", type=int, default=20)
    parser.add_argument("--objective-plot", type=Path, help="save combined true/surrogate/relative-error contour plot to this path")
    parser.add_argument("--surrogate-error-plot", type=Path, help="save combined surrogate diagnostic plot to this path")
    parser.add_argument("--objective-plot-grid", type=int, default=35)
    parser.add_argument("--objective-contour-count", type=int, default=20)
    parser.add_argument("--objective-plot-n-quad", type=int, help="N used for the objective contour grid; defaults to --n-quad")
    parser.add_argument("--trajectory-every", type=int, default=5, help="legacy option kept for compatibility; evaluation points are plotted without trajectory lines")
    parser.add_argument("--sharp-points", type=int, default=0, help="find this many high-curvature candidates on the fixed-delta epsilon,kappa objective grid")
    parser.add_argument("--sharp-grid", type=int, default=61, help="grid size used for high-curvature point detection")
    parser.add_argument("--sharp-min-distance", type=float, default=0.08, help="minimum normalized spacing between reported sharp-point candidates")
    parser.add_argument(
        "--delta-scan-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("EPSILON", "KAPPA"),
        help="add a fixed epsilon,kappa point for objective-vs-delta scanning; repeat for multiple curves",
    )
    parser.add_argument("--delta-scan-plot", type=Path, help="save objective-vs-delta scan plot to this path")
    parser.add_argument("--delta-scan-count", type=int, default=201, help="number of delta samples in each one-parameter scan")
    parser.add_argument("--delta-scan-n-quad", type=int, help="N used for delta scans; defaults to --n-quad")
    parser.add_argument("--show-plot", action="store_true", default=True, help="show plot on completion (default: True)")
    parser.add_argument("--no-show-plot", action="store_true", help="do not show plot (overrides --show-plot)")
    parser.add_argument("--gradient-step", type=float, default=1e-3, help="finite-difference step for final gradient diagnostics")
    parser.add_argument("--no-gradient-diagnostics", action="store_true", help="do not print final finite-difference gradients")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--test-perturbation-radius", type=float, default=1e-3)
    parser.add_argument("--test-perturbation-samples", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=0)
    return parser


def main(argv=None):
    run_start = time.perf_counter()
    parser = _build_parser()
    args = parser.parse_args(argv)
    bounds = (tuple(args.epsilon_bounds), tuple(args.kappa_bounds), tuple(args.delta_bounds))
    fixed_values = (args.fix_epsilon, args.fix_kappa, args.fix_delta)
    fixed_count = sum(value is not None for value in fixed_values)
    if fixed_count > 1:
        parser.error("fix at most one of --fix-epsilon, --fix-kappa, or --fix-delta")
    if (args.objective_plot or args.surrogate_error_plot) and fixed_count != 1:
        parser.error("--objective-plot and --surrogate-error-plot require exactly one fixed parameter")
    if args.sharp_points < 0:
        parser.error("--sharp-points must be non-negative")
    if args.sharp_points and args.fix_delta is None:
        parser.error("--sharp-points requires --fix-delta because it analyzes an epsilon,kappa slice")
    if args.sharp_grid < 3:
        parser.error("--sharp-grid must be at least 3")
    if args.delta_scan_count < 2:
        parser.error("--delta-scan-count must be at least 2")
    if args.delta_scan_plot and not args.delta_scan_point and not args.sharp_points:
        parser.error("--delta-scan-plot requires --delta-scan-point or --sharp-points")

    optimization_start = time.perf_counter()
    result = optimize_shape_bayesian(
        bounds=bounds,
        A=args.A,
        objective_name=args.objective,
        q=args.q,
        fixed_values=fixed_values,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=args.random_state,
        n_quad=args.n_quad,
        surrogate=args.surrogate,
    )
    optimization_elapsed = time.perf_counter() - optimization_start
    diagnostics_elapsed = 0.0
    tests_elapsed = 0.0
    plot_elapsed = 0.0
    scan_elapsed = 0.0

    optimizer = result["optimizer"]
    show_flag = bool(args.show_plot)
    if getattr(args, "no_show_plot", False):
        show_flag = False

    if result["fixed_indices"]:
        optimized_names = ", ".join(PARAM_NAMES[i] for i in result["free_indices"])
        print(f"Fixed parameter: {_format_fixed_values(result['fixed_values'])}")
        print(f"Optimized parameters: {optimized_names}")
    print(f"Objective: {result['objective_label']}")
    if result["objective_name"] == "beta_toroidal_fixed_q":
        print(f"Fixed q: {float(result['q']):.12g}")
    print(f"Initial shape: {_format_shape(result['initial_shape'])}")
    print(f"Initial objective value: {float(result['initial_objective_value']):.12g}")
    print(f"Final shape: {_format_shape(result['shape'])}")
    print(f"Final objective value: {float(result['objective_value']):.12g}")
    print(f"Final volume_average_pressure: {float(result['final_volume_average_pressure']):.12g}")
    print(f"Best call: {result.get('best_call', 'unknown')}")
    print(f"Bayesian calls: {result.get('calls', 'unknown')}, initial points: {result.get('initial_points', 'unknown')}")
    print(f"Bayesian optimization runtime: {optimization_elapsed:.3f} s")
    if hasattr(optimizer, "status"):
        print(f"Optimizer status: {int(optimizer.status)}")
    if hasattr(optimizer, "success"):
        print(f"Optimizer success: {bool(optimizer.success)}")
    if not args.no_gradient_diagnostics:
        diagnostics_start = time.perf_counter()
        print_final_gradient_diagnostics(
            result,
            A=args.A,
            bounds=bounds,
            n_quad=args.n_quad,
            objective_name=args.objective,
            q=args.q,
            step=args.gradient_step,
        )
        diagnostics_elapsed = time.perf_counter() - diagnostics_start
    if args.run_tests:
        tests_start = time.perf_counter()
        run_unit_tests(
            result,
            A=args.A,
            bounds=bounds,
            n_quad=args.n_quad,
            objective_name=args.objective,
            q=args.q,
            perturbation_radius=args.test_perturbation_radius,
            perturbation_samples=args.test_perturbation_samples,
            perturbation_seed=args.test_seed,
        )
        tests_elapsed = time.perf_counter() - tests_start
    if result["fixed_indices"] and (args.objective_plot or args.surrogate_error_plot or show_flag):
        objective_plot_n_quad = args.objective_plot_n_quad or args.n_quad
        print("Computing fixed-parameter objective and surrogate contour plots...")
        plot_start = time.perf_counter()
        plot_result = plot_fixed_parameter_objective_contours(
            result,
            A=args.A,
            bounds=bounds,
            fixed_values=result["fixed_values"],
            output_path=args.objective_plot,
            error_output_path=args.surrogate_error_plot,
            grid_size=args.objective_plot_grid,
            contour_count=args.objective_contour_count,
            trajectory_every=args.trajectory_every,
            n_quad=objective_plot_n_quad,
            objective_name=args.objective,
            q=args.q,
            show=show_flag,
        )
        plot_elapsed += time.perf_counter() - plot_start
        surrogate_error = plot_result.get("surrogate_error")
        if surrogate_error is not None:
            print(
                "Relative surrogate error: "
                f"Mean={surrogate_error['mean_relative']:.12g}, "
                f"RMS={surrogate_error['rmse']:.12g}, "
                f"Max={surrogate_error['max_relative']:.12g}"
            )
        if args.objective_plot:
            print(f"Saved combined objective/surrogate/error plot: {args.objective_plot}")
        if args.surrogate_error_plot:
            print(f"Saved combined surrogate diagnostic plot: {args.surrogate_error_plot}")
    delta_scan_points = []
    if args.sharp_points:
        sharp_n_quad = args.objective_plot_n_quad or args.delta_scan_n_quad or args.n_quad
        print("Finding high-curvature candidates on the fixed-delta objective grid...")
        scan_start = time.perf_counter()
        sharp_points = find_sharp_fixed_delta_points(
            args.fix_delta,
            A=args.A,
            bounds=bounds,
            count=args.sharp_points,
            grid_size=args.sharp_grid,
            min_distance=args.sharp_min_distance,
            n_quad=sharp_n_quad,
            objective_name=args.objective,
            q=args.q,
        )
        scan_elapsed += time.perf_counter() - scan_start
        if sharp_points:
            print("High-curvature candidates:")
            for index, point in enumerate(sharp_points, start=1):
                print(
                    f"  {index}. epsilon={point['epsilon']:.12g}, "
                    f"kappa={point['kappa']:.12g}, "
                    f"delta={point['delta']:.12g}, "
                    f"objective={point['objective']:.12g}, "
                    f"sharpness={point['sharpness']:.12g}"
                )
            delta_scan_points.extend(
                (point["epsilon"], point["kappa"])
                for point in sharp_points
            )
        else:
            print("No finite high-curvature candidates found.")
    if args.delta_scan_point:
        delta_scan_points.extend(
            (float(epsilon), float(kappa))
            for epsilon, kappa in args.delta_scan_point
        )
    if delta_scan_points:
        scan_n_quad = args.delta_scan_n_quad or args.n_quad
        print("Computing objective-vs-delta scans...")
        scan_start = time.perf_counter()
        scan_result = plot_delta_objective_scans(
            delta_scan_points,
            A=args.A,
            bounds=bounds,
            output_path=args.delta_scan_plot,
            delta_count=args.delta_scan_count,
            n_quad=scan_n_quad,
            objective_name=args.objective,
            q=args.q,
            reference_delta=args.fix_delta,
            show=show_flag,
        )
        scan_elapsed += time.perf_counter() - scan_start
        print("Delta scan summaries:")
        for index, curve in enumerate(scan_result["curves"], start=1):
            if np.isfinite(curve["best_value"]):
                reference_text = ""
                if np.isfinite(curve["reference_value"]):
                    reference_text = (
                        f", value at delta={curve['reference_delta']:.12g}: "
                        f"{curve['reference_value']:.12g}"
                    )
                print(
                    f"  {index}. epsilon={curve['epsilon']:.12g}, "
                    f"kappa={curve['kappa']:.12g}: "
                    f"best delta={curve['best_delta']:.12g}, "
                    f"best objective={curve['best_value']:.12g}, "
                    f"min objective={curve['min_value']:.12g}, "
                    f"finite samples={curve['finite_count']}"
                    f"{reference_text}"
                )
            else:
                print(
                    f"  {index}. epsilon={curve['epsilon']:.12g}, "
                    f"kappa={curve['kappa']:.12g}: no finite objective values"
                )
        if args.delta_scan_plot:
            print(f"Saved delta scan plot: {args.delta_scan_plot}")
    if args.plot or show_flag:
        plot_start = time.perf_counter()
        plot_flux_contours(
            result["shape"],
            coefficients=result["coefficients"],
            A=args.A,
            output_path=args.plot,
            grid_size=args.plot_grid,
            contour_count=args.contour_count,
            show=show_flag,
        )
        plot_elapsed += time.perf_counter() - plot_start
        if args.plot:
            print(f"Saved plot: {args.plot}")
    total_elapsed = time.perf_counter() - run_start
    print("Runtime:")
    print(f"  Bayesian optimization: {optimization_elapsed:.3f} s")
    if not args.no_gradient_diagnostics:
        print(f"  Gradient diagnostics: {diagnostics_elapsed:.3f} s")
    if args.run_tests:
        print(f"  Unit tests: {tests_elapsed:.3f} s")
    if plot_elapsed:
        print(f"  Plot/display: {plot_elapsed:.3f} s")
    if scan_elapsed:
        print(f"  Delta scans: {scan_elapsed:.3f} s")
    print(f"  Total elapsed: {total_elapsed:.3f} s")


if __name__ == "__main__":
    main()
