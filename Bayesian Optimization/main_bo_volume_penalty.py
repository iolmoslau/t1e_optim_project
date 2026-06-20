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
    PROJECT_ROOT / "optimal_JAX",
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
from pressure_integral import pressure_utils

# Fix scikit-optimize GBRT compatibility with newer scikit-learn tag handling.
GradientBoostingQuantileRegressor.__sklearn_tags__ = lambda self: default_tags(self)


DEFAULT_A = -0.05
DEFAULT_OBJECTIVE = "normalized_psi_pressure"
DEFAULT_VOLUME_PENALTY = 1.0
DEFAULT_TARGET_SHAPE = (0.31, 1.97, 0.54)
DEFAULT_R0 = 1.0
DEFAULT_I = 1.0
DEFAULT_P0 = 2e6
DEFAULT_B0 = 2.0
DEFAULT_Q_MIN = 2.0
DEFAULT_KAPPA_MAX = 2.1
DEFAULT_Q_PENALTY = 1.0
DEFAULT_KAPPA_PENALTY = 1.0
#DEFAULT_TARGET_SHAPE = (0.3, 1.5, 0.0)
#DEFAULT_BOUNDS = ((0.1, 0.45), (1, 1.7), (-0.3, 0.3))  # epsilon,kappa,delta
#DEFAULT_BOUNDS = ((0.1, 0.45), (1, 1.7), (-0.3, 0.3))  # epsilon,kappa,delta
DEFAULT_BOUNDS = ((0.1, 0.5), (1,4.0), (-0.5, 0.5))  # epsilon,kappa,delta
PARAM_NAMES = ("epsilon", "kappa", "delta")
PARAM_LABELS = {
    "epsilon": r"$\epsilon$",
    "kappa": r"$\kappa$",
    "delta": r"$\delta$",
}
OBJECTIVE_LABELS = {
    "normalized_psi_pressure": "normalized_psi_pressure_volume_constrained",
    "beta_toroidal_variable_q": "beta_toroidal_variable_q_volume_constrained",
    "beta_toroidal_updated": "beta_toroidal_updated_volume_q_kappa_constrained",
    "beta_toroidal_updated_no_kappa_penalty": "beta_toroidal_updated_volume_q_constrained",
}
LANDSCAPE_COMPONENT_ALIASES = {
    "penalized": "penalized",
    "objective": "penalized",
    "penalised": "penalized",
    "raw": "raw",
    "base": "raw",
    "unpenalized": "raw",
    "unpenalised": "raw",
    "optimal_new_beta_t": "optimal_new_beta_t",
    "optimal-new-beta-t": "optimal_new_beta_t",
    "new_beta_t": "optimal_new_beta_t",
    "friend": "optimal_new_beta_t",
    "friend-style": "optimal_new_beta_t",
}
LANDSCAPE_COMPONENT_LABELS = {
    "penalized": "penalized objective",
    "raw": "raw base objective (no penalties)",
    "optimal_new_beta_t": "optimal_new_beta_t raw beta_t with V_sep",
}
LANDSCAPE_CMAP = "plasma"
CALL_CMAP = "plasma"
NON_FEASIBLE_CMAP_FACTOR = 0.65
FIRST_CALL_COLOR = "#E69F00"
FINAL_BEST_COLOR = "#0072B2"
Q_CONSTRAINT_COLOR = "#56B4E9"
KAPPA_CONSTRAINT_COLOR = "#CC79A7"
VOLUME_CONSTRAINT_COLOR = "#FF0000"
MACHINE_PRESETS = {
    "sparc": {"label": "SPARC", "R0": 1.85, "I": 8.7e6, "B0": 12.2},
    "c-mod": {"label": "C-Mod", "R0": 0.67, "I": 2.0e6, "B0": 8.0},
    "aug": {"label": "AUG", "R0": 1.65, "I": 1.6e6, "B0": 3.9},
    "diii-d": {"label": "DIII-D", "R0": 1.66, "I": 2.0e6, "B0": 2.2},
    "east": {"label": "EAST", "R0": 1.70, "I": 1.0e6, "B0": 3.5},
    "kstar": {"label": "KSTAR", "R0": 1.80, "I": 2.0e6, "B0": 3.5},
    "ignitor": {"label": "Ignitor", "R0": 1.32, "I": 11.0e6, "B0": 13.0},
    "cit": {"label": "CIT", "R0": 2.10, "I": 11.0e6, "B0": 10.0},
    "fire": {"label": "FIRE", "R0": 2.14, "I": 7.7e6, "B0": 10.0},
    "bpx": {"label": "BPX", "R0": 2.59, "I": 11.8e6, "B0": 9.0},
    "iter": {"label": "ITER", "R0": 6.20, "I": 15.0e6, "B0": 5.3},
}
MACHINE_ALIASES = {
    "cmod": "c-mod",
    "c_mod": "c-mod",
    "diiid": "diii-d",
    "diii_d": "diii-d",
    "d3d": "diii-d",
}

_OPTIMAL_NEW_BETA_LANDSCAPE = None


def _normalize_objective_name(objective_name):
    objective_name = str(objective_name).lower()
    aliases = {
        "psi_normalized_pressure": "normalized_psi_pressure",
        "psi_normalised_pressure": "normalized_psi_pressure",
        "beta_t_variable_q": "beta_toroidal_variable_q",
        "beta_t_alternative": "beta_toroidal_variable_q",
        "beta_t_updated": "beta_toroidal_updated",
        "beta_toroidal_updated_constrained": "beta_toroidal_updated",
        "beta_toroidal_updated_q_constrained": "beta_toroidal_updated_no_kappa_penalty",
        "beta_toroidal_updated_no_kappa": "beta_toroidal_updated_no_kappa_penalty",
        "beta_t_updated_no_kappa": "beta_toroidal_updated_no_kappa_penalty",
    }
    objective_name = aliases.get(objective_name, objective_name)
    if objective_name not in OBJECTIVE_LABELS:
        choices = ", ".join(OBJECTIVE_LABELS)
        raise ValueError(f"unknown objective {objective_name!r}; choose one of: {choices}")
    return objective_name


def objective_label(objective_name):
    return OBJECTIVE_LABELS[_normalize_objective_name(objective_name)]


def _is_updated_beta_objective(objective_name):
    return _normalize_objective_name(objective_name) in {
        "beta_toroidal_updated",
        "beta_toroidal_updated_no_kappa_penalty",
    }


def _normalize_landscape_component(component=None, landscape_raw=False):
    if component is None:
        return "raw" if landscape_raw else "penalized"
    key = str(component).strip().lower()
    key = LANDSCAPE_COMPONENT_ALIASES.get(key, key)
    if key not in LANDSCAPE_COMPONENT_LABELS:
        choices = ", ".join(sorted(LANDSCAPE_COMPONENT_LABELS))
        raise ValueError(f"unknown landscape component {component!r}; choose one of: {choices}")
    return key


def _optimal_new_beta_landscape_module():
    global _OPTIMAL_NEW_BETA_LANDSCAPE
    if _OPTIMAL_NEW_BETA_LANDSCAPE is None:
        try:
            import landscape_beta_t_update as landscape_module
        except ModuleNotFoundError:
            import landscape_new_beta_t as landscape_module

        _OPTIMAL_NEW_BETA_LANDSCAPE = landscape_module
    return _OPTIMAL_NEW_BETA_LANDSCAPE


def _optimal_new_beta_target_volume(n_quad):
    landscape_module = _optimal_new_beta_landscape_module()
    return float(landscape_module.beta_opt.sep_volume(point_count=int(n_quad)))


def _optimal_new_beta_sep_shape():
    landscape_module = _optimal_new_beta_landscape_module()
    return tuple(float(value) for value in landscape_module.beta_opt.SEP_SHAPE)


def _optimal_new_beta_landscape_values(shape, A=DEFAULT_A, P0=DEFAULT_P0, n_quad=500):
    landscape_module = _optimal_new_beta_landscape_module()
    beta_t = landscape_module.beta_t_landscape_from_shape(
        shape,
        p_0=float(P0),
        A=float(A),
        N=int(n_quad),
    )
    volume = landscape_module.volume_landscape_from_shape(shape, point_count=int(n_quad))
    q_star = landscape_module.q_star_landscape_from_shape(shape)
    return float(beta_t), float(volume), float(q_star)


def _normalize_machine_name(machine_name):
    if machine_name is None:
        return None
    machine_name = str(machine_name).strip().lower()
    machine_name = MACHINE_ALIASES.get(machine_name, machine_name)
    if machine_name not in MACHINE_PRESETS:
        choices = ", ".join(sorted(MACHINE_PRESETS))
        raise ValueError(f"unknown machine {machine_name!r}; choose one of: {choices}")
    return machine_name


def machine_preset(machine_name):
    machine_name = _normalize_machine_name(machine_name)
    if machine_name is None:
        return None
    return MACHINE_PRESETS[machine_name]


def darkened_colormap(name, factor=NON_FEASIBLE_CMAP_FACTOR):
    """Return a darker copy of an existing Matplotlib colormap."""
    import matplotlib.colors
    import matplotlib.pyplot as plt

    colors = plt.get_cmap(name)(np.linspace(0.0, 1.0, 256))
    colors[:, :3] *= float(factor)
    return matplotlib.colors.ListedColormap(colors, name=f"{name}_dark")


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


def volume_from_shape_np(shape_values, A=DEFAULT_A, n_quad=2048):
    e, k, d = [float(value) for value in shape_values]
    psi, _, _ = pressure_utils.make_psi(e, k, d, A)
    x_lim, y_lim = pressure_utils._plasma_domain(e, k)
    xs, ys = pressure_utils.extract_zero_contour(psi, x_lim, y_lim, n=int(n_quad))
    return pressure_utils.int_contour_boundary(
        pressure_utils._volume_integrand,
        xs,
        ys,
    )


def objective_np(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
):
    """Evaluate the selected constrained objective.

    This wrapper is intended for Bayesian optimization.
    """
    return objective_components_np(
        shape,
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        volume_penalty=volume_penalty,
        target_shape=target_shape,
        R0=R0,
        I=I,
        P0=P0,
        B0=B0,
        q_min=q_min,
        kappa_max=kappa_max,
        q_penalty=q_penalty,
        kappa_penalty=kappa_penalty,
    )["objective"]


def objective_components_np(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
):
    """Return objective components, including final volume diagnostics."""
    objective_name = _normalize_objective_name(objective_name)
    epsilon, kappa, delta = shape
    n_quad = int(n_quad)

    with np.errstate(divide="ignore", invalid="ignore"):
        if objective_name == "normalized_psi_pressure":
            base_objective = pressure_utils.normalized_psi_pressure(
                epsilon,
                kappa,
                delta,
                A=A,
                method="contour",
                N=n_quad,
            )
        elif objective_name == "beta_toroidal_variable_q":
            base_objective = pressure_utils.beta_t_alternative(
                epsilon,
                kappa,
                delta,
                A=A,
                N=n_quad,
            )
        elif _is_updated_beta_objective(objective_name):
            base_objective = pressure_utils.beta_toroidal_updated(
                epsilon,
                kappa,
                delta,
                R0,
                I,
                P0,
                B0,
                A=A,
                N=n_quad,
            )
        else:
            raise ValueError(f"unsupported objective {objective_name!r}")

        volume = volume_from_shape_np((epsilon, kappa, delta), A=A, n_quad=n_quad)
        if target_shape is None:
            raise ValueError("target_shape is required when target_volume is not supplied")
        target_volume = volume_from_shape_np(target_shape, A=A, n_quad=n_quad)
        volume_error = volume - target_volume
        volume_penalty_term = volume_penalty * volume_error**2

        q_star = np.nan
        q_violation = 0.0
        q_penalty_term = 0.0
        kappa_violation = 0.0
        kappa_penalty_term = 0.0
        if _is_updated_beta_objective(objective_name):
            q_star = pressure_utils.q_star_updated(epsilon, kappa, delta, R0, I, B0)
            q_violation = max(0.0, q_min - q_star)
            kappa_violation = max(0.0, kappa - kappa_max)
            q_penalty_term = q_penalty * q_violation**2
            if objective_name == "beta_toroidal_updated":
                kappa_penalty_term = kappa_penalty * kappa_violation**2

        penalty_term = volume_penalty_term + q_penalty_term + kappa_penalty_term
        objective = base_objective - penalty_term

    return {
        "objective": float(objective),
        "base_objective": float(base_objective),
        "volume": float(volume),
        "target_volume": float(target_volume),
        "volume_error": float(volume_error),
        "penalty": float(volume_penalty),
        "volume_penalty_term": float(volume_penalty_term),
        "q_star": float(q_star),
        "q_violation": float(q_violation),
        "q_penalty": float(q_penalty),
        "q_penalty_term": float(q_penalty_term),
        "kappa_violation": float(kappa_violation),
        "kappa_penalty": float(kappa_penalty),
        "kappa_penalty_term": float(kappa_penalty_term),
        "penalty_term": float(penalty_term),
    }


def negative_volume_penalty_objective_np(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
):
    """Negated objective for minimizers expecting lower-is-better."""
    return -objective_np(
        shape,
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        volume_penalty=volume_penalty,
        target_shape=target_shape,
        R0=R0,
        I=I,
        P0=P0,
        B0=B0,
        q_min=q_min,
        kappa_max=kappa_max,
        q_penalty=q_penalty,
        kappa_penalty=kappa_penalty,
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
    fixed_values=None,
    n_calls=50,
    n_initial_points=10,
    random_state=0,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
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
    objective = lambda x: negative_volume_penalty_objective_np(
        _shape_from_free_values(x, fixed_values),
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        volume_penalty=volume_penalty,
        target_shape=target_shape,
        R0=R0,
        I=I,
        P0=P0,
        B0=B0,
        q_min=q_min,
        kappa_max=kappa_max,
        q_penalty=q_penalty,
        kappa_penalty=kappa_penalty,
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
    best_volume_components = objective_components_np(
        best_shape,
        A=A,
        n_quad=n_quad,
        objective_name=objective_name,
        volume_penalty=volume_penalty,
        target_shape=target_shape,
        R0=R0,
        I=I,
        P0=P0,
        B0=B0,
        q_min=q_min,
        kappa_max=kappa_max,
        q_penalty=q_penalty,
        kappa_penalty=kappa_penalty,
    )
    best_volume_penalty_objective = best_volume_components["objective"]
    best_volume_average_pressure = float(
        pressure_utils.get_vol_av_p_from_params(
            best_shape[0],
            best_shape[1],
            best_shape[2],
            A=A,
            method="contour",
            N=int(n_quad),
        )
    )
    best_normalized_psi_pressure = float(
        pressure_utils.normalized_psi_pressure(
            best_shape[0],
            best_shape[1],
            best_shape[2],
            A=A,
            method="contour",
            N=int(n_quad),
        )
    )
    best_scaled_normalized_psi_pressure = float(P0) * best_normalized_psi_pressure
    target_volume = float(best_volume_components["target_volume"])
    absolute_volume_error = abs(float(best_volume_components["volume_error"]))
    relative_volume_error = (
        absolute_volume_error / abs(target_volume)
        if target_volume != 0.0
        else np.nan
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
        first_volume_penalty_objective = objective_np(
            first_shape,
            A=A,
            n_quad=n_quad,
            objective_name=objective_name,
            volume_penalty=volume_penalty,
            target_shape=target_shape,
            R0=R0,
            I=I,
            P0=P0,
            B0=B0,
            q_min=q_min,
            kappa_max=kappa_max,
            q_penalty=q_penalty,
            kappa_penalty=kappa_penalty,
        )
    else:
        first_shape = best_shape
        first_volume_penalty_objective = best_volume_penalty_objective

    return {
        "initial_shape": jnp.asarray(first_shape),
        "initial_volume_penalty_objective": first_volume_penalty_objective,
        "raw_initial": unconstrained_from_bounded(jnp.asarray(first_shape), bounds),
        "shape": jnp.asarray(best_shape),
        "volume_penalty_objective": best_volume_penalty_objective,
        "final_volume_average_pressure": best_volume_average_pressure,
        "final_normalized_psi_pressure": best_normalized_psi_pressure,
        "final_P0_times_normalized_psi_pressure": best_scaled_normalized_psi_pressure,
        "objective_name": objective_name,
        "objective_label": objective_label(objective_name),
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
        "volume_penalty": float(volume_penalty),
        "target_shape": tuple(float(value) for value in target_shape),
        "R0": float(R0),
        "I": float(I),
        "P0": float(P0),
        "B0": float(B0),
        "q_min": float(q_min),
        "kappa_max": float(kappa_max),
        "q_penalty": float(q_penalty),
        "kappa_penalty": float(kappa_penalty),
        "final_base_objective": best_volume_components["base_objective"],
        "final_volume": best_volume_components["volume"],
        "target_volume": target_volume,
        "volume_error": best_volume_components["volume_error"],
        "absolute_volume_error": absolute_volume_error,
        "relative_volume_error": relative_volume_error,
        "volume_penalty_term": best_volume_components["volume_penalty_term"],
        "q_star": best_volume_components["q_star"],
        "q_violation": best_volume_components["q_violation"],
        "q_penalty_term": best_volume_components["q_penalty_term"],
        "kappa_violation": best_volume_components["kappa_violation"],
        "kappa_penalty_term": best_volume_components["kappa_penalty_term"],
        "penalty_term": best_volume_components["penalty_term"],
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
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
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
    label = objective_label(objective_name)
    optimum_value = float(result["volume_penalty_objective"])
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
                volume_penalty=volume_penalty,
                target_shape=target_shape,
                R0=R0,
                I=I,
                P0=P0,
                B0=B0,
                q_min=q_min,
                kappa_max=kappa_max,
                q_penalty=q_penalty,
                kappa_penalty=kappa_penalty,
            )
            for point in points
        ]
    )
    max_perturbed_value = float(np.max(perturbed_values))

    checks = {
        "perturbation": max_perturbed_value <= optimum_value + perturbation_tol,
    }

    free_names = [names[i] for i in free_indices] or ["none"]
    print("Unit tests:")
    print(f"  Free variables: {', '.join(free_names)}")
    print(f"  Max perturbed {label}: {max_perturbed_value:.12g}")
    print(f"  Optimum {label}: {optimum_value:.12g}")

    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"failed unit checks: {', '.join(failures)}")
    print("  All unit checks passed.")

    return {
        "max_perturbed_volume_penalty_objective": max_perturbed_value,
        "checks": checks,
    }


def finite_difference_objective_gradient(
    shape,
    A=DEFAULT_A,
    bounds=DEFAULT_BOUNDS,
    n_quad=2048,
    objective_name=DEFAULT_OBJECTIVE,
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
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
                volume_penalty=volume_penalty,
                target_shape=target_shape,
                R0=R0,
                I=I,
                P0=P0,
                B0=B0,
                q_min=q_min,
                kappa_max=kappa_max,
                q_penalty=q_penalty,
                kappa_penalty=kappa_penalty,
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
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
    step=1e-3,
):
    objective_name = _normalize_objective_name(objective_name or result.get("objective_name", DEFAULT_OBJECTIVE))
    diagnostics = finite_difference_objective_gradient(
        result["shape"],
        A=A,
        bounds=bounds,
        n_quad=n_quad,
        objective_name=objective_name,
        volume_penalty=volume_penalty,
        target_shape=target_shape,
        R0=R0,
        I=I,
        P0=P0,
        B0=B0,
        q_min=q_min,
        kappa_max=kappa_max,
        q_penalty=q_penalty,
        kappa_penalty=kappa_penalty,
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
    colorbar=True,
    show_y_axis=True,
    show=False,
):
    """Plot plasma contours using the shared pressure_utils implementation."""
    import matplotlib.pyplot as plt

    epsilon, kappa, delta = [float(v) for v in shape]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pressure_utils.plot_plasma_profile(
        epsilon,
        kappa,
        delta,
        A=A,
        N=int(grid_size),
        n_levels=int(contour_count),
        colorbar=bool(colorbar),
        title=True,
        ylabel=bool(show_y_axis),
        ax=ax,
    )
    if not show_y_axis:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

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
    grid_size=35,
    contour_count=20,
    trajectory_every=5,
    n_quad=500,
    objective_name=None,
    volume_penalty=DEFAULT_VOLUME_PENALTY,
    target_shape=DEFAULT_TARGET_SHAPE,
    R0=DEFAULT_R0,
    I=DEFAULT_I,
    P0=DEFAULT_P0,
    B0=DEFAULT_B0,
    q_min=DEFAULT_Q_MIN,
    kappa_max=DEFAULT_KAPPA_MAX,
    q_penalty=DEFAULT_Q_PENALTY,
    kappa_penalty=DEFAULT_KAPPA_PENALTY,
    landscape_raw=False,
    landscape_component=None,
    colorbar=True,
    call_colorbar=True,
    show_y_axis=True,
    show=False,
):
    """Plot a 2D objective landscape with all BO evaluation points when one parameter is fixed."""
    import matplotlib.pyplot as plt
    import numpy as np

    bounds = _validate_bounds(bounds)
    fixed_values = result.get("fixed_values", fixed_values)
    fixed_values = _normalize_fixed_values(fixed_values, bounds)
    fixed_indices, free_indices = _fixed_and_free_indices(fixed_values)

    if len(fixed_indices) != 1 or len(free_indices) != 2:
        raise ValueError("objective contour plots require exactly one fixed parameter")

    objective_name = _normalize_objective_name(objective_name or result.get("objective_name", DEFAULT_OBJECTIVE))
    label = objective_label(objective_name)
    landscape_component = _normalize_landscape_component(landscape_component, landscape_raw=landscape_raw)
    if landscape_component == "penalized":
        landscape_label = label
    elif landscape_component == "raw":
        landscape_label = f"raw {objective_name}"
    else:
        landscape_label = r"optimal_new_beta_t raw $\beta_t$"
    normalized_objective_name = _normalize_objective_name(objective_name)
    show_updated_beta_constraints = _is_updated_beta_objective(normalized_objective_name)

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
    Z = np.full_like(X, np.nan, dtype=float)
    Volume = np.full_like(X, np.nan, dtype=float)
    Q_star = np.full_like(X, np.nan, dtype=float)
    target_volume = np.nan
    if landscape_component == "optimal_new_beta_t":
        try:
            target_volume = _optimal_new_beta_target_volume(n_quad)
            print(
                "Landscape V_sep source: "
                f"optimal_new_beta_t.SEP_SHAPE={_format_shape(_optimal_new_beta_sep_shape())}"
            )
        except Exception:
            target_volume = np.nan
    elif target_shape is not None:
        try:
            target_volume = volume_from_shape_np(target_shape, A=A, n_quad=n_quad)
        except Exception:
            target_volume = np.nan

    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            shape = [None, None, None]
            shape[fixed_index] = fixed_value
            shape[x_index] = float(x_value)
            shape[y_index] = float(y_value)
            try:
                if landscape_component == "optimal_new_beta_t":
                    beta_t, volume, q_star = _optimal_new_beta_landscape_values(
                        tuple(shape),
                        A=A,
                        P0=P0,
                        n_quad=n_quad,
                    )
                    Z[row, col] = beta_t
                    Volume[row, col] = volume
                    if show_updated_beta_constraints:
                        Q_star[row, col] = q_star
                else:
                    components = objective_components_np(
                        tuple(shape),
                        A=A,
                        n_quad=n_quad,
                        objective_name=objective_name,
                        volume_penalty=volume_penalty,
                        target_shape=target_shape,
                        R0=R0,
                        I=I,
                        P0=P0,
                        B0=B0,
                        q_min=q_min,
                        kappa_max=kappa_max,
                        q_penalty=q_penalty,
                        kappa_penalty=kappa_penalty,
                    )
                    Z[row, col] = components["base_objective"] if landscape_component == "raw" else components["objective"]
                    Volume[row, col] = components["volume"]
                    if show_updated_beta_constraints:
                        Q_star[row, col] = components["q_star"]
            except Exception:
                Z[row, col] = np.nan
                Volume[row, col] = np.nan
                Q_star[row, col] = np.nan

    q_values = Q_star[np.isfinite(Q_star)] if show_updated_beta_constraints else np.asarray([], dtype=float)
    q_masked = np.ma.masked_invalid(Q_star) if show_updated_beta_constraints else None
    q_star_min = np.nan
    q_star_max = np.nan
    q_contour_visible = False
    if q_values.size:
        q_star_min = float(np.min(q_values))
        q_star_max = float(np.max(q_values))
        q_contour_visible = q_star_min <= q_min <= q_star_max

    fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    finite_values = Z[np.isfinite(Z)]
    if finite_values.size:
        z_min = float(np.min(finite_values))
        z_max = float(np.max(finite_values))
        if z_min == z_max:
            span = max(abs(z_min), 1.0) * 1e-12
            levels = np.linspace(z_min - span, z_max + span, contour_count)
        else:
            levels = np.linspace(z_min, z_max, contour_count)

        filled = ax.contourf(X, Y, Z, levels=levels, cmap=LANDSCAPE_CMAP)
        if q_contour_visible and q_star_min < q_min:
            ax.contourf(
                X,
                Y,
                q_masked,
                levels=[q_star_min, q_min],
                colors="black",
                alpha=1.0 - NON_FEASIBLE_CMAP_FACTOR,
                antialiased=False,
                corner_mask=False,
            )
        ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.35, alpha=0.45)
        if colorbar:
            fig.colorbar(filled, ax=ax, label=landscape_label)
    else:
        ax.text(0.5, 0.5, "No finite objective values", transform=ax.transAxes, ha="center", va="center")

    legend_handles = []
    volume_values = Volume[np.isfinite(Volume)]
    if volume_values.size and np.isfinite(target_volume):
        volume_min = float(np.min(volume_values))
        volume_max = float(np.max(volume_values))
        volume_contour_visible = volume_min <= target_volume <= volume_max
        volume_label = r"$V=V_{\rm sep}$" if landscape_component == "optimal_new_beta_t" else r"$V=V_{\rm target}$"
        print(
            "Volume range on objective contour grid: "
            f"min={volume_min:.12g}, max={volume_max:.12g}, "
            f"target={float(target_volume):.12g}, "
            f"volume contour visible={'yes' if volume_contour_visible else 'no'}"
        )
        if volume_contour_visible:
            volume_contour = ax.contour(
                X,
                Y,
                Volume,
                levels=[target_volume],
                colors=VOLUME_CONSTRAINT_COLOR,
                linestyles="-",
                linewidths=2.0,
            )
            ax.clabel(
                volume_contour,
                fmt={target_volume: volume_label},
                fontsize=8,
            )
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=VOLUME_CONSTRAINT_COLOR,
                    linestyle="-",
                    linewidth=2.0,
                    label=volume_label,
                )
            )
    else:
        print(
            "Volume range on objective contour grid: "
            "no finite volume values or target volume unavailable, volume contour visible=no"
        )
    if show_updated_beta_constraints:
        if q_values.size:
            print(
                "q_star range on objective contour grid: "
                f"min={q_star_min:.12g}, max={q_star_max:.12g}, "
                f"q_min={float(q_min):.12g}, "
                f"q contour visible={'yes' if q_contour_visible else 'no'}"
            )
        else:
            print(
                "q_star range on objective contour grid: "
                "no finite q_star values, q contour visible=no"
            )
        if q_contour_visible:
            q_contour = ax.contour(
                X,
                Y,
                q_masked,
                levels=[q_min],
                colors=Q_CONSTRAINT_COLOR,
                linestyles="--",
                linewidths=2.0,
                corner_mask=False,
            )
            ax.clabel(q_contour, fmt={q_min: rf"$q_*={q_min:g}$"}, fontsize=8)
            legend_handles.append(
                plt.Line2D([0], [0], color=Q_CONSTRAINT_COLOR, linestyle="--", linewidth=2.0, label=rf"$q_*={q_min:g}$")
            )
        if normalized_objective_name == "beta_toroidal_updated":
            kappa_line_visible = False
            if 1 in free_indices and np.isfinite(kappa_max):
                kappa_line_visible = bounds[1][0] <= float(kappa_max) <= bounds[1][1]
                print(
                    "kappa constraint on objective contour grid: "
                    f"kappa_max={float(kappa_max):.12g}, "
                    f"visible={'yes' if kappa_line_visible else 'no'}"
                )
                if kappa_line_visible:
                    kappa_label = rf"$\kappa={float(kappa_max):g}$ max"
                    if x_index == 1:
                        ax.axvline(
                            float(kappa_max),
                            color=KAPPA_CONSTRAINT_COLOR,
                            linestyle=":",
                            linewidth=2.0,
                        )
                    else:
                        ax.axhline(
                            float(kappa_max),
                            color=KAPPA_CONSTRAINT_COLOR,
                            linestyle=":",
                            linewidth=2.0,
                        )
                    legend_handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=KAPPA_CONSTRAINT_COLOR,
                            linestyle=":",
                            linewidth=2.0,
                            label=kappa_label,
                        )
                    )
            else:
                print(
                    "kappa constraint on objective contour grid: "
                    "kappa is fixed in this slice or kappa_max is unavailable, visible=no"
                )

    shape_iters = np.asarray(result.get("shape_iters", []), dtype=float)
    call_scatter = None
    if shape_iters.size:
        calls = np.arange(1, len(shape_iters) + 1)
        call_scatter = ax.scatter(
            shape_iters[:, x_index],
            shape_iters[:, y_index],
            c=calls,
            cmap=CALL_CMAP,
            edgecolor="black",
            linewidth=0.35,
            s=34,
            label="BO evaluations",
            zorder=4,
        )
        ax.scatter(
            shape_iters[0, x_index],
            shape_iters[0, y_index],
            color=FIRST_CALL_COLOR,
            edgecolor="black",
            s=65,
            marker="s",
            label="first call",
            zorder=5,
        )

    final_shape = np.asarray(result["shape"], dtype=float)
    best_call = result.get("best_call")
    ax.scatter(
        final_shape[x_index],
        final_shape[y_index],
        color=FINAL_BEST_COLOR,
        edgecolor="black",
        s=95,
        marker="*",
        label=f"final best (call {best_call})" if best_call is not None else "final best",
        zorder=5,
    )
    if best_call is not None:
        ax.annotate(
            f"best call = {best_call}",
            (final_shape[x_index], final_shape[y_index]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="black",
            weight="bold",
        )
    if call_scatter is not None and call_colorbar:
        fig.colorbar(call_scatter, ax=ax, label="BO call")

    ax.set_xlabel(PARAM_LABELS[x_name])
    ax.set_ylabel(PARAM_LABELS[y_name])
    if not show_y_axis:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    ax.set_xlim(bounds[x_index][0], bounds[x_index][1])
    ax.set_ylim(bounds[y_index][0], bounds[y_index][1])
    if legend_handles:
        current_handles, current_labels = ax.get_legend_handles_labels()
        ax.legend(current_handles + legend_handles, current_labels + [handle.get_label() for handle in legend_handles], loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

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
            "constrained objective to maximize: normalized_psi_pressure, "
            "beta_toroidal_variable_q, beta_toroidal_updated, "
            "or beta_toroidal_updated_no_kappa_penalty"
        ),
    )
    parser.add_argument(
        "--machine",
        help=(
            "machine preset for beta_toroidal_updated R0/I/B0 values; "
            "supported: " + ", ".join(sorted(MACHINE_PRESETS))
        ),
    )
    parser.add_argument("--n-quad", type=int, default=2048)
    parser.add_argument("--volume-penalty", type=float, default=DEFAULT_VOLUME_PENALTY)
    parser.add_argument("--R0", type=float, help="major radius used by beta_toroidal_updated; overrides --machine")
    parser.add_argument("--I", type=float, help="plasma current in A used by beta_toroidal_updated; overrides --machine")
    parser.add_argument("--P0", type=float, default=DEFAULT_P0, help="pressure scaling P_0 used by beta_toroidal_updated")
    parser.add_argument("--B0", type=float, help="toroidal field used by beta_toroidal_updated; overrides --machine")
    parser.add_argument("--q-min", type=float, default=DEFAULT_Q_MIN, help="minimum q_star for beta_toroidal_updated")
    parser.add_argument("--kappa-max", type=float, default=DEFAULT_KAPPA_MAX, help="maximum kappa for beta_toroidal_updated")
    parser.add_argument("--q-penalty", type=float, default=DEFAULT_Q_PENALTY, help="penalty coefficient for max(0, q_min - q_star)^2")
    parser.add_argument("--kappa-penalty", type=float, default=DEFAULT_KAPPA_PENALTY, help="penalty coefficient for max(0, kappa - kappa_max)^2")
    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=float,
        default=DEFAULT_TARGET_SHAPE,
        metavar=("EPSILON", "KAPPA", "DELTA"),
        help="shape whose volume defines the fixed-volume target",
    )
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
    parser.add_argument("--plasma-no-colorbar", action="store_true", help="hide the colorbar on the optimized plasma contour plot")
    parser.add_argument("--plasma-hide-y-axis", action="store_true", help="hide the y-axis ticks and label on the optimized plasma contour plot")
    parser.add_argument("--objective-plot", type=Path, help="save fixed-parameter objective contour plot with all BO evaluation points to this path")
    parser.add_argument("--objective-plot-grid", type=int, default=35)
    parser.add_argument("--objective-contour-count", type=int, default=20)
    parser.add_argument("--objective-plot-n-quad", type=int, help="N used for the objective contour grid; defaults to --n-quad")
    parser.add_argument("--objective-no-colorbar", action="store_true", help="hide the objective landscape colorbar on the fixed-parameter contour plot")
    parser.add_argument("--objective-no-call-colorbar", action="store_true", help="hide the BO call colorbar on the fixed-parameter objective contour plot")
    parser.add_argument("--objective-hide-y-axis", action="store_true", help="hide the y-axis ticks and label on the fixed-parameter objective contour plot")
    parser.add_argument("--landscape-raw", action="store_true", help="for fixed-parameter landscape plots, show the unpenalized base objective while keeping optimization unchanged")
    parser.add_argument(
        "--landscape-component",
        help=(
            "fixed-parameter landscape background: penalized, raw, or "
            "optimal_new_beta_t/friend for the optimal_JAX beta_t + V_sep style"
        ),
    )
    parser.add_argument("--trajectory-every", type=int, default=5, help="legacy option kept for compatibility; all BO calls are plotted as points")
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
    target_shape = tuple(float(value) for value in args.target_shape)
    target_outside_bounds = [
        f"{name}={value:.12g} outside [{lo:.12g}, {hi:.12g}]"
        for name, value, (lo, hi) in zip(PARAM_NAMES, target_shape, bounds)
        if value < lo or value > hi
    ]
    if target_outside_bounds:
        print(
            "Warning: target volume shape is outside optimization bounds: "
            + "; ".join(target_outside_bounds)
        )
        print("         The optimizer can match only the target volume, not that exact target shape.")
    try:
        landscape_component = _normalize_landscape_component(
            args.landscape_component,
            landscape_raw=args.landscape_raw,
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        preset = machine_preset(args.machine)
    except ValueError as exc:
        parser.error(str(exc))
    R0 = float(args.R0 if args.R0 is not None else (preset["R0"] if preset else DEFAULT_R0))
    I = float(args.I if args.I is not None else (preset["I"] if preset else DEFAULT_I))
    P0 = float(args.P0)
    B0 = float(args.B0 if args.B0 is not None else (preset["B0"] if preset else DEFAULT_B0))
    machine_label = preset["label"] if preset else None
    fixed_count = sum(value is not None for value in fixed_values)
    if fixed_count > 1:
        parser.error("fix at most one of --fix-epsilon, --fix-kappa, or --fix-delta")
    if args.objective_plot and fixed_count != 1:
        parser.error("--objective-plot requires exactly one fixed parameter")

    optimization_start = time.perf_counter()
    result = optimize_shape_bayesian(
        bounds=bounds,
        A=args.A,
        fixed_values=fixed_values,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=args.random_state,
        n_quad=args.n_quad,
        objective_name=args.objective,
        volume_penalty=args.volume_penalty,
        target_shape=target_shape,
        R0=R0,
        I=I,
        P0=P0,
        B0=B0,
        q_min=args.q_min,
        kappa_max=args.kappa_max,
        q_penalty=args.q_penalty,
        kappa_penalty=args.kappa_penalty,
        surrogate=args.surrogate,
    )
    optimization_elapsed = time.perf_counter() - optimization_start
    diagnostics_elapsed = 0.0
    tests_elapsed = 0.0
    plot_elapsed = 0.0

    optimizer = result["optimizer"]
    show_flag = bool(args.show_plot)
    if getattr(args, "no_show_plot", False):
        show_flag = False

    if result["fixed_indices"]:
        optimized_names = ", ".join(PARAM_NAMES[i] for i in result["free_indices"])
        print(f"Fixed parameter: {_format_fixed_values(result['fixed_values'])}")
        print(f"Optimized parameters: {optimized_names}")
    print(f"Objective: {result['objective_label']}")
    print(f"Volume penalty: {float(result['volume_penalty']):.12g}")
    print(f"Target volume shape: {_format_shape(result['target_shape'])}")
    if _is_updated_beta_objective(result["objective_name"]):
        if machine_label is not None:
            print(f"Machine preset: {machine_label}")
        print(
            "Updated beta parameters: "
            f"R0={float(result['R0']):.12g}, "
            f"I={float(result['I']):.12g}, "
            f"P0={float(result['P0']):.12g}, "
            f"B0={float(result['B0']):.12g}"
        )
        print(
            "Inequality constraints: "
            f"q_star >= {float(result['q_min']):.12g}"
        )
        if result["objective_name"] == "beta_toroidal_updated":
            print(f"Kappa constraint: kappa <= {float(result['kappa_max']):.12g}")
            print(
                "Inequality penalties: "
                f"q_penalty={float(result['q_penalty']):.12g}, "
                f"kappa_penalty={float(result['kappa_penalty']):.12g}"
            )
        else:
            print("Kappa constraint: disabled")
            print(
                "Inequality penalties: "
                f"q_penalty={float(result['q_penalty']):.12g}, "
                "kappa_penalty=disabled"
            )
    print(f"Initial shape: {_format_shape(result['initial_shape'])}")
    print(f"Initial volume_penalty_objective: {float(result['initial_volume_penalty_objective']):.12g}")
    print(f"Final shape: {_format_shape(result['shape'])}")
    print(f"Final volume_penalty_objective: {float(result['volume_penalty_objective']):.12g}")
    print(f"Final base objective: {float(result['final_base_objective']):.12g}")
    print(f"Final raw objective (no penalties): {float(result['final_base_objective']):.12g}")
    print(f"Final volume_average_pressure: {float(result['final_volume_average_pressure']):.12g}")
    print(f"Final normalized_psi_pressure: {float(result['final_normalized_psi_pressure']):.12g}")
    print(f"Final P0_times_normalized_psi_pressure: {float(result['final_P0_times_normalized_psi_pressure']):.12g}")
    print(f"Total penalty term: {float(result['penalty_term']):.12g}")
    print(f"Volume penalty term: {float(result['volume_penalty_term']):.12g}")
    if _is_updated_beta_objective(result["objective_name"]):
        print(f"Final q_star: {float(result['q_star']):.12g}")
        print(f"q_star violation: {float(result['q_violation']):.12g}")
        print(f"q_star penalty term: {float(result['q_penalty_term']):.12g}")
        print(f"kappa violation: {float(result['kappa_violation']):.12g}")
        if result["objective_name"] == "beta_toroidal_updated":
            print(f"kappa penalty term: {float(result['kappa_penalty_term']):.12g}")
        else:
            print("kappa penalty term: disabled")
    print(f"Final volume: {float(result['final_volume']):.12g}")
    print(f"Target volume: {float(result['target_volume']):.12g}")
    print(f"Volume difference (final - target): {float(result['volume_error']):.12g}")
    print(f"Absolute volume difference: {float(result['absolute_volume_error']):.12g}")
    print(f"Relative volume error: {float(result['relative_volume_error']):.12g}")
    print(f"Relative volume error (%): {100.0 * float(result['relative_volume_error']):.12g}")
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
            volume_penalty=args.volume_penalty,
            target_shape=target_shape,
            R0=R0,
            I=I,
            P0=P0,
            B0=B0,
            q_min=args.q_min,
            kappa_max=args.kappa_max,
            q_penalty=args.q_penalty,
            kappa_penalty=args.kappa_penalty,
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
            volume_penalty=args.volume_penalty,
            target_shape=target_shape,
            R0=R0,
            I=I,
            P0=P0,
            B0=B0,
            q_min=args.q_min,
            kappa_max=args.kappa_max,
            q_penalty=args.q_penalty,
            kappa_penalty=args.kappa_penalty,
            perturbation_radius=args.test_perturbation_radius,
            perturbation_samples=args.test_perturbation_samples,
            perturbation_seed=args.test_seed,
        )
        tests_elapsed = time.perf_counter() - tests_start
    if result["fixed_indices"] and (args.objective_plot or show_flag):
        objective_plot_n_quad = args.objective_plot_n_quad or args.n_quad
        print("Computing fixed-parameter objective contour plot...")
        print(
            "Landscape component: "
            f"{LANDSCAPE_COMPONENT_LABELS[landscape_component]}"
        )
        plot_start = time.perf_counter()
        plot_fixed_parameter_objective_contours(
            result,
            A=args.A,
            bounds=bounds,
            fixed_values=result["fixed_values"],
            output_path=args.objective_plot,
            grid_size=args.objective_plot_grid,
            contour_count=args.objective_contour_count,
            trajectory_every=args.trajectory_every,
            n_quad=objective_plot_n_quad,
            objective_name=args.objective,
            volume_penalty=args.volume_penalty,
            target_shape=target_shape,
            R0=R0,
            I=I,
            P0=P0,
            B0=B0,
            q_min=args.q_min,
            kappa_max=args.kappa_max,
            q_penalty=args.q_penalty,
            kappa_penalty=args.kappa_penalty,
            landscape_raw=args.landscape_raw,
            landscape_component=landscape_component,
            colorbar=not args.objective_no_colorbar,
            call_colorbar=not args.objective_no_call_colorbar,
            show_y_axis=not args.objective_hide_y_axis,
            show=show_flag,
        )
        plot_elapsed += time.perf_counter() - plot_start
        if args.objective_plot:
            print(f"Saved objective contour plot: {args.objective_plot}")
    if args.plot or show_flag:
        plot_start = time.perf_counter()
        plot_flux_contours(
            result["shape"],
            coefficients=result["coefficients"],
            A=args.A,
            output_path=args.plot,
            grid_size=args.plot_grid,
            contour_count=args.contour_count,
            colorbar=not args.plasma_no_colorbar,
            show_y_axis=not args.plasma_hide_y_axis,
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
    print(f"  Total elapsed: {total_elapsed:.3f} s")


if __name__ == "__main__":
    main()
