"""Optimize a symmetric Cerfon-Freidberg Solov'ev plasma shape.

The script searches for the three shape parameters that give the largest
selected pressure objective:

* epsilon: horizontal size of the plasma cross-section
* kappa: vertical elongation
* delta: triangularity, or how strongly the top point leans inward

The workflow is:

1. Build the Solov'ev flux function for a requested shape.
2. Ask the pressure_integral package for the selected pressure objective.
3. Let JAX gradient ascent adjust the shape and maximize that pressure.
4. Print, test, plot, or save the optimized result.
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import warnings
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from scipy.optimize import OptimizeResult

REPO_ROOT = Path(__file__).resolve().parents[1]
for source_dir in (
    REPO_ROOT,
    REPO_ROOT / "pressure_integral",
    REPO_ROOT / "ITER_Equilibria",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pressure_integral.pressure_utils import (
    _normalized_psi_pressure,
    get_vol_av_p_from_params,
    make_psi,
)


DEFAULT_A = -0.05
SHAPE_PARAMETER_NAMES = ("epsilon", "kappa", "delta")

# A shape is always stored in this order: epsilon, kappa, delta.
DEFAULT_INITIAL_SHAPE = (0.32, 1.30, 0.20)
DOMAIN_EPS = 1e-8
DOMAIN_BOUNDS = (
    (DOMAIN_EPS, 1),
    (DOMAIN_EPS, float("inf")),
    (-0.3, 0.3),
)
RANDOM_INITIAL_RANGES = ((0.1, 0.45), (1.0, 1.7), (-0.3, 0.3))
DEFAULT_RANDOM_SUMMARY_CSV = Path("optimal_JAX/output/random_initials_summary.csv")
DEFAULT_RANDOM_MAP_PLOT = Path("optimal_JAX/output/random_initials_shape_map.png")
DEFAULT_RANDOM_HISTORY_PLOT = Path(
    "optimal_JAX/output/best_random_averaged_volume_pressure.png"
)
DEFAULT_RANDOM_FLUX_PLOT = Path("optimal_JAX/output/best_random_flux_contours.png")
PRESSURE_GRAD_STEP = 1e-5
PRESSURE_DOMAIN_EDGE_TOL = 1e-4
OBJECTIVE_AVERAGED_VOLUME_PRESSURE = "averaged-volume-pressure"
OBJECTIVE_NORMALIZED_PSI_PRESSURE = "normalized-psi-pressure"
PRESSURE_OBJECTIVE_CHOICES = (
    OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    OBJECTIVE_NORMALIZED_PSI_PRESSURE,
)
PRESSURE_OBJECTIVE_LABELS = {
    OBJECTIVE_AVERAGED_VOLUME_PRESSURE: "averaged volume pressure",
    OBJECTIVE_NORMALIZED_PSI_PRESSURE: "normalized psi pressure",
}
BOUNDED_SEARCH_INFORMATIONAL_CHECKS = frozenset(("perturbation", "interior_gradient"))


# ---------------------------------------------------------------------------
# Pressure evaluation through the existing pressure_integral package
# ---------------------------------------------------------------------------
#
# pressure_integral is ordinary NumPy/SciPy code, not native JAX code. The
# custom JVP below tells JAX how to differentiate through that pressure value:
# the value comes from pressure_integral, and the derivative is a centered
# finite difference in epsilon, kappa, and delta.


def _pressure_value_from_shape_host(shape, A, n_quad):
    """Evaluate averaged volume pressure on the host side, outside JAX tracing."""
    n_quad = int(n_quad)
    if n_quad <= 0:
        raise ValueError("averaged pressure grid size must be positive")
    try:
        epsilon, kappa, delta = _validate_shape_domain(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = get_vol_av_p_from_params(
                epsilon,
                kappa,
                delta,
                A=float(A),
                method="contour",
                N=n_quad,
            )
            if not np.isfinite(value):
                value = np.nan
    except (ArithmeticError, ValueError, RuntimeWarning, np.linalg.LinAlgError):
        value = np.nan
    return np.asarray(value, dtype=np.float64)


def _pressure_gradient_from_shape_host(shape, A, n_quad, step=PRESSURE_GRAD_STEP):
    """Finite-difference d<averaged volume pressure>/d(epsilon, kappa, delta)."""
    shape = np.asarray(_validate_shape_domain(shape), dtype=float)
    domain_bounds = np.asarray(DOMAIN_BOUNDS, dtype=float)
    gradient = np.empty(3, dtype=np.float64)
    step = float(step)
    if step <= 0:
        raise ValueError("finite-difference step must be positive")

    center_value = float(_pressure_value_from_shape_host(shape, A, n_quad))
    for index in range(3):
        axis_step = step
        lower, upper = domain_bounds[index]
        if np.isfinite(lower):
            axis_step = min(axis_step, 0.5 * (shape[index] - lower))
        if np.isfinite(upper):
            axis_step = min(axis_step, 0.5 * (upper - shape[index]))
        if axis_step <= 0:
            raise ValueError("finite-difference step does not fit inside shape domain")

        plus = shape.copy()
        minus = shape.copy()
        plus[index] += axis_step
        minus[index] -= axis_step
        plus_value = float(_pressure_value_from_shape_host(plus, A, n_quad))
        minus_value = float(_pressure_value_from_shape_host(minus, A, n_quad))
        if np.isfinite(plus_value) and np.isfinite(minus_value):
            gradient[index] = (plus_value - minus_value) / (2.0 * axis_step)
        elif np.isfinite(plus_value) and np.isfinite(center_value):
            gradient[index] = (plus_value - center_value) / axis_step
        elif np.isfinite(minus_value) and np.isfinite(center_value):
            gradient[index] = (center_value - minus_value) / axis_step
        else:
            gradient[index] = np.nan

    return gradient


def _normalized_pressure_value_from_shape_host(shape, A, method, grid_size):
    """Evaluate normalized psi pressure on the host side, outside JAX tracing."""
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("normalized pressure grid size must be positive")
    if method not in ("contour", "masking"):
        raise ValueError("normalized pressure method must be 'contour' or 'masking'")
    try:
        epsilon, kappa, delta = _validate_shape_domain(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = _normalized_psi_pressure(
                epsilon,
                kappa,
                delta,
                A=float(A),
                method=method,
                N=grid_size,
            )
    except (ArithmeticError, ValueError, RuntimeWarning, np.linalg.LinAlgError):
        value = np.nan
    return np.asarray(value, dtype=np.float64)


def _normalized_pressure_gradient_from_shape_host(
    shape,
    A,
    method,
    grid_size,
    step=PRESSURE_GRAD_STEP,
):
    """Finite-difference d<normalized psi pressure>/d(epsilon, kappa, delta)."""
    shape = np.asarray(_validate_shape_domain(shape), dtype=float)
    gradient = np.empty(3, dtype=np.float64)
    step = float(step)
    if step <= 0:
        raise ValueError("finite-difference step must be positive")

    center_value = float(
        _normalized_pressure_value_from_shape_host(shape, A, method, grid_size)
    )
    for index in range(3):
        axis_step = step
        lower, upper = DOMAIN_BOUNDS[index]
        if np.isfinite(lower):
            axis_step = min(axis_step, 0.5 * (shape[index] - lower))
        if np.isfinite(upper):
            axis_step = min(axis_step, 0.5 * (upper - shape[index]))
        if axis_step <= 0:
            raise ValueError("finite-difference step does not fit inside shape domain")

        plus = shape.copy()
        minus = shape.copy()
        plus[index] += axis_step
        minus[index] -= axis_step
        plus_value = float(
            _normalized_pressure_value_from_shape_host(plus, A, method, grid_size)
        )
        minus_value = float(
            _normalized_pressure_value_from_shape_host(minus, A, method, grid_size)
        )
        if np.isfinite(plus_value) and np.isfinite(minus_value):
            gradient[index] = (plus_value - minus_value) / (2.0 * axis_step)
        elif np.isfinite(plus_value) and np.isfinite(center_value):
            gradient[index] = (plus_value - center_value) / axis_step
        elif np.isfinite(minus_value) and np.isfinite(center_value):
            gradient[index] = (center_value - minus_value) / axis_step
        else:
            gradient[index] = np.nan

    return gradient


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _volume_averaged_pressure_from_shape(shape, A, n_quad):
    """JAX-visible wrapper around the host averaged volume pressure calculation."""
    return jax.pure_callback(
        lambda host_shape: _pressure_value_from_shape_host(host_shape, A, n_quad),
        jax.ShapeDtypeStruct((), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )


@_volume_averaged_pressure_from_shape.defjvp
def _volume_averaged_pressure_from_shape_jvp(A, n_quad, primals, tangents):
    (shape,) = primals
    (shape_dot,) = tangents
    value = _volume_averaged_pressure_from_shape(shape, A, n_quad)
    gradient = jax.pure_callback(
        lambda host_shape: _pressure_gradient_from_shape_host(host_shape, A, n_quad),
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )
    return value, jnp.dot(gradient, shape_dot)


def volume_averaged_pressure(epsilon, kappa, delta, A=DEFAULT_A, n_quad=2048):
    """Return volume-averaged pressure for one physical shape."""
    shape = jnp.stack([epsilon, kappa, delta])
    return _volume_averaged_pressure_from_shape(shape, float(A), int(n_quad))


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _normalized_psi_pressure_from_shape(shape, A, method, grid_size):
    """JAX-visible wrapper around the host normalized psi pressure calculation."""
    return jax.pure_callback(
        lambda host_shape: _normalized_pressure_value_from_shape_host(
            host_shape,
            A,
            method,
            grid_size,
        ),
        jax.ShapeDtypeStruct((), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )


@_normalized_psi_pressure_from_shape.defjvp
def _normalized_psi_pressure_from_shape_jvp(A, method, grid_size, primals, tangents):
    (shape,) = primals
    (shape_dot,) = tangents
    value = _normalized_psi_pressure_from_shape(shape, A, method, grid_size)
    gradient = jax.pure_callback(
        lambda host_shape: _normalized_pressure_gradient_from_shape_host(
            host_shape,
            A,
            method,
            grid_size,
        ),
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )
    return value, jnp.dot(gradient, shape_dot)


def normalized_psi_pressure(
    epsilon,
    kappa,
    delta,
    A=DEFAULT_A,
    method="contour",
    grid_size=500,
):
    """Return normalized psi pressure for one physical shape."""
    shape = jnp.stack([epsilon, kappa, delta])
    return _normalized_psi_pressure_from_shape(
        shape,
        float(A),
        str(method),
        int(grid_size),
    )


def pressure_objective_value(
    shape,
    A=DEFAULT_A,
    n_quad=2048,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
):
    """Evaluate the selected pressure objective for one physical shape."""
    if pressure_objective == OBJECTIVE_AVERAGED_VOLUME_PRESSURE:
        return volume_averaged_pressure(*shape, A=A, n_quad=n_quad)
    if pressure_objective == OBJECTIVE_NORMALIZED_PSI_PRESSURE:
        return normalized_psi_pressure(
            *shape,
            A=A,
            method=normalized_method,
            grid_size=normalized_grid_size,
        )
    raise ValueError(f"unknown pressure objective: {pressure_objective}")


def _make_psi_for_shape(shape, A=DEFAULT_A):
    """Build psi and coefficients with pressure_integral's matrix solve."""
    epsilon, kappa, delta = _validate_shape_domain(shape)
    psi, coefficients, _ = make_psi(epsilon, kappa, delta, float(A))
    return psi, np.asarray(coefficients, dtype=np.float64)


def _try_make_coefficients_for_shape(shape, A=DEFAULT_A):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _, coefficients = _make_psi_for_shape(shape, A=A)
    except (ArithmeticError, ValueError, RuntimeWarning, np.linalg.LinAlgError):
        return np.full(7, np.nan, dtype=np.float64)
    return coefficients


# ---------------------------------------------------------------------------
# Shape validation and optimizer coordinate transforms
# ---------------------------------------------------------------------------
#
# The optimizer works in unconstrained numbers. These helpers
# convert those raw numbers into physical epsilon/kappa/delta values before
# pressure is evaluated.


def domain_from_unconstrained(raw):
    """Map unrestricted numbers into epsilon > 0, kappa > 0, free delta."""
    return jnp.stack(
        [
            jnp.exp(raw[0]),
            jnp.exp(raw[1]),
            raw[2],
        ]
    )


def _effective_shape_bounds(bounds, use_bounds=False):
    if bounds is None:
        if not use_bounds:
            return None
        bounds = DOMAIN_BOUNDS

    bounds_array = np.asarray(bounds, dtype=float)
    if bounds_array.shape != (3, 2):
        raise ValueError("bounds must be a sequence of three (min, max) pairs")
    for index, name in enumerate(SHAPE_PARAMETER_NAMES):
        lower, upper = bounds_array[index]
        if not np.isfinite(lower) and not np.isneginf(lower):
            raise ValueError(f"{name} lower bound must be finite or -inf")
        if not np.isfinite(upper) and not np.isposinf(upper):
            raise ValueError(f"{name} upper bound must be finite or inf")
        if lower >= upper:
            raise ValueError(f"{name} lower bound must be less than upper bound")
    return bounds_array


def _validate_shape_bounds(shape, bounds):
    if bounds is None:
        return
    shape_array = np.asarray(shape, dtype=float)
    bounds_array = np.asarray(bounds, dtype=float)
    for index, name in enumerate(SHAPE_PARAMETER_NAMES):
        lower, upper = bounds_array[index]
        if shape_array[index] < lower or shape_array[index] > upper:
            raise ValueError(
                f"initial {name}={shape_array[index]:.12g} is outside "
                f"[{lower:.12g}, {upper:.12g}]"
            )


def domain_from_bounded(raw, bounds):
    """Map unrestricted numbers into the supplied shape search box."""
    raw = jnp.asarray(raw, dtype=jnp.float64)
    bounds = np.asarray(bounds, dtype=float)
    values = []
    for index in range(3):
        lower = bounds[index, 0]
        upper = bounds[index, 1]
        if np.isfinite(lower) and np.isfinite(upper):
            values.append(lower + (upper - lower) * jax.nn.sigmoid(raw[index]))
        elif np.isfinite(lower):
            values.append(lower + jnp.exp(raw[index]))
        elif np.isfinite(upper):
            values.append(upper - jnp.exp(raw[index]))
        else:
            values.append(raw[index])
    return jnp.stack(values)


def unconstrained_from_bounded(shape, bounds):
    """Inverse of domain_from_bounded for a shape inside the search box."""
    shape = jnp.asarray(_validate_shape_domain(shape), dtype=jnp.float64)
    bounds = np.asarray(bounds, dtype=float)
    raw = []
    for index in range(3):
        value = shape[index]
        lower = bounds[index, 0]
        upper = bounds[index, 1]
        if np.isfinite(lower) and np.isfinite(upper):
            fraction = (value - lower) / (upper - lower)
            fraction = jnp.clip(fraction, 1e-12, 1.0 - 1e-12)
            raw.append(jnp.log(fraction) - jnp.log1p(-fraction))
        elif np.isfinite(lower):
            raw.append(jnp.log(jnp.maximum(value - lower, 1e-12)))
        elif np.isfinite(upper):
            raw.append(jnp.log(jnp.maximum(upper - value, 1e-12)))
        else:
            raw.append(value)
    return jnp.stack(raw)


def unconstrained_from_domain(shape):
    """Inverse of domain_from_unconstrained for a valid physical shape."""
    epsilon, kappa, delta = jnp.asarray(_validate_shape_domain(shape))
    return jnp.stack(
        [
            jnp.log(epsilon),
            jnp.log(kappa),
            delta,
        ]
    )


def pressure_from_raw(
    raw,
    A=DEFAULT_A,
    bounds=None,
    n_quad=2048,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
    use_bounds=False,
):
    """Convert optimizer variables to a physical shape, then evaluate pressure."""
    optimizer_bounds = _effective_shape_bounds(bounds, use_bounds=use_bounds)
    if optimizer_bounds is None:
        shape = domain_from_unconstrained(raw)
    else:
        shape = domain_from_bounded(raw, optimizer_bounds)
    return pressure_objective_value(
        shape,
        A=A,
        n_quad=n_quad,
        pressure_objective=pressure_objective,
        normalized_method=normalized_method,
        normalized_grid_size=normalized_grid_size,
    )


def negative_pressure_from_raw(raw, A=DEFAULT_A, bounds=None, n_quad=2048, **kwargs):
    return -pressure_from_raw(raw, A=A, bounds=bounds, n_quad=n_quad, **kwargs)


def _validate_shape_domain(shape):
    epsilon, kappa, delta = (float(v) for v in shape)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    return (epsilon, kappa, delta)


# ---------------------------------------------------------------------------
# Optimization runs
# ---------------------------------------------------------------------------


def _prepare_optimizer_inputs(
    initial_shape,
    A,
    bounds,
    n_quad,
    use_bounds,
    pressure_objective,
    normalized_method,
    normalized_grid_size,
):
    """Validate inputs and build the functions the optimizer needs.

    The optimizer sees only raw, unrestricted numbers. shape_from_raw converts those raw
    numbers back to epsilon/kappa/delta for reporting, plotting, and coefficient
    solves.
    """
    if pressure_objective not in PRESSURE_OBJECTIVE_CHOICES:
        raise ValueError(f"unknown pressure objective: {pressure_objective}")
    if normalized_method not in ("contour", "masking"):
        raise ValueError("normalized_method must be 'contour' or 'masking'")
    if int(normalized_grid_size) <= 0:
        raise ValueError("normalized_grid_size must be positive")
    checked_initial_shape = _validate_shape_domain(initial_shape)
    checked_bounds = _effective_shape_bounds(bounds, use_bounds=use_bounds)
    _validate_shape_bounds(checked_initial_shape, checked_bounds)
    if checked_bounds is None:
        raw_initial = unconstrained_from_domain(checked_initial_shape)
        shape_from_raw = domain_from_unconstrained
    else:
        raw_initial = unconstrained_from_bounded(checked_initial_shape, checked_bounds)

        def shape_from_raw(raw):
            return domain_from_bounded(raw, checked_bounds)

    def loss_to_minimize(raw):
        shape = shape_from_raw(raw)
        return -pressure_objective_value(
            shape,
            A=A,
            n_quad=n_quad,
            pressure_objective=pressure_objective,
            normalized_method=normalized_method,
            normalized_grid_size=normalized_grid_size,
        )

    return (
        checked_initial_shape,
        checked_bounds,
        raw_initial,
        loss_to_minimize,
        shape_from_raw,
    )


def optimize_shape(
    initial_shape=DEFAULT_INITIAL_SHAPE,
    A=DEFAULT_A,
    bounds=None,
    n_quad=2048,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
    maxiter=200,
    gtol=1e-6,
    use_bounds=False,
):
    """Maximize the selected pressure objective over epsilon, kappa, delta."""
    initial_shape, bounds, raw0, objective, shape_from_raw = _prepare_optimizer_inputs(
        initial_shape,
        A,
        bounds,
        n_quad,
        use_bounds,
        pressure_objective,
        normalized_method,
        normalized_grid_size,
    )

    del objective
    maxiter = 200 if maxiter is None else int(maxiter)
    value_and_grad = jax.value_and_grad(
        lambda raw: pressure_objective_value(
            shape_from_raw(raw),
            A=A,
            n_quad=n_quad,
            pressure_objective=pressure_objective,
            normalized_method=normalized_method,
            normalized_grid_size=normalized_grid_size,
        )
    )
    history = []

    def append_history(raw):
        shape = shape_from_raw(jnp.asarray(raw, dtype=jnp.float64))
        objective_pressure = pressure_objective_value(
            shape,
            A=A,
            n_quad=n_quad,
            pressure_objective=pressure_objective,
            normalized_method=normalized_method,
            normalized_grid_size=normalized_grid_size,
        )
        entry = {
            "iteration": len(history),
            "shape": np.asarray(shape, dtype=float),
            "objective_pressure": float(objective_pressure),
        }
        if pressure_objective == OBJECTIVE_AVERAGED_VOLUME_PRESSURE:
            entry["averaged_volume_pressure"] = float(objective_pressure)
        history.append(entry)

    def pressure_and_gradient(raw):
        try:
            value, gradient = value_and_grad(jnp.asarray(raw, dtype=jnp.float64))
        except Exception:
            return np.nan, jnp.full_like(raw, np.nan)
        return value, gradient

    def pressure_value(raw):
        try:
            value = pressure_objective_value(
                shape_from_raw(jnp.asarray(raw, dtype=jnp.float64)),
                A=A,
                n_quad=n_quad,
                pressure_objective=pressure_objective,
                normalized_method=normalized_method,
                normalized_grid_size=normalized_grid_size,
            )
        except Exception:
            return np.nan
        return float(value)

    search_masks = (
        jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float64),
        jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float64),
        jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float64),
        jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float64),
    )

    def find_improving_step(raw, gradient, current_value):
        best_raw = None
        best_value = float(current_value)
        for mask in search_masks:
            direction = gradient * mask
            direction_norm = float(jnp.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm == 0.0:
                continue

            step = 1.0
            for _line_search_step in range(16):
                candidate_raw = raw + step * direction
                candidate_value = pressure_value(candidate_raw)
                if np.isfinite(candidate_value) and candidate_value > best_value:
                    best_raw = candidate_raw
                    best_value = candidate_value
                    break
                step *= 0.5

        return best_raw, best_value

    raw = jnp.asarray(raw0, dtype=jnp.float64)
    append_history(raw)
    success = False
    status = 1
    message = "maximum iterations reached"
    accepted_iterations = 0
    final_value, final_grad = pressure_and_gradient(raw)

    for _ in range(maxiter):
        current_value, gradient = pressure_and_gradient(raw)
        gradient_norm = float(jnp.linalg.norm(gradient))
        if not np.isfinite(float(current_value)) or not np.isfinite(gradient_norm):
            status = 2
            message = "non-finite objective or gradient"
            final_value, final_grad = current_value, gradient
            break
        if gradient_norm <= gtol:
            success = True
            status = 0
            message = "gradient tolerance reached"
            final_value, final_grad = current_value, gradient
            break

        candidate_raw, candidate_value = find_improving_step(raw, gradient, current_value)
        if candidate_raw is None:
            status = 2
            message = "no improving finite step found"
            final_value, final_grad = current_value, gradient
            break

        raw = candidate_raw
        append_history(raw)
        accepted_iterations += 1
        final_value, final_grad = pressure_and_gradient(raw)

    else:
        final_value, final_grad = pressure_and_gradient(raw)

    final_raw = jnp.asarray(raw, dtype=jnp.float64)
    final_shape = shape_from_raw(final_raw)
    final_coefficients = _try_make_coefficients_for_shape(final_shape, A=A)
    initial_pressure = pressure_objective_value(
        jnp.asarray(initial_shape, dtype=jnp.float64),
        A=A,
        n_quad=n_quad,
        pressure_objective=pressure_objective,
        normalized_method=normalized_method,
        normalized_grid_size=normalized_grid_size,
    )
    final_pressure = pressure_objective_value(
        final_shape,
        A=A,
        n_quad=n_quad,
        pressure_objective=pressure_objective,
        normalized_method=normalized_method,
        normalized_grid_size=normalized_grid_size,
    )
    initial_averaged_volume_pressure = volume_averaged_pressure(
        *initial_shape,
        A=A,
        n_quad=n_quad,
    )
    final_averaged_volume_pressure = volume_averaged_pressure(
        *final_shape,
        A=A,
        n_quad=n_quad,
    )
    result = OptimizeResult(
        x=np.asarray(final_raw, dtype=float),
        fun=-float(final_pressure),
        jac=-np.asarray(final_grad, dtype=float),
        nit=accepted_iterations,
        success=success,
        status=status,
        message=message,
    )

    return {
        "initial_shape": jnp.asarray(initial_shape),
        "initial_pressure": initial_pressure,
        "initial_objective_pressure": initial_pressure,
        "initial_averaged_volume_pressure": initial_averaged_volume_pressure,
        "raw_initial": raw0,
        "shape": final_shape,
        "pressure": final_pressure,
        "objective_pressure": final_pressure,
        "averaged_volume_pressure": final_averaged_volume_pressure,
        "coefficients": final_coefficients,
        "raw": final_raw,
        "gradient": final_grad,
        "gradient_norm": jnp.linalg.norm(final_grad),
        "objective_value": final_value,
        "pressure_objective": pressure_objective,
        "pressure_objective_label": PRESSURE_OBJECTIVE_LABELS[pressure_objective],
        "normalized_method": normalized_method,
        "normalized_grid_size": normalized_grid_size,
        "history": history,
        "optimizer": result,
        "bounds": bounds,
        "use_bounds": bounds is not None,
    }


def single_run(
    initial_shape,
    A=DEFAULT_A,
    bounds=None,
    n_quad=2048,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
    maxiter=200,
    gtol=1e-6,
    use_bounds=False,
):
    """Run one optimization from a single initial shape."""
    return optimize_shape(
        initial_shape=initial_shape,
        A=A,
        bounds=bounds,
        n_quad=n_quad,
        pressure_objective=pressure_objective,
        normalized_method=normalized_method,
        normalized_grid_size=normalized_grid_size,
        maxiter=maxiter,
        gtol=gtol,
        use_bounds=use_bounds,
    )


def random_initial_shapes(count=10, seed=0, ranges=RANDOM_INITIAL_RANGES):
    """Generate random initial shapes in epsilon, kappa, delta ranges."""
    ranges = jnp.asarray(ranges)
    low = ranges[:, 0]
    high = ranges[:, 1]
    key = jax.random.PRNGKey(int(seed))
    samples = low + (high - low) * jax.random.uniform(
        key,
        (int(count), 3),
        dtype=low.dtype,
    )
    return [tuple(float(value) for value in sample) for sample in samples]


def _random_ranges_for_bounds(ranges, bounds, use_bounds=False):
    effective_bounds = _effective_shape_bounds(bounds, use_bounds=use_bounds)
    if effective_bounds is None:
        return ranges

    ranges_array = np.asarray(ranges, dtype=float)
    clipped_ranges = np.column_stack(
        [
            np.maximum(ranges_array[:, 0], effective_bounds[:, 0]),
            np.minimum(ranges_array[:, 1], effective_bounds[:, 1]),
        ]
    )
    for index, name in enumerate(SHAPE_PARAMETER_NAMES):
        if clipped_ranges[index, 0] >= clipped_ranges[index, 1]:
            raise ValueError(
                f"random initial {name} range has no overlap with search box"
            )
    return clipped_ranges


def multiple_random_runs(
    count=10,
    seed=0,
    ranges=RANDOM_INITIAL_RANGES,
    A=DEFAULT_A,
    bounds=None,
    n_quad=2048,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
    maxiter=200,
    gtol=1e-6,
    use_bounds=False,
):
    """Run optimizations from random initial shapes."""
    results = []
    random_ranges = _random_ranges_for_bounds(ranges, bounds, use_bounds=use_bounds)
    for initial_shape in random_initial_shapes(
        count=count,
        seed=seed,
        ranges=random_ranges,
    ):
        results.append(
            single_run(
                initial_shape=initial_shape,
                A=A,
                bounds=bounds,
                n_quad=n_quad,
                pressure_objective=pressure_objective,
                normalized_method=normalized_method,
                normalized_grid_size=normalized_grid_size,
                maxiter=maxiter,
                gtol=gtol,
                use_bounds=use_bounds,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Local optimality checks
# ---------------------------------------------------------------------------


def _pressure_objective_for_unit_tests(
    A,
    bounds=None,
    n_quad=2048,
    use_bounds=False,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
):
    """Return domain limits and a pressure function in optimizer coordinates."""
    optimizer_bounds = _effective_shape_bounds(bounds, use_bounds=use_bounds)
    shape_from_raw = domain_from_unconstrained
    if optimizer_bounds is not None:
        shape_from_raw = lambda raw: domain_from_bounded(raw, optimizer_bounds)

    def pressure_objective_fn(raw):
        shape = shape_from_raw(raw)
        return pressure_objective_value(
            shape,
            A=A,
            n_quad=n_quad,
            pressure_objective=pressure_objective,
            normalized_method=normalized_method,
            normalized_grid_size=normalized_grid_size,
        )

    test_bounds = optimizer_bounds
    if test_bounds is None:
        test_bounds = jnp.asarray(DOMAIN_BOUNDS, dtype=jnp.float64)
    return test_bounds, pressure_objective_fn


def _active_shape_bounds(shape, low, high, active_bound_tol):
    """Find active domain limits so perturbations stay in allowed directions."""
    lower_active = shape <= low + active_bound_tol
    upper_active = shape >= high - active_bound_tol
    return lower_active, upper_active


def _finite_pressure_for_shape(
    shape,
    A,
    n_quad,
    pressure_objective=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    normalized_method="contour",
    normalized_grid_size=500,
):
    try:
        value = pressure_objective_value(
            jnp.asarray(shape, dtype=jnp.float64),
            A=A,
            n_quad=n_quad,
            pressure_objective=pressure_objective,
            normalized_method=normalized_method,
            normalized_grid_size=normalized_grid_size,
        )
    except Exception:
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


def _max_pressure_near_shape(
    shape,
    low,
    high,
    lower_active,
    upper_active,
    A,
    n_quad,
    pressure_objective,
    normalized_method,
    normalized_grid_size,
    perturbation_radius,
    perturbation_samples,
    perturbation_seed,
):
    """Sample nearby shapes and return the largest selected objective found."""
    key = jax.random.PRNGKey(int(perturbation_seed))
    directions = jax.random.normal(
        key,
        (int(perturbation_samples), 3),
        dtype=shape.dtype,
    )
    directions = jnp.where(upper_active, -jnp.abs(directions), directions)
    directions = jnp.where(lower_active, jnp.abs(directions), directions)
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    perturbations = float(perturbation_radius) * directions
    nearby_shapes = jnp.clip(shape + perturbations, low, high)
    nearby_pressures = np.asarray(
        [
            _finite_pressure_for_shape(
                point,
                A=A,
                n_quad=n_quad,
                pressure_objective=pressure_objective,
                normalized_method=normalized_method,
                normalized_grid_size=normalized_grid_size,
            )
            for point in nearby_shapes
        ],
        dtype=float,
    )
    finite_pressures = nearby_pressures[np.isfinite(nearby_pressures)]
    if finite_pressures.size == 0:
        return np.nan
    return float(np.max(finite_pressures))


def _one_parameter_finite_difference(function, value, low, high, step, name):
    """Use a centered difference when possible, otherwise one-sided."""
    center = float(function(value))
    plus = np.nan
    minus = np.nan
    if float(value) + step < float(high):
        plus = float(function(value + step))
    if float(value) - step > float(low):
        minus = float(function(value - step))

    if np.isfinite(plus) and np.isfinite(minus):
        return (plus - minus) / (2.0 * step)
    if np.isfinite(plus) and np.isfinite(center):
        return (plus - center) / step
    if np.isfinite(minus) and np.isfinite(center):
        return (center - minus) / step
    raise ValueError(f"finite-difference {name} step does not fit inside limits")


def _compare_shape_derivatives_to_finite_difference(
    shape,
    low,
    high,
    A,
    n_quad,
    pressure_objective,
    normalized_method,
    normalized_grid_size,
    step,
):
    """Compare JAX derivatives against direct finite differences."""
    jax_derivatives = {}
    finite_difference_derivatives = {}
    finite_difference_errors = {}
    step = float(step)

    for index, name in enumerate(SHAPE_PARAMETER_NAMES):
        parameter_value = shape[index]

        def parameter_objective(value, index=index):
            candidate = shape.at[index].set(value)
            return pressure_objective_value(
                candidate,
                A=A,
                n_quad=n_quad,
                pressure_objective=pressure_objective,
                normalized_method=normalized_method,
                normalized_grid_size=normalized_grid_size,
            )

        jax_derivative = float(jax.grad(parameter_objective)(parameter_value))
        finite_difference_derivative = _one_parameter_finite_difference(
            parameter_objective,
            parameter_value,
            low[index],
            high[index],
            step,
            name,
        )

        jax_derivatives[name] = jax_derivative
        finite_difference_derivatives[name] = finite_difference_derivative
        finite_difference_errors[name] = abs(jax_derivative - finite_difference_derivative)

    return jax_derivatives, finite_difference_derivatives, finite_difference_errors


def _centered_shape_steps_are_finite(
    shape,
    low,
    high,
    A,
    n_quad,
    pressure_objective,
    normalized_method,
    normalized_grid_size,
    step,
):
    shape = np.asarray(shape, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    step = float(step)
    edge_tol = max(PRESSURE_DOMAIN_EDGE_TOL, 10.0 * step)
    if shape[0] >= 1.0 - edge_tol:
        return False
    if abs(shape[2]) >= 1.0 - edge_tol:
        return False
    for index in range(3):
        if shape[index] + step >= high[index]:
            return False
        if shape[index] - step <= low[index]:
            return False
        plus = shape.copy()
        minus = shape.copy()
        plus[index] += step
        minus[index] -= step
        if not np.isfinite(
            _finite_pressure_for_shape(
                plus,
                A=A,
                n_quad=n_quad,
                pressure_objective=pressure_objective,
                normalized_method=normalized_method,
                normalized_grid_size=normalized_grid_size,
            )
        ):
            return False
        if not np.isfinite(
            _finite_pressure_for_shape(
                minus,
                A=A,
                n_quad=n_quad,
                pressure_objective=pressure_objective,
                normalized_method=normalized_method,
                normalized_grid_size=normalized_grid_size,
            )
        ):
            return False
    return True


def _print_unit_test_report(
    objective_label,
    gradient_norm,
    gradient_tol,
    gradient_check_required,
    max_perturbed_pressure,
    pressure,
    jax_derivatives,
    finite_difference_derivatives,
    finite_difference_errors,
    finite_difference_tol,
):
    """Print the local optimality checks in a human-readable order."""
    print("Unit tests:")
    for name in SHAPE_PARAMETER_NAMES:
        print(
            f"  d<{objective_label}>/d {name} JAX vs finite difference: "
            f"{jax_derivatives[name]:.12g} vs "
            f"{finite_difference_derivatives[name]:.12g}"
        )
        print(
            f"  d<{objective_label}>/d {name} absolute error: "
            f"{finite_difference_errors[name]:.6e} <= {finite_difference_tol:.6e}"
        )
    if gradient_check_required:
        print(f"  Interior gradient norm: {gradient_norm:.6e} <= {gradient_tol:.6e}")
    else:
        print(
            "  Interior gradient norm skipped: "
            "the point is on/near an implicit pressure-domain edge"
        )
    print(f"  Max perturbed {objective_label}: {max_perturbed_pressure:.12g}")
    print(f"  Optimum {objective_label}: {float(pressure):.12g}")


def run_unit_tests(
    result,
    A=DEFAULT_A,
    bounds=None,
    n_quad=2048,
    pressure_objective=None,
    normalized_method=None,
    normalized_grid_size=None,
    gradient_tol=1e-5,
    perturbation_tol=1e-9,
    finite_difference_step=1e-5,
    finite_difference_tol=1e-5,
    perturbation_radius=1e-3,
    perturbation_samples=100,
    perturbation_seed=0,
    active_bound_tol=1e-4,
    use_bounds=False,
    raise_on_failure=True,
    verbose=True,
):
    """Run local optimality checks around the optimizer result."""
    shape = result["shape"]
    pressure = result["pressure"]
    raw = result["raw"]
    if bounds is None:
        bounds = result.get("bounds")
    if bounds is not None:
        use_bounds = True
    selected_pressure_objective = pressure_objective or result.get(
        "pressure_objective",
        OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
    )
    normalized_method = normalized_method or result.get("normalized_method", "contour")
    normalized_grid_size = normalized_grid_size or result.get(
        "normalized_grid_size",
        500,
    )
    objective_label = PRESSURE_OBJECTIVE_LABELS[selected_pressure_objective]

    bounds_array, pressure_objective_fn = _pressure_objective_for_unit_tests(
        A,
        bounds,
        n_quad,
        use_bounds,
        selected_pressure_objective,
        normalized_method,
        normalized_grid_size,
    )
    low = bounds_array[:, 0]
    high = bounds_array[:, 1]

    gradient = jax.grad(pressure_objective_fn)(raw)
    gradient_norm = float(jnp.linalg.norm(gradient))

    lower_active, upper_active = _active_shape_bounds(
        shape,
        low,
        high,
        active_bound_tol,
    )
    max_perturbed_pressure = _max_pressure_near_shape(
        shape,
        low,
        high,
        lower_active,
        upper_active,
        A,
        n_quad,
        selected_pressure_objective,
        normalized_method,
        normalized_grid_size,
        perturbation_radius,
        perturbation_samples,
        perturbation_seed,
    )
    (
        jax_derivatives,
        finite_difference_derivatives,
        finite_difference_errors,
    ) = _compare_shape_derivatives_to_finite_difference(
        shape,
        low,
        high,
        A,
        n_quad,
        selected_pressure_objective,
        normalized_method,
        normalized_grid_size,
        finite_difference_step,
    )
    max_finite_difference_error = max(finite_difference_errors.values())
    gradient_check_required = _centered_shape_steps_are_finite(
        shape,
        low,
        high,
        A,
        n_quad,
        selected_pressure_objective,
        normalized_method,
        normalized_grid_size,
        finite_difference_step,
    )

    checks = {
        "perturbation": (
            np.isfinite(max_perturbed_pressure)
            and max_perturbed_pressure <= float(pressure) + perturbation_tol
        ),
        "finite_difference_derivatives": (
            max_finite_difference_error <= finite_difference_tol
        ),
    }
    if gradient_check_required:
        checks["interior_gradient"] = gradient_norm <= gradient_tol

    if verbose:
        _print_unit_test_report(
            objective_label,
            gradient_norm,
            gradient_tol,
            gradient_check_required,
            max_perturbed_pressure,
            pressure,
            jax_derivatives,
            finite_difference_derivatives,
            finite_difference_errors,
            finite_difference_tol,
        )

    failures = [name for name, passed in checks.items() if not passed]
    informational_failures = []
    if bounds is not None:
        informational_failures = [
            name for name in failures if name in BOUNDED_SEARCH_INFORMATIONAL_CHECKS
        ]
    fatal_failures = [
        name for name in failures if name not in set(informational_failures)
    ]

    if fatal_failures:
        if raise_on_failure:
            raise AssertionError(f"failed unit checks: {', '.join(fatal_failures)}")
    elif informational_failures and verbose:
        print(
            "  Bounded-search checks not treated as failures: "
            f"{', '.join(informational_failures)}"
        )
    elif verbose:
        print("  All unit checks passed.")

    return {
        "gradient_norm": gradient_norm,
        "max_perturbed_pressure": max_perturbed_pressure,
        "jax_derivatives": jax_derivatives,
        "finite_difference_derivatives": finite_difference_derivatives,
        "finite_difference_errors": finite_difference_errors,
        "checks": checks,
        "failures": failures,
        "informational_failures": informational_failures,
        "fatal_failures": fatal_failures,
        "passed": not fatal_failures,
    }


# ---------------------------------------------------------------------------
# Plotting, summaries, and command-line entry point
# ---------------------------------------------------------------------------


def plot_flux_contours(
    shape,
    A=DEFAULT_A,
    output_path=None,
    grid_size=600,
    contour_count=20,
    show=False,
):
    """Plot Solov'ev flux contours with the paper-style jet colormap."""
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon, kappa, delta = _validate_shape_domain(shape)
    psi, _ = _make_psi_for_shape((epsilon, kappa, delta), A=A)

    x_min = max(np.finfo(float).tiny, 1 - epsilon - 0.05)
    x = np.linspace(x_min, 1 + epsilon + 0.1, int(grid_size))
    y = np.linspace(-kappa * epsilon - 0.05, kappa * epsilon + 0.025, int(grid_size))
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(psi(X, Y), dtype=float)

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


def plot_averaged_volume_pressure_history(result, output_path=DEFAULT_RANDOM_HISTORY_PLOT):
    """Plot averaged volume pressure over optimizer iterations for one run."""
    import matplotlib.pyplot as plt

    history = result.get("history", [])
    if not history:
        raise ValueError("result has no optimization history")

    iterations = [entry["iteration"] for entry in history]
    objective_pressures = [
        entry["objective_pressure"]
        if "objective_pressure" in entry
        else entry["averaged_volume_pressure"]
        for entry in history
    ]
    objective_label = result.get(
        "pressure_objective_label",
        PRESSURE_OBJECTIVE_LABELS[OBJECTIVE_AVERAGED_VOLUME_PRESSURE],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(iterations, objective_pressures, marker="o", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(objective_label.title())
    ax.set_title("Best random run")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _marker_sizes_from_pressure(pressures, min_size=60.0, max_size=360.0):
    """Scale pressure values into readable scatter marker areas."""
    pressures = np.asarray(pressures, dtype=float)
    finite_pressures = pressures[np.isfinite(pressures)]
    if finite_pressures.size == 0:
        return np.full_like(pressures, (min_size + max_size) / 2.0)

    pressure_min = float(np.min(finite_pressures))
    pressure_max = float(np.max(finite_pressures))
    if np.isclose(pressure_min, pressure_max):
        return np.full_like(pressures, (min_size + max_size) / 2.0)

    normalized = (pressures - pressure_min) / (pressure_max - pressure_min)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    return min_size + normalized * (max_size - min_size)


def plot_random_final_shape_map(results, output_path=DEFAULT_RANDOM_MAP_PLOT):
    """Plot final epsilon/delta points, with kappa color and pressure size."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("at least one random run is needed to plot the shape map")

    final_shapes = np.asarray(
        [[float(value) for value in result["shape"]] for result in results],
        dtype=float,
    )
    final_averaged_volume_pressures = np.asarray(
        [
            float(result.get("objective_pressure", result["averaged_volume_pressure"]))
            for result in results
        ],
        dtype=float,
    )
    objective_label = results[0].get(
        "pressure_objective_label",
        PRESSURE_OBJECTIVE_LABELS[OBJECTIVE_AVERAGED_VOLUME_PRESSURE],
    )

    epsilons = final_shapes[:, 0]
    kappas = final_shapes[:, 1]
    deltas = final_shapes[:, 2]
    marker_sizes = _marker_sizes_from_pressure(final_averaged_volume_pressures)

    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    scatter = ax.scatter(
        epsilons,
        deltas,
        c=kappas,
        s=marker_sizes,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.7,
        alpha=0.85,
    )
    ax.set_xlabel("Final $\\epsilon$")
    ax.set_ylabel("Final $\\delta$")
    ax.set_title("Final shape-parameter map")
    ax.grid(True, alpha=0.3)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Final $\\kappa$")

    finite_pressures = final_averaged_volume_pressures[
        np.isfinite(final_averaged_volume_pressures)
    ]
    if finite_pressures.size == 0:
        legend_pressures = np.asarray([np.nan])
    else:
        pressure_min = float(np.min(finite_pressures))
        pressure_max = float(np.max(finite_pressures))
        if np.isclose(pressure_min, pressure_max):
            legend_pressures = np.asarray([pressure_min])
        else:
            legend_pressures = np.linspace(pressure_min, pressure_max, 3)
    legend_sizes = _marker_sizes_from_pressure(legend_pressures)
    legend_handles = [
        ax.scatter(
            [],
            [],
            s=size,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            label=f"{pressure:.3g}",
        )
        for pressure, size in zip(legend_pressures, legend_sizes)
    ]
    ax.legend(
        handles=legend_handles,
        title=f"Final {objective_label}",
        scatterpoints=1,
        frameon=True,
        loc="best",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _format_shape(shape):
    epsilon, kappa, delta = [float(v) for v in shape]
    return f"epsilon={epsilon:.12g}, kappa={kappa:.12g}, delta={delta:.12g}"


def _format_bounds(bounds):
    parts = []
    for name, (lower, upper) in zip(SHAPE_PARAMETER_NAMES, np.asarray(bounds, dtype=float)):
        parts.append(f"{name} in [{lower:.12g}, {upper:.12g}]")
    return ", ".join(parts)


def _print_single_run_summary(result):
    optimizer = result["optimizer"]
    objective_label = result.get(
        "pressure_objective_label",
        PRESSURE_OBJECTIVE_LABELS[OBJECTIVE_AVERAGED_VOLUME_PRESSURE],
    )

    print(f"Initial shape: {_format_shape(result['initial_shape'])}")
    print(
        f"Initial {objective_label}: "
        f"{float(result.get('initial_objective_pressure', result['initial_pressure'])):.12g}"
    )
    print(f"Final shape: {_format_shape(result['shape'])}")
    print(
        f"Final {objective_label}: "
        f"{float(result.get('objective_pressure', result['pressure'])):.12g}"
    )
    if result.get("pressure_objective") != OBJECTIVE_AVERAGED_VOLUME_PRESSURE:
        print(
            "Final averaged volume pressure: "
            f"{float(result['averaged_volume_pressure']):.12g}"
        )
    print(f"Gradient norm: {float(result['gradient_norm']):.12g}")
    if result.get("bounds") is not None:
        print(f"Constraints: {_format_bounds(result['bounds'])}")
    else:
        print("Constraints: epsilon > 0, kappa > 0")
    if hasattr(optimizer, "status"):
        print(f"Optimizer status: {int(optimizer.status)}")
    if hasattr(optimizer, "success"):
        print(f"Optimizer success: {bool(optimizer.success)}")


def print_random_run_summary(results):
    objective_label = results[0].get(
        "pressure_objective_label",
        PRESSURE_OBJECTIVE_LABELS[OBJECTIVE_AVERAGED_VOLUME_PRESSURE],
    )
    print("Summary:")
    print(f"{'Run':>3} {'Final shape':<58}  {('Final ' + objective_label):>26}  Unit tests")
    for index, result in enumerate(results, start=1):
        print(
            f"{index:>3}  "
            f"{_format_shape(result['shape']):<58}  "
            f"{float(result.get('objective_pressure', result['pressure'])):>26.12g}  "
            f"{result.get('unit_test_result', 'not run')}"
        )


def write_random_run_summary_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_number",
        "starting_shape",
        "final_shape",
        "pressure_objective",
        "final_objective_pressure",
        "final_averaged_volume_pressure",
        "unit_test_result",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, result in enumerate(results):
            writer.writerow(
                {
                    "run_number": index + 1,
                    "starting_shape": _format_shape(result["initial_shape"]),
                    "final_shape": _format_shape(result["shape"]),
                    "pressure_objective": result.get(
                        "pressure_objective",
                        OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
                    ),
                    "final_objective_pressure": float(
                        result.get("objective_pressure", result["pressure"])
                    ),
                    "final_averaged_volume_pressure": float(
                        result["averaged_volume_pressure"]
                    ),
                    "unit_test_result": result.get("unit_test_result", "not run"),
                }
            )
    return output_path


def _shape_bounds_from_args(args):
    supplied_values = (
        args.epsilon_min,
        args.epsilon_max,
        args.kappa_min,
        args.kappa_max,
        args.delta_min,
        args.delta_max,
    )
    if all(value is None for value in supplied_values):
        return None

    bounds = np.asarray(
        [
            [
                DOMAIN_EPS if args.epsilon_min is None else args.epsilon_min,
                float("inf") if args.epsilon_max is None else args.epsilon_max,
            ],
            [
                DOMAIN_EPS if args.kappa_min is None else args.kappa_min,
                float("inf") if args.kappa_max is None else args.kappa_max,
            ],
            [
                -float("inf") if args.delta_min is None else args.delta_min,
                float("inf") if args.delta_max is None else args.delta_max,
            ],
        ],
        dtype=float,
    )
    return _effective_shape_bounds(bounds, use_bounds=True)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument(
        "--initial",
        nargs=3,
        type=float,
        default=DEFAULT_INITIAL_SHAPE,
        metavar=("EPS", "KAP", "DLT"),
    )
    parser.add_argument(
        "--n-quad",
        type=int,
        default=2048,
        help="grid size for averaged-volume-pressure contour integration",
    )
    parser.add_argument(
        "--pressure-objective",
        choices=PRESSURE_OBJECTIVE_CHOICES,
        default=OBJECTIVE_AVERAGED_VOLUME_PRESSURE,
        help="pressure objective to maximize",
    )
    parser.add_argument(
        "--normalized-method",
        choices=("contour", "masking"),
        default="contour",
        help="integration method for --pressure-objective normalized-psi-pressure",
    )
    parser.add_argument(
        "--normalized-grid-size",
        type=int,
        default=500,
        help="grid size N for --pressure-objective normalized-psi-pressure",
    )
    parser.add_argument("--epsilon-min", type=float, help="lower epsilon search bound")
    parser.add_argument("--epsilon-max", type=float, help="upper epsilon search bound")
    parser.add_argument("--kappa-min", type=float, help="lower kappa search bound")
    parser.add_argument("--kappa-max", type=float, help="upper kappa search bound")
    parser.add_argument("--delta-min", type=float, help="lower delta search bound")
    parser.add_argument("--delta-max", type=float, help="upper delta search bound")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--plot", type=Path, help="save optimized flux contours to this path")
    parser.add_argument("--plot-grid", type=int, default=600)
    parser.add_argument("--contour-count", type=int, default=20)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--random-initials",
        action="store_true",
        help="run random initial shapes",
    )
    parser.add_argument("--random-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--random-summary-csv",
        type=Path,
        default=DEFAULT_RANDOM_SUMMARY_CSV,
        help="CSV path for --random-initials summary",
    )
    parser.add_argument(
        "--random-map-plot",
        type=Path,
        default=DEFAULT_RANDOM_MAP_PLOT,
        help="PNG path for --random-initials final shape-parameter map",
    )
    parser.add_argument(
        "--random-history-plot",
        type=Path,
        default=DEFAULT_RANDOM_HISTORY_PLOT,
        help="PNG path for the best random run's objective history",
    )
    parser.add_argument(
        "--random-flux-plot",
        type=Path,
        default=DEFAULT_RANDOM_FLUX_PLOT,
        help="PNG path for the best random run's flux contours",
    )
    parser.add_argument("--fd-step", type=float, default=1e-5)
    parser.add_argument("--fd-tol", type=float, default=1e-5)
    parser.add_argument("--test-perturbation-radius", type=float, default=1e-3)
    parser.add_argument("--test-perturbation-samples", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=0)
    return parser


def _unit_test_result_label(result, args, run_number):
    bounds = _shape_bounds_from_args(args)
    try:
        unit_test_result = run_unit_tests(
            result,
            A=args.A,
            bounds=bounds,
            n_quad=args.n_quad,
            pressure_objective=args.pressure_objective,
            normalized_method=args.normalized_method,
            normalized_grid_size=args.normalized_grid_size,
            use_bounds=bounds is not None,
            finite_difference_step=args.fd_step,
            finite_difference_tol=args.fd_tol,
            perturbation_radius=args.test_perturbation_radius,
            perturbation_samples=args.test_perturbation_samples,
            perturbation_seed=args.test_seed + run_number - 1,
            raise_on_failure=False,
            verbose=False,
        )
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"

    fatal_failures = unit_test_result.get("fatal_failures", unit_test_result["failures"])
    informational_failures = unit_test_result.get("informational_failures", [])
    if fatal_failures:
        return f"failed: {', '.join(fatal_failures)}"
    if informational_failures:
        return f"passed (bounded: {', '.join(informational_failures)})"
    return "passed"


def _run_random_initials_from_args(args):
    bounds = _shape_bounds_from_args(args)
    results = multiple_random_runs(
        count=args.random_count,
        seed=args.random_seed,
        A=args.A,
        bounds=bounds,
        n_quad=args.n_quad,
        pressure_objective=args.pressure_objective,
        normalized_method=args.normalized_method,
        normalized_grid_size=args.normalized_grid_size,
        maxiter=args.maxiter,
        gtol=args.gtol,
        use_bounds=bounds is not None,
    )
    for run_number, result in enumerate(results, start=1):
        result["unit_test_result"] = _unit_test_result_label(result, args, run_number)

    best_result = max(
        results,
        key=lambda item: (
            float(item["objective_pressure"])
            if np.isfinite(float(item["objective_pressure"]))
            else -np.inf
        ),
    )

    objective_label = PRESSURE_OBJECTIVE_LABELS[args.pressure_objective]
    print_random_run_summary(results)
    print("Best result:")
    print(f"  Initial shape: {_format_shape(best_result['initial_shape'])}")
    print(f"  Final shape: {_format_shape(best_result['shape'])}")
    print(
        f"  Final {objective_label}: "
        f"{float(best_result['objective_pressure']):.12g}"
    )
    if args.pressure_objective != OBJECTIVE_AVERAGED_VOLUME_PRESSURE:
        print(
            "  Final averaged volume pressure: "
            f"{float(best_result['averaged_volume_pressure']):.12g}"
        )
    print(f"  Unit tests: {best_result['unit_test_result']}")
    if bounds is not None:
        print(f"Constraints: {_format_bounds(bounds)}")
    else:
        print("Constraints: epsilon > 0, kappa > 0")

    csv_path = write_random_run_summary_csv(results, args.random_summary_csv)
    print(f"Saved random summary CSV: {csv_path}")

    map_path = plot_random_final_shape_map(results, args.random_map_plot)
    print(f"Saved random shape map: {map_path}")

    history_path = plot_averaged_volume_pressure_history(
        best_result,
        args.random_history_plot,
    )
    print(f"Saved best-run averaged volume pressure history: {history_path}")

    plot_flux_contours(
        best_result["shape"],
        A=args.A,
        output_path=args.random_flux_plot,
        grid_size=args.plot_grid,
        contour_count=args.contour_count,
    )
    print(f"Saved best-run flux contours: {args.random_flux_plot}")


def _run_single_initial_from_args(args):
    bounds = _shape_bounds_from_args(args)
    result = single_run(
        initial_shape=tuple(args.initial),
        A=args.A,
        bounds=bounds,
        n_quad=args.n_quad,
        pressure_objective=args.pressure_objective,
        normalized_method=args.normalized_method,
        normalized_grid_size=args.normalized_grid_size,
        maxiter=args.maxiter,
        gtol=args.gtol,
        use_bounds=bounds is not None,
    )

    _print_single_run_summary(result)
    if not args.skip_tests:
        run_unit_tests(
            result,
            A=args.A,
            bounds=bounds,
            n_quad=args.n_quad,
            pressure_objective=args.pressure_objective,
            normalized_method=args.normalized_method,
            normalized_grid_size=args.normalized_grid_size,
            use_bounds=bounds is not None,
            finite_difference_step=args.fd_step,
            finite_difference_tol=args.fd_tol,
            perturbation_radius=args.test_perturbation_radius,
            perturbation_samples=args.test_perturbation_samples,
            perturbation_seed=args.test_seed,
        )
    if args.plot or args.show_plot:
        plot_flux_contours(
            result["shape"],
            A=args.A,
            output_path=args.plot,
            grid_size=args.plot_grid,
            contour_count=args.contour_count,
            show=args.show_plot,
        )
        if args.plot:
            print(f"Saved plot: {args.plot}")


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.random_initials:
        _run_random_initials_from_args(args)
    else:
        _run_single_initial_from_args(args)


if __name__ == "__main__":
    main()
