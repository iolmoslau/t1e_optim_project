#!/usr/bin/env python3
"""Unconstrained shape optimization for selected pressure objectives.

Optimize the shape triple [epsilon, kappa, delta] with only positive lower
bounds on epsilon and kappa.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for source_dir in (REPO_ROOT, REPO_ROOT / "pressure_integral", REPO_ROOT / "ITER_Equilibria"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from optimal_JAX.utils_JAX import (  # noqa: E402
    DEFAULT_A,
    beta_toroidal,
    get_vol_av_p_from_params,
    normalized_psi_pressure,
)
from optimal_JAX.plot_contour_from_shape import plot_contour_from_shape  # noqa: E402


PARAMETER_NAMES = ("epsilon", "kappa", "delta")
DEFAULT_INITIAL_SHAPE = np.array([0.32, 1.30, 0.20], dtype=float)
DEFAULT_N = 500
DEFAULT_METHOD = "contour"
DEFAULT_Q = 2.0
DEFAULT_MAXITER = 200
DEFAULT_FTOL = 1e-13
DEFAULT_GTOL = 1e-10
DEFAULT_FINITE_DIFF_STEP = 1e-10
DEFAULT_POSITIVE_FLOOR = 0.10
DEFAULT_OUTPUT_DIR = Path("optimal_JAX/output")
BAD_OBJECTIVE_VALUE = 1e100

OBJECTIVE_VOLUME_PRESSURE = "get_vol_av_p_from_params"
OBJECTIVE_NORMALIZED_PRESSURE = "normalized_psi_pressure"
OBJECTIVE_BETA_TOROIDAL = "beta_toroidal"
OBJECTIVE_CHOICES = (
    OBJECTIVE_VOLUME_PRESSURE,
    OBJECTIVE_NORMALIZED_PRESSURE,
    OBJECTIVE_BETA_TOROIDAL,
)
OBJECTIVE_SLUGS = {
    OBJECTIVE_VOLUME_PRESSURE: "volume_pressure",
    OBJECTIVE_NORMALIZED_PRESSURE: "normalized_psi_pressure",
    OBJECTIVE_BETA_TOROIDAL: "beta_toroidal",
}


def default_plot_path(objective: str) -> Path:
    """Return the default final-shape PNG path for one objective."""
    return DEFAULT_OUTPUT_DIR / f"optimal_unconstrained_{OBJECTIVE_SLUGS[objective]}_final_shape.png"


def finite_float(name: str):
    """Return an argparse type accepting finite floating-point values."""

    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from exc
        if not np.isfinite(parsed):
            raise argparse.ArgumentTypeError(f"{name} must be finite")
        return parsed

    return parse


def positive_float(name: str):
    """Return an argparse type accepting positive finite floats."""

    parse_finite = finite_float(name)

    def parse(value: str) -> float:
        parsed = parse_finite(value)
        if parsed <= 0.0:
            raise argparse.ArgumentTypeError(f"{name} must be positive")
        return parsed

    return parse


def int_at_least(name: str, minimum: int):
    """Return an argparse type accepting integers at least minimum."""

    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{name} must be at least {minimum}")
        return parsed

    return parse


def validate_shape(shape, epsilon_floor: float, kappa_floor: float) -> np.ndarray:
    """Validate only the optimizer's explicit shape bounds."""
    values = np.asarray(shape, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("shape must contain finite epsilon, kappa, and delta")
    if values[0] < epsilon_floor:
        raise ValueError(f"epsilon must be at least {epsilon_floor:g}")
    if values[1] < kappa_floor:
        raise ValueError(f"kappa must be at least {kappa_floor:g}")
    return values


def optimizer_bounds(epsilon_floor: float, kappa_floor: float):
    """Only epsilon and kappa get positive lower bounds."""
    return [(float(epsilon_floor), None), (float(kappa_floor), None), (-0.70, 0.80)]


def objective_value(
    shape,
    objective: str,
    A: float = DEFAULT_A,
    method: str = DEFAULT_METHOD,
    N: int = DEFAULT_N,
    q: float = DEFAULT_Q,
) -> float:
    """Evaluate the selected objective for one shape."""
    epsilon, kappa, delta = (float(value) for value in np.asarray(shape, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if objective == OBJECTIVE_VOLUME_PRESSURE:
            value = get_vol_av_p_from_params(
                epsilon,
                kappa,
                delta,
                A=float(A),
                method=method,
                N=int(N),
            )
        elif objective == OBJECTIVE_NORMALIZED_PRESSURE:
            value = normalized_psi_pressure(
                epsilon,
                kappa,
                delta,
                A=float(A),
                method=method,
                N=int(N),
            )
        elif objective == OBJECTIVE_BETA_TOROIDAL:
            value = beta_toroidal(
                epsilon,
                kappa,
                delta,
                A=float(A),
                q=float(q),
                N=int(N),
            )
        else:
            raise ValueError(f"unknown objective: {objective}")

    return float(value)


def safe_objective_value(shape, *args, **kwargs) -> float:
    """Return NaN instead of letting invalid trial shapes stop the optimizer."""
    try:
        value = objective_value(shape, *args, **kwargs)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def objective_gradient(
    shape,
    objective: str,
    A: float = DEFAULT_A,
    method: str = DEFAULT_METHOD,
    N: int = DEFAULT_N,
    q: float = DEFAULT_Q,
    finite_diff_step: float = DEFAULT_FINITE_DIFF_STEP,
    epsilon_floor: float = DEFAULT_POSITIVE_FLOOR,
    kappa_floor: float = DEFAULT_POSITIVE_FLOOR,
) -> np.ndarray:
    """Finite-difference gradient used by the JAX callback wrapper."""
    shape = np.asarray(shape, dtype=float)
    if (
        shape.shape != (3,)
        or not np.all(np.isfinite(shape))
        or shape[0] < epsilon_floor
        or shape[1] < kappa_floor
        or finite_diff_step <= 0.0
    ):
        return np.full(3, np.nan, dtype=float)

    center_value = safe_objective_value(
        shape,
        objective,
        A=A,
        method=method,
        N=N,
        q=q,
    )
    gradient = np.empty(3, dtype=float)
    lower_bounds = (float(epsilon_floor), float(kappa_floor), -np.inf)

    for index, lower_bound in enumerate(lower_bounds):
        plus_shape = shape.copy()
        plus_shape[index] += finite_diff_step
        plus_value = safe_objective_value(
            plus_shape,
            objective,
            A=A,
            method=method,
            N=N,
            q=q,
        )

        minus_value = np.nan
        if shape[index] - finite_diff_step >= lower_bound:
            minus_shape = shape.copy()
            minus_shape[index] -= finite_diff_step
            minus_value = safe_objective_value(
                minus_shape,
                objective,
                A=A,
                method=method,
                N=N,
                q=q,
            )

        if np.isfinite(plus_value) and np.isfinite(minus_value):
            gradient[index] = (plus_value - minus_value) / (2.0 * finite_diff_step)
        elif np.isfinite(plus_value) and np.isfinite(center_value):
            gradient[index] = (plus_value - center_value) / finite_diff_step
        elif np.isfinite(minus_value) and np.isfinite(center_value):
            gradient[index] = (center_value - minus_value) / finite_diff_step
        else:
            gradient[index] = np.nan

    return gradient


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def objective_value_jax(
    shape,
    objective,
    A,
    method,
    N,
    q,
    finite_diff_step,
    epsilon_floor,
    kappa_floor,
):
    """JAX-facing wrapper around the selected host-side objective."""
    return jax.pure_callback(
        lambda host_shape: np.asarray(
            safe_objective_value(
                host_shape,
                objective,
                A=A,
                method=method,
                N=N,
                q=q,
            ),
            dtype=np.float64,
        ),
        jax.ShapeDtypeStruct((), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )


@objective_value_jax.defjvp
def objective_value_jax_jvp(
    objective,
    A,
    method,
    N,
    q,
    finite_diff_step,
    epsilon_floor,
    kappa_floor,
    primals,
    tangents,
):
    """Tell JAX how to differentiate the host callback."""
    (shape,) = primals
    (shape_dot,) = tangents
    value = objective_value_jax(
        shape,
        objective,
        A,
        method,
        N,
        q,
        finite_diff_step,
        epsilon_floor,
        kappa_floor,
    )
    gradient = jax.pure_callback(
        lambda host_shape: np.asarray(
            objective_gradient(
                host_shape,
                objective,
                A=A,
                method=method,
                N=N,
                q=q,
                finite_diff_step=finite_diff_step,
                epsilon_floor=epsilon_floor,
                kappa_floor=kappa_floor,
            ),
            dtype=np.float64,
        ),
        jax.ShapeDtypeStruct((3,), jnp.float64),
        jnp.asarray(shape, dtype=jnp.float64),
    )
    return value, jnp.dot(gradient, shape_dot)


def optimize_shape(
    initial_shape=DEFAULT_INITIAL_SHAPE,
    objective: str = OBJECTIVE_VOLUME_PRESSURE,
    A: float = DEFAULT_A,
    method: str = DEFAULT_METHOD,
    N: int = DEFAULT_N,
    q: float = DEFAULT_Q,
    maxiter: int = DEFAULT_MAXITER,
    ftol: float = DEFAULT_FTOL,
    gtol: float = DEFAULT_GTOL,
    finite_diff_step: float = DEFAULT_FINITE_DIFF_STEP,
    epsilon_floor: float = DEFAULT_POSITIVE_FLOOR,
    kappa_floor: float = DEFAULT_POSITIVE_FLOOR,
):
    """Maximize the selected objective with L-BFGS-B lower bounds only."""
    initial_shape = validate_shape(initial_shape, epsilon_floor, kappa_floor)
    bounds = optimizer_bounds(epsilon_floor, kappa_floor)
    jax_gradient_evaluations = 0

    objective_with_gradient = jax.value_and_grad(
        lambda shape: objective_value_jax(
            shape,
            objective,
            float(A),
            method,
            int(N),
            float(q),
            float(finite_diff_step),
            float(epsilon_floor),
            float(kappa_floor),
        )
    )

    def value_and_gradient(shape):
        nonlocal jax_gradient_evaluations
        jax_gradient_evaluations += 1
        try:
            value, gradient = objective_with_gradient(
                jnp.asarray(shape, dtype=jnp.float64)
            )
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
        except Exception:
            return np.nan, np.full(3, np.nan, dtype=float)

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return np.nan, np.full(3, np.nan, dtype=float)
        return value, gradient

    initial_objective, initial_gradient = value_and_gradient(initial_shape)
    path = [initial_shape.copy()]

    def negative_objective_and_gradient(shape):
        value, gradient = value_and_gradient(shape)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return BAD_OBJECTIVE_VALUE, np.zeros(3, dtype=float)
        return -value, -gradient

    def save_optimizer_step(shape):
        path.append(np.asarray(shape, dtype=float).copy())

    result = minimize(
        negative_objective_and_gradient,
        initial_shape,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=save_optimizer_step,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
        },
    )

    final_shape = np.asarray(result.x, dtype=float)
    if not np.allclose(path[-1], final_shape):
        path.append(final_shape.copy())
    final_objective, final_gradient = value_and_gradient(final_shape)
    gradient_norm = float(np.linalg.norm(final_gradient))

    return {
        "result": result,
        "path": np.asarray(path, dtype=float),
        "objective": objective,
        "initial_shape": initial_shape,
        "final_shape": final_shape,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "initial_gradient": initial_gradient,
        "final_gradient": final_gradient,
        "gradient_norm": gradient_norm,
        "jax_gradient_evaluations": jax_gradient_evaluations,
        "finite_diff_step": float(finite_diff_step),
        "A": float(A),
        "method": method,
        "N": int(N),
        "q": float(q),
        "bounds": bounds,
    }


def print_shape(label: str, shape) -> None:
    """Print a shape triple in a compact block."""
    epsilon, kappa, delta = (float(value) for value in shape)
    print(label)
    print(f"  epsilon: {epsilon:.10g}")
    print(f"  kappa:   {kappa:.10g}")
    print(f"  delta:   {delta:.10g}")


def format_objective(value: float) -> str:
    if np.isfinite(value):
        return f"{value:.12g}"
    return "nan"


def print_summary(run) -> None:
    """Print the requested optimizer summary."""
    result = run["result"]
    print()
    print(f"objective: {run['objective']}")
    print(f"A: {run['A']:.12g}")
    print(f"N: {run['N']}")
    if run["objective"] in (OBJECTIVE_VOLUME_PRESSURE, OBJECTIVE_NORMALIZED_PRESSURE):
        print(f"method: {run['method']}")
    if run["objective"] == OBJECTIVE_BETA_TOROIDAL:
        print(f"q: {run['q']:.12g}")
    print("optimizer: L-BFGS-B")
    print("gradient source: jax.value_and_grad with custom finite-difference JVP")
    print(f"optimizer success: {bool(result.success)}")
    print(f"optimizer status: {int(result.status)}")
    print(f"optimizer message: {result.message}")
    print(f"iterations: {result.nit}")
    print(f"function evaluations: {result.nfev}")
    if hasattr(result, "njev"):
        print(f"gradient evaluations: {result.njev}")
    print(f"JAX gradient calls: {run['jax_gradient_evaluations']}")
    print(f"finite-difference step: {run['finite_diff_step']:.12g}")
    print_shape("initial shape", run["initial_shape"])
    print_shape("final shape", run["final_shape"])
    print(f"initial objective: {format_objective(run['initial_objective'])}")
    print(f"final objective: {format_objective(run['final_objective'])}")
    print(f"final gradient norm: {format_objective(run['gradient_norm'])}")


def result_payload(run) -> dict:
    """Build a JSON-serializable optimization summary."""
    result = run["result"]
    return {
        "objective": run["objective"],
        "A": run["A"],
        "N": run["N"],
        "method": run["method"],
        "q": run["q"],
        "initial_shape": run["initial_shape"].tolist(),
        "final_shape": run["final_shape"].tolist(),
        "initial_objective": run["initial_objective"],
        "final_objective": run["final_objective"],
        "initial_gradient": run["initial_gradient"].tolist(),
        "final_gradient": run["final_gradient"].tolist(),
        "gradient_norm": run["gradient_norm"],
        "jax_gradient_evaluations": run["jax_gradient_evaluations"],
        "finite_diff_step": run["finite_diff_step"],
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "gradient_evaluations": int(getattr(result, "njev", 0)),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize [epsilon, kappa, delta] with no constraints except "
            "epsilon > 0 and kappa > 0."
        )
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_CHOICES,
        default=OBJECTIVE_VOLUME_PRESSURE,
        help="Objective function to maximize.",
    )
    parser.add_argument(
        "--start-epsilon",
        type=finite_float("start epsilon"),
        default=DEFAULT_INITIAL_SHAPE[0],
        help="Starting epsilon.",
    )
    parser.add_argument(
        "--start-kappa",
        type=finite_float("start kappa"),
        default=DEFAULT_INITIAL_SHAPE[1],
        help="Starting kappa.",
    )
    parser.add_argument(
        "--start-delta",
        type=finite_float("start delta"),
        default=DEFAULT_INITIAL_SHAPE[2],
        help="Starting delta.",
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
        help="Maximum L-BFGS-B iterations.",
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a JSON summary.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help=(
            "PNG path for the optimized final shape contour plot. Defaults to "
            "an objective-specific name in optimal_JAX/output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_shape = np.array(
        [args.start_epsilon, args.start_kappa, args.start_delta],
        dtype=float,
    )
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
    print_summary(run)

    epsilon, kappa, delta = (float(value) for value in run["final_shape"])
    output_path = args.plot if args.plot is not None else default_plot_path(args.objective)
    plot_path = plot_contour_from_shape(
        epsilon,
        kappa,
        delta,
        output_path,
        A=args.A,
    )
    print(f"saved final shape contour plot: {plot_path}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result_payload(run), indent=2) + "\n")


if __name__ == "__main__":
    main()
