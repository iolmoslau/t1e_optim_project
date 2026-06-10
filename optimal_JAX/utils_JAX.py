#!/usr/bin/env python3
"""JAX pressure utilities mirroring pressure_integral.pressure_utils."""

from __future__ import annotations

import functools

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from contourpy import contour_generator

from ITER_Equilibria import ExactSolutions as _exact
from pressure_integral.psi_anti_deriv_exact import G_total as _pressure_G_total


MU_0 = 4 * np.pi * 1e-7
DEFAULT_A = -0.05
DEFAULT_N = 500

__all__ = [
    "make_psi",
    "extract_zero_contour",
    "int_contour_boundary",
    "poloidal_circum",
    "int_masking",
    "volume_average",
    "get_vol_av_p_from_params",
    "normalized_psi_pressure",
    "integral_multiplier",
    "inv_q_star",
    "beta_t_alternative",
    "beta_poloidal",
    "beta_toroidal",
    "beta_poloidal_updated",
    "q_star_updated",
    "beta_toroidal_updated",
    "MU_0",
    "DEFAULT_A",
]


def accepts_shape_params(func):
    """Give scalar shape diagnostics the pressure_utils calling convention."""

    @functools.wraps(func)
    def wrapper(epsilon, kappa=None, delta=None, *args, **kwargs):
        arr = np.asarray(epsilon, dtype=float)
        if kappa is None and delta is None and arr.ndim == 2 and arr.shape[1] == 3:
            epsilon, kappa, delta = arr[:, 0], arr[:, 1], arr[:, 2]
        elif kappa is None or delta is None:
            raise ValueError(
                "kappa and delta are required when epsilon is not an Nx3 array"
            )

        if np.ndim(epsilon) or np.ndim(kappa) or np.ndim(delta):
            scalar = lambda e, k, d: func(e, k, d, *args, **kwargs)
            return np.vectorize(scalar)(epsilon, kappa, delta)

        return func(epsilon, kappa, delta, *args, **kwargs)

    return wrapper


def _validate_N(N) -> int:
    """Validate that the grid or boundary resolution is a positive integer."""
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Grid resolution N must be a positive integer.")
    return N


def _plasma_domain(epsilon, kappa, pad: float = 0.1):
    """Bounding box of the plasma in (x, y), padded on each side."""
    x_lim = (1 - epsilon - pad, 1 + epsilon + pad)
    y_lim = (-kappa * epsilon - pad, kappa * epsilon + pad)
    return x_lim, y_lim


def safe_log(x):
    """Logarithm used by the Solov'ev basis."""
    return jnp.log(jnp.maximum(x, 1e-12))


def basis_values(x, y):
    """Seven homogeneous Solov'ev basis functions."""
    x, y = jnp.broadcast_arrays(x, y)
    log_x = safe_log(x)
    x2 = x * x
    y2 = y * y

    return jnp.stack(
        [
            jnp.ones_like(x),
            x2,
            y2 - x2 * log_x,
            x2 * x2 - 4.0 * x2 * y2,
            2.0 * y2 * y2
            - 9.0 * x2 * y2
            + 3.0 * x2 * x2 * log_x
            - 12.0 * x2 * y2 * log_x,
            x2 * x2 * x2 - 12.0 * x2 * x2 * y2 + 8.0 * x2 * y2 * y2,
            8.0 * y2 * y2 * y2
            - 140.0 * x2 * y2 * y2
            + 75.0 * x2 * x2 * y2
            - 15.0 * x2 * x2 * x2 * log_x
            + 180.0 * x2 * x2 * y2 * log_x
            - 120.0 * x2 * y2 * y2 * log_x,
        ],
        axis=0,
    )


def particular_value(x, y, A):
    """Particular Solov'ev solution for source A + (1 - A) * x^2."""
    del y
    return A * (0.5 * x * x * safe_log(x)) + (1.0 - A) * (x**4 / 8.0)


def basis_x(x, y):
    return jax.jacfwd(lambda value: basis_values(value, y))(x)


def basis_y(x, y):
    return jax.jacfwd(lambda value: basis_values(x, value))(y)


def basis_xx(x, y):
    return jax.jacfwd(lambda value: basis_x(value, y))(x)


def basis_yy(x, y):
    return jax.jacfwd(lambda value: basis_y(x, value))(y)


def particular_x(x, y, A):
    return jax.grad(lambda value: particular_value(value, y, A))(x)


def particular_y(x, y, A):
    return jax.grad(lambda value: particular_value(x, value, A))(y)


def particular_xx(x, y, A):
    return jax.grad(lambda value: particular_x(value, y, A))(x)


def particular_yy(x, y, A):
    return jax.grad(lambda value: particular_y(x, value, A))(y)


def solve_coefficients(epsilon, kappa, delta, A=DEFAULT_A):
    """Find coefficients that make the flux match the requested boundary."""
    alpha = jnp.arcsin(delta)
    curv1 = -((1.0 + alpha) ** 2) / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * jnp.cos(alpha) ** 2)
    curv3 = ((1.0 - alpha) ** 2) / (epsilon * kappa**2)

    x_outer = 1.0 + epsilon
    x_inner = 1.0 - epsilon
    x_high = 1.0 - epsilon * delta
    y_high = kappa * epsilon
    zero = jnp.asarray(0.0)

    matrix = jnp.stack(
        [
            basis_values(x_outer, zero),
            basis_values(x_inner, zero),
            basis_values(x_high, y_high),
            basis_x(x_high, y_high),
            curv1 * basis_x(x_outer, zero) + basis_yy(x_outer, zero),
            curv3 * basis_x(x_inner, zero) + basis_yy(x_inner, zero),
            curv2 * basis_y(x_high, y_high) + basis_xx(x_high, y_high),
        ]
    )

    right_hand_side = -jnp.stack(
        [
            particular_value(x_outer, zero, A),
            particular_value(x_inner, zero, A),
            particular_value(x_high, y_high, A),
            particular_x(x_high, y_high, A),
            curv1 * particular_x(x_outer, zero, A)
            + particular_yy(x_outer, zero, A),
            curv3 * particular_x(x_inner, zero, A)
            + particular_yy(x_inner, zero, A),
            curv2 * particular_y(x_high, y_high, A)
            + particular_xx(x_high, y_high, A),
        ]
    )

    return jnp.linalg.solve(matrix, right_hand_side)


def make_psi(epsilon, kappa, delta, A: float = DEFAULT_A):
    """Return a JAX-callable Solov'ev flux function and its coefficients."""
    coefficients = solve_coefficients(epsilon, kappa, delta, A)

    def psi(x, y):
        return jnp.tensordot(
            coefficients,
            basis_values(jnp.asarray(x), jnp.asarray(y)),
            axes=(0, 0),
        ) + particular_value(jnp.asarray(x), jnp.asarray(y), A)

    return psi, coefficients, A


def _make_psi_numpy(epsilon, kappa, delta, A: float = DEFAULT_A):
    """NumPy pressure_utils-compatible Solov'ev flux function."""
    ex = _exact
    alpha = np.arcsin(delta)
    curv1 = -(1 + alpha) ** 2 / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * np.cos(alpha) ** 2)
    curv3 = (1 - alpha) ** 2 / (epsilon * kappa**2)

    xhi = 1 - epsilon * delta
    yhi = kappa * epsilon

    M = np.array([
        [ex.psi1(1+epsilon,0), ex.psi2(1+epsilon,0), ex.psi3(1+epsilon,0), ex.psi4(1+epsilon,0), ex.psi5(1+epsilon,0), ex.psi6(1+epsilon,0), ex.psi7(1+epsilon,0)],
        [ex.psi1(1-epsilon,0), ex.psi2(1-epsilon,0), ex.psi3(1-epsilon,0), ex.psi4(1-epsilon,0), ex.psi5(1-epsilon,0), ex.psi6(1-epsilon,0), ex.psi7(1-epsilon,0)],
        [ex.psi1(xhi,yhi),     ex.psi2(xhi,yhi),     ex.psi3(xhi,yhi),     ex.psi4(xhi,yhi),     ex.psi5(xhi,yhi),     ex.psi6(xhi,yhi),     ex.psi7(xhi,yhi)],
        [ex.psi1x(xhi,yhi),    ex.psi2x(xhi,yhi),    ex.psi3x(xhi,yhi),    ex.psi4x(xhi,yhi),    ex.psi5x(xhi,yhi),    ex.psi6x(xhi,yhi),    ex.psi7x(xhi,yhi)],
        [curv1*ex.psi1x(1+epsilon,0) + ex.psi1yy(1+epsilon,0),
         curv1*ex.psi2x(1+epsilon,0) + ex.psi2yy(1+epsilon,0),
         curv1*ex.psi3x(1+epsilon,0) + ex.psi3yy(1+epsilon,0),
         curv1*ex.psi4x(1+epsilon,0) + ex.psi4yy(1+epsilon,0),
         curv1*ex.psi5x(1+epsilon,0) + ex.psi5yy(1+epsilon,0),
         curv1*ex.psi6x(1+epsilon,0) + ex.psi6yy(1+epsilon,0),
         curv1*ex.psi7x(1+epsilon,0) + ex.psi7yy(1+epsilon,0)],
        [curv3*ex.psi1x(1-epsilon,0) + ex.psi1yy(1-epsilon,0),
         curv3*ex.psi2x(1-epsilon,0) + ex.psi2yy(1-epsilon,0),
         curv3*ex.psi3x(1-epsilon,0) + ex.psi3yy(1-epsilon,0),
         curv3*ex.psi4x(1-epsilon,0) + ex.psi4yy(1-epsilon,0),
         curv3*ex.psi5x(1-epsilon,0) + ex.psi5yy(1-epsilon,0),
         curv3*ex.psi6x(1-epsilon,0) + ex.psi6yy(1-epsilon,0),
         curv3*ex.psi7x(1-epsilon,0) + ex.psi7yy(1-epsilon,0)],
        [curv2*ex.psi1y(xhi,yhi) + ex.psi1xx(xhi,yhi),
         curv2*ex.psi2y(xhi,yhi) + ex.psi2xx(xhi,yhi),
         curv2*ex.psi3y(xhi,yhi) + ex.psi3xx(xhi,yhi),
         curv2*ex.psi4y(xhi,yhi) + ex.psi4xx(xhi,yhi),
         curv2*ex.psi5y(xhi,yhi) + ex.psi5xx(xhi,yhi),
         curv2*ex.psi6y(xhi,yhi) + ex.psi6xx(xhi,yhi),
         curv2*ex.psi7y(xhi,yhi) + ex.psi7xx(xhi,yhi)],
    ])

    b = -np.array([
        A*ex.psipart1(1+epsilon,0)  + (1-A)*ex.psipart2(1+epsilon,0),
        A*ex.psipart1(1-epsilon,0)  + (1-A)*ex.psipart2(1-epsilon,0),
        A*ex.psipart1(xhi,yhi)      + (1-A)*ex.psipart2(xhi,yhi),
        A*ex.psipart1x(xhi,yhi)     + (1-A)*ex.psipart2x(xhi,yhi),
        A*(curv1*ex.psipart1x(1+epsilon,0) + ex.psipart1yy(1+epsilon,0))
            + (1-A)*(curv1*ex.psipart2x(1+epsilon,0) + ex.psipart2yy(1+epsilon,0)),
        A*(curv3*ex.psipart1x(1-epsilon,0) + ex.psipart1yy(1-epsilon,0))
            + (1-A)*(curv3*ex.psipart2x(1-epsilon,0) + ex.psipart2yy(1-epsilon,0)),
        A*(curv2*ex.psipart1y(xhi,yhi) + ex.psipart1xx(xhi,yhi))
            + (1-A)*(curv2*ex.psipart2y(xhi,yhi) + ex.psipart2xx(xhi,yhi)),
    ])

    C = np.linalg.solve(M, b)

    def psi(x, y):
        return (C[0]*ex.psi1(x,y)  + C[1]*ex.psi2(x,y)  + C[2]*ex.psi3(x,y)  + C[3]*ex.psi4(x,y)
              + C[4]*ex.psi5(x,y)  + C[5]*ex.psi6(x,y)  + C[6]*ex.psi7(x,y)
              + A*ex.psipart1(x,y) + (1-A)*ex.psipart2(x,y))

    return psi, C, A


def psi_value(x, y, epsilon, kappa, delta, A=DEFAULT_A):
    """Evaluate the local JAX flux function."""
    coefficients = solve_coefficients(epsilon, kappa, delta, A)
    return jnp.tensordot(coefficients, basis_values(x, y), axes=(0, 0)) + particular_value(
        x, y, A
    )


def G_total(x, y, A, coefficients):
    """Antiderivative of x * psi used in contour volume averages."""
    log_x = safe_log(x)
    x2 = x * x
    y2 = y * y

    G_base = x**5 * y / 8.0
    G_A = x**3 * y * (-x2 + 4.0 * log_x) / 8.0
    G_1 = x * y
    G_2 = x**3 * y
    G_3 = x * y * (-x2 * log_x + y2 / 3.0)
    G_4 = x**3 * y * (x2 - 4.0 * y2 / 3.0)
    G_5 = x * y * (
        15.0 * x2 * x2 * log_x
        + x2 * y2 * (-20.0 * log_x - 15.0)
        + 2.0 * y2 * y2
    ) / 5.0
    G_6 = x**3 * y * (x2 * x2 - 4.0 * x2 * y2 + 8.0 * y2 * y2 / 5.0)
    G_7 = x * y * (
        -105.0 * x2 * x2 * x2 * log_x
        + x2 * x2 * y2 * (420.0 * log_x + 175.0)
        + x2 * y2 * y2 * (-168.0 * log_x - 196.0)
        + 8.0 * y2 * y2 * y2
    ) / 7.0

    return (
        G_base
        + A * G_A
        + coefficients[0] * G_1
        + coefficients[1] * G_2
        + coefficients[2] * G_3
        + coefficients[3] * G_4
        + coefficients[4] * G_5
        + coefficients[5] * G_6
        + coefficients[6] * G_7
    )


def miller_boundary(shape, point_count=DEFAULT_N):
    """Return points along the closed Miller boundary for one shape."""
    epsilon, kappa, delta = shape
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(point_count) + 1)
    alpha = jnp.arcsin(delta)
    x = 1.0 + epsilon * jnp.cos(theta + alpha * jnp.sin(theta))
    y = kappa * epsilon * jnp.sin(theta)
    return x, y


def _boundary_midpoints(xs, ys):
    x0, x1 = xs[:-1], xs[1:]
    y0, y1 = ys[:-1], ys[1:]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0


def extract_zero_contour(psi, x_lim, y_lim, n: int = DEFAULT_N):
    """
    Extract the zero contour of psi(x, y) as a closed polygon.

    This runs host-side NumPy/contourpy marching squares and is not
    JAX-differentiable.
    """
    x = np.linspace(x_lim[0], x_lim[1], n)
    y = np.linspace(y_lim[0], y_lim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = np.asarray(psi(X, Y), dtype=float)

    gen = contour_generator(x=x, y=y, z=Z)
    lines = [line for line in gen.lines(0.0) if line is not None]

    if not lines:
        raise ValueError("No zero contour found within the specified domain.")

    seg = np.asarray(max(lines, key=len))
    xs, ys = seg[:, 0].copy(), seg[:, 1].copy()

    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])

    signed_area = 0.5 * np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1])
    if signed_area < 0.0:
        xs, ys = xs[::-1], ys[::-1]

    return xs, ys


def int_contour_boundary(G, xs, ys):
    """Compute a Green's-theorem boundary integral on a closed polygon."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x_mid, y_mid, dx = _boundary_midpoints(xs, ys)
    values = G(x_mid, y_mid) if callable(G) else G
    return -float(np.sum(np.asarray(values, dtype=float) * dx))


def poloidal_circum(xs, ys):
    """Approximate the poloidal circumference of a closed boundary."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def _volume_integrand(x, y):
    return x * y


def _q_factor_integrand(A):
    return lambda x, y: y * (A / x + (1.0 - A) * x)


def volume_average(G, xs, ys):
    """Volume average of f using a y-antiderivative of x * f."""
    numerator = int_contour_boundary(G, xs, ys)
    denominator = int_contour_boundary(_volume_integrand, xs, ys)
    return numerator / denominator


def int_masking(psi, epsilon, kappa, A, N):
    """Grid-mask volume-averaged pressure, matching pressure_utils.int_masking."""
    x = np.linspace(1 - epsilon, 1 + epsilon, N)
    y = np.linspace(-kappa * epsilon, kappa * epsilon, N)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dA = dx * dy

    X, Y = np.meshgrid(x, y)
    PSI = np.asarray(psi(X, Y), dtype=float)

    indicator = (PSI <= 0).astype(float)
    P = -(1 - A) * PSI

    numerator = dA * np.sum(X * P * indicator)
    denominator = dA * np.sum(X * indicator)
    return numerator / denominator


def _compute_psi_min(psi, epsilon, kappa, N: int = DEFAULT_N):
    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    x_g = np.linspace(x_lim[0], x_lim[1], N)
    y_g = np.linspace(y_lim[0], y_lim[1], N)
    PSI_g = np.asarray(psi(*np.meshgrid(x_g, y_g)), dtype=float)
    interior = PSI_g[PSI_g <= 0]
    if interior.size == 0:
        raise ValueError("No interior points found; check domain bounds.")
    return float(np.min(interior))


@accepts_shape_params
def get_vol_av_p_from_params(
    epsilon,
    kappa,
    delta,
    A: float = DEFAULT_A,
    method: str = "contour",
    **kwargs,
):
    """Volume-averaged pressure for one shape."""
    N = _validate_N(kwargs.get("N", DEFAULT_N))
    psi, coefficients, _ = _make_psi_numpy(epsilon, kappa, delta, A)

    if method == "contour":
        x_lim, y_lim = _plasma_domain(epsilon, kappa)
        xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
        return -(1.0 - A) * volume_average(
            lambda x, y: _pressure_G_total(x, y, A, coefficients),
            xs,
            ys,
        )
    if method == "masking":
        return int_masking(psi, epsilon, kappa, A, N)
    raise ValueError(f"Unknown method: '{method}'. Choose 'contour' or 'masking'.")


@accepts_shape_params
def normalized_psi_pressure(
    epsilon,
    kappa,
    delta,
    A: float = DEFAULT_A,
    method: str = "contour",
    **kwargs,
):
    """Volume-averaged normalized pressure, psi / psi_min."""
    N = _validate_N(kwargs.get("N", DEFAULT_N))
    psi, coefficients, _ = _make_psi_numpy(epsilon, kappa, delta, A)
    psi_min = _compute_psi_min(psi, epsilon, kappa, N=N)

    if method == "contour":
        x_lim, y_lim = _plasma_domain(epsilon, kappa)
        xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
        return volume_average(
            lambda x, y: _pressure_G_total(x, y, A, coefficients) / psi_min,
            xs,
            ys,
        )
    if method == "masking":
        return int_masking(psi, epsilon, kappa, A, N) / (-(1.0 - A) * psi_min)
    raise ValueError(f"Unknown method: '{method}'. Choose 'contour' or 'masking'.")


@accepts_shape_params
def integral_multiplier(epsilon, kappa, delta, A: float = DEFAULT_A, N: int = DEFAULT_N):
    """Compute the safety-factor integral via the zero-contour boundary."""
    N = _validate_N(N)
    psi, _, _ = _make_psi_numpy(epsilon, kappa, delta, A)
    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
    return int_contour_boundary(_q_factor_integrand(A), xs, ys)


@accepts_shape_params
def inv_q_star(
    epsilon,
    kappa,
    delta,
    A: float = DEFAULT_A,
    R_0: float = 1.0,
    N: int = DEFAULT_N,
):
    """Compute the inverse cylindrical safety factor 1/q*."""
    N = _validate_N(N)
    psi, _, _ = _make_psi_numpy(epsilon, kappa, delta, A)
    psi_min = _compute_psi_min(psi, epsilon, kappa, N=N)
    psi_0_squared = -(R_0**4) / ((1.0 - A) * psi_min)

    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
    circumference = poloidal_circum(xs, ys)
    integral_factor = integral_multiplier(epsilon, kappa, delta, A, N=N)

    B_0 = 2.0
    return -(
        np.sqrt(psi_0_squared) / (epsilon * R_0**2 * B_0)
    ) * integral_factor / circumference


@accepts_shape_params
def beta_t_alternative(epsilon, kappa, delta, A: float = DEFAULT_A, N: int = DEFAULT_N):
    """Alternative toroidal beta using inv_q_star."""
    beta_p = beta_poloidal(epsilon, kappa, delta, A=A, N=N)
    inv_q = inv_q_star(epsilon, kappa, delta, A=A, N=N)
    return epsilon**2 * beta_p * inv_q**2


@accepts_shape_params
def beta_poloidal(epsilon, kappa, delta, A: float = DEFAULT_A, N: int = DEFAULT_N):
    """Poloidal beta from plasma shape via Green's-theorem integration."""
    N = _validate_N(N)
    psi, coefficients, _ = _make_psi_numpy(epsilon, kappa, delta, A)
    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)

    circumference = poloidal_circum(xs, ys)
    volume = int_contour_boundary(_volume_integrand, xs, ys)
    psi_integral = int_contour_boundary(
        lambda x, y: _pressure_G_total(x, y, A, coefficients),
        xs,
        ys,
    )
    factor = int_contour_boundary(_q_factor_integrand(A), xs, ys)

    return -2.0 * (1.0 - A) * (circumference**2 / volume) * psi_integral * factor**-2


@accepts_shape_params
def beta_toroidal(
    epsilon,
    kappa,
    delta,
    A: float = DEFAULT_A,
    q: float = 2.0,
    N: int = DEFAULT_N,
):
    """Toroidal beta from plasma shape and a prescribed safety factor q."""
    beta_p = beta_poloidal(epsilon, kappa, delta, A=A, N=N)
    return epsilon**2 * beta_p / q**2


@accepts_shape_params
def beta_poloidal_updated(
    epsilon,
    kappa,
    delta,
    R_0: float,
    I: float,
    A: float = DEFAULT_A,
    N: int = DEFAULT_N,
):
    """Poloidal beta from physical parameters."""
    p_avg = normalized_psi_pressure(epsilon, kappa, delta, A=A, N=N)
    prefactor = (
        4
        * np.pi**2
        * epsilon**2
        * R_0**2
        * (1 + kappa**2)
        * 1e6
        / (MU_0 * I**2)
    )
    return prefactor * p_avg


@accepts_shape_params
def q_star_updated(epsilon, kappa, delta, R_0: float, I: float, B_0: float):
    """Cylindrical safety factor q* from physical parameters."""
    return (
        2
        * np.pi
        * epsilon**2
        * R_0**2
        * B_0
        * (1 + kappa**2)
        / (2 * MU_0 * R_0 * I)
    )


@accepts_shape_params
def beta_toroidal_updated(
    epsilon,
    kappa,
    delta,
    R_0: float,
    I: float,
    B_0: float,
    A: float = DEFAULT_A,
    N: int = DEFAULT_N,
):
    """Toroidal beta from physical parameters."""
    beta_p = beta_poloidal_updated(epsilon, kappa, delta, R_0, I, A=A, N=N)
    q_star = q_star_updated(epsilon, kappa, delta, R_0, I, B_0)
    return (epsilon**2 * beta_p / q_star**2) * (1 + kappa**2) / 2
