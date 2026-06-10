# -*- coding: utf-8 -*-
"""
pressure_utils
==============

Analytic Solov'ev equilibria and plasma diagnostics.

The module solves the Solov'ev boundary-value problem for given plasma shape
parameters (``epsilon``, ``kappa``, ``delta``) and provides volume-averaged
pressure and plasma-beta diagnostics computed via Green's-theorem contour
integration (or a grid-masking fallback).

Calling convention
------------------
Every shape-parameter diagnostic accepts its inputs in three interchangeable
forms:

* three scalars   ``f(epsilon, kappa, delta, ...)``
* three arrays    ``f(eps_array, kap_array, dlt_array, ...)``  (elementwise)
* one ``Nx3`` array ``f(params, ...)`` where ``params[:, 0:3]`` is
  ``[epsilon, kappa, delta]`` per row.

When passing an ``Nx3`` array, supply all other parameters (``A``, ``N``, ``q``,
``R_0`` …) as **keyword** arguments. ``make_psi`` is the one exception: it
returns callables and so only accepts scalar shape parameters.
"""

import functools

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
from contourpy import contour_generator

from psi_anti_deriv_exact import G_total

from ExactSolutions import (
    psi1, psi2, psi3, psi4, psi5, psi6, psi7,
    psi8, psi9, psi10, psi11, psi12,
    psi1x, psi2x, psi3x, psi4x, psi5x, psi6x, psi7x,
    psi1y, psi2y, psi3y, psi4y, psi5y, psi6y, psi7y,
    psi1xx, psi2xx, psi3xx, psi4xx, psi5xx, psi6xx, psi7xx,
    psi1yy, psi2yy, psi3yy, psi4yy, psi5yy, psi6yy, psi7yy,
    psipart1, psipart2,
    psipart1x, psipart2x,
    psipart1y, psipart2y,
    psipart1xx, psipart2xx,
    psipart1yy, psipart2yy,
)

# ── module constants ──────────────────────────────────────────────────────────
MU_0 = 4 * np.pi * 1e-7   # magnetic permeability of free space [T·m/A]
DEFAULT_A = -0.05         # default Solov'ev pressure/current profile parameter

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
    "plot_plasma_profile",
]


# ── internal helpers ──────────────────────────────────────────────────────────
def accepts_shape_params(func):
    """
    Decorator giving a scalar ``func(epsilon, kappa, delta, *args, **kwargs)``
    the standard shape-parameter calling convention: three scalars, three
    arrays (broadcast elementwise), or a single ``Nx3`` array as the first
    argument.

    When the first argument is an ``Nx3`` array, ``kappa`` and ``delta`` must be
    omitted (they are taken from its columns) and every other parameter passed
    by keyword.  Passing a 2-D ``epsilon`` together with explicit ``kappa`` and
    ``delta`` is treated as an elementwise grid evaluation, not as ``Nx3``.
    """
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


def _plasma_domain(epsilon, kappa, pad: float = 0.1):
    """
    Bounding box of the plasma in (x, y), padded by ``pad`` on each side.

    Returns
    -------
    x_lim, y_lim : tuple(float, float)
    """
    x_lim = (1 - epsilon - pad, 1 + epsilon + pad)
    y_lim = (-kappa * epsilon - pad, kappa * epsilon + pad)
    return x_lim, y_lim


def _validate_N(N) -> int:
    """Validate that the grid resolution ``N`` is a positive integer."""
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Grid resolution N must be a positive integer.")
    return N


def _volume_integrand(x, y):
    """y-antiderivative of ``x`` (the plasma cross-section ``∬ x dxdy``)."""
    return x * y


def _q_factor_integrand(A):
    """
    y-antiderivative of ``[A + (1-A)x²]/x``, i.e. ``G(x,y) = y·[A/x + (1-A)x]``,
    used for the safety-factor integral ``∬_Ω [A + (1-A)x²]/x dxdy``.
    """
    return lambda x, y: y * (A / x + (1 - A) * x)


# ── core equilibrium / geometry ───────────────────────────────────────────────
def make_psi(epsilon, kappa, delta, A: float = DEFAULT_A):
    """
    Solve the Solov'ev boundary-value problem for the given shape parameters
    and return a callable psi(x, y) that evaluates the poloidal flux function
    at arbitrary points.

    Only scalar shape parameters are supported (the function returns callables
    and cannot be vectorized).

    Parameters
    ----------
    epsilon : float  – inverse aspect ratio
    kappa   : float  – elongation
    delta   : float  – triangularity  (negative values are valid)
    A       : float  – pressure/current profile parameter (default DEFAULT_A)

    Returns
    -------
    psi : callable(x, y) -> float or ndarray
        Evaluates ψ at the point(s) (x, y).  x and y may be scalars or
        NumPy arrays of any shape.
    c : ndarray, shape (7,)
        Coefficients c[0]…c[6] for the homogeneous basis functions ψ₁…ψ₇.
    A : float
        The profile parameter (echoed back so callers need not track it).
    """
    alpha = np.arcsin(delta)
    curv1 = -(1 + alpha)**2 / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * np.cos(alpha)**2)
    curv3 = (1 - alpha)**2 / (epsilon * kappa**2)

    xhi = 1 - epsilon * delta
    yhi = kappa * epsilon

    M = np.array([
        [psi1(1+epsilon,0), psi2(1+epsilon,0), psi3(1+epsilon,0), psi4(1+epsilon,0), psi5(1+epsilon,0), psi6(1+epsilon,0), psi7(1+epsilon,0)],
        [psi1(1-epsilon,0), psi2(1-epsilon,0), psi3(1-epsilon,0), psi4(1-epsilon,0), psi5(1-epsilon,0), psi6(1-epsilon,0), psi7(1-epsilon,0)],
        [psi1(xhi,yhi),     psi2(xhi,yhi),     psi3(xhi,yhi),     psi4(xhi,yhi),     psi5(xhi,yhi),     psi6(xhi,yhi),     psi7(xhi,yhi)],
        [psi1x(xhi,yhi),    psi2x(xhi,yhi),    psi3x(xhi,yhi),    psi4x(xhi,yhi),    psi5x(xhi,yhi),    psi6x(xhi,yhi),    psi7x(xhi,yhi)],
        [curv1*psi1x(1+epsilon,0) + psi1yy(1+epsilon,0),
         curv1*psi2x(1+epsilon,0) + psi2yy(1+epsilon,0),
         curv1*psi3x(1+epsilon,0) + psi3yy(1+epsilon,0),
         curv1*psi4x(1+epsilon,0) + psi4yy(1+epsilon,0),
         curv1*psi5x(1+epsilon,0) + psi5yy(1+epsilon,0),
         curv1*psi6x(1+epsilon,0) + psi6yy(1+epsilon,0),
         curv1*psi7x(1+epsilon,0) + psi7yy(1+epsilon,0)],
        [curv3*psi1x(1-epsilon,0) + psi1yy(1-epsilon,0),
         curv3*psi2x(1-epsilon,0) + psi2yy(1-epsilon,0),
         curv3*psi3x(1-epsilon,0) + psi3yy(1-epsilon,0),
         curv3*psi4x(1-epsilon,0) + psi4yy(1-epsilon,0),
         curv3*psi5x(1-epsilon,0) + psi5yy(1-epsilon,0),
         curv3*psi6x(1-epsilon,0) + psi6yy(1-epsilon,0),
         curv3*psi7x(1-epsilon,0) + psi7yy(1-epsilon,0)],
        [curv2*psi1y(xhi,yhi) + psi1xx(xhi,yhi),
         curv2*psi2y(xhi,yhi) + psi2xx(xhi,yhi),
         curv2*psi3y(xhi,yhi) + psi3xx(xhi,yhi),
         curv2*psi4y(xhi,yhi) + psi4xx(xhi,yhi),
         curv2*psi5y(xhi,yhi) + psi5xx(xhi,yhi),
         curv2*psi6y(xhi,yhi) + psi6xx(xhi,yhi),
         curv2*psi7y(xhi,yhi) + psi7xx(xhi,yhi)],
    ])

    b = -np.array([
        A*psipart1(1+epsilon,0)  + (1-A)*psipart2(1+epsilon,0),
        A*psipart1(1-epsilon,0)  + (1-A)*psipart2(1-epsilon,0),
        A*psipart1(xhi,yhi)      + (1-A)*psipart2(xhi,yhi),
        A*psipart1x(xhi,yhi)     + (1-A)*psipart2x(xhi,yhi),
        A*(curv1*psipart1x(1+epsilon,0) + psipart1yy(1+epsilon,0))
            + (1-A)*(curv1*psipart2x(1+epsilon,0) + psipart2yy(1+epsilon,0)),
        A*(curv3*psipart1x(1-epsilon,0) + psipart1yy(1-epsilon,0))
            + (1-A)*(curv3*psipart2x(1-epsilon,0) + psipart2yy(1-epsilon,0)),
        A*(curv2*psipart1y(xhi,yhi) + psipart1xx(xhi,yhi))
            + (1-A)*(curv2*psipart2y(xhi,yhi) + psipart2xx(xhi,yhi)),
    ])

    C = np.linalg.solve(M, b)

    def psi(x, y):
        return (C[0]*psi1(x,y)  + C[1]*psi2(x,y)  + C[2]*psi3(x,y)  + C[3]*psi4(x,y)
              + C[4]*psi5(x,y)  + C[5]*psi6(x,y)  + C[6]*psi7(x,y)
              + A*psipart1(x,y) + (1-A)*psipart2(x,y))

    return psi, C, A


def extract_zero_contour(psi, x_lim, y_lim, n: int = 500):
    """
    Extract the zero contour of psi(x, y) as a closed polygon.

    Evaluates psi on an n×n grid over [x_lim, y_lim], runs marching squares
    via contourpy, and returns the longest closed loop (the main plasma
    boundary, ignoring any spurious small loops near the grid edges).

    Parameters
    ----------
    psi   : callable(x, y) -> ndarray  – poloidal flux function
    x_lim : (float, float)             – (x_min, x_max) of the evaluation grid
    y_lim : (float, float)             – (y_min, y_max) of the evaluation grid
    n     : int                        – grid resolution per axis (default 500)

    Returns
    -------
    xs, ys : ndarray, shape (N+1,)
        Closed polygon vertices (first == last point).
    """
    x = np.linspace(x_lim[0], x_lim[1], n)
    y = np.linspace(y_lim[0], y_lim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = psi(X, Y)

    gen = contour_generator(x=x, y=y, z=Z)
    lines = [l for l in gen.lines(0.0) if l is not None]

    if not lines:
        raise ValueError("No zero contour found within the specified domain.")

    # Keep the longest loop — that is the plasma boundary
    seg = np.asarray(max(lines, key=len))
    xs, ys = seg[:, 0].copy(), seg[:, 1].copy()

    # Ensure the polygon is explicitly closed
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])

    # Normalise to counterclockwise orientation (positive shoelace area)
    signed_area = 0.5 * np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1])
    if signed_area < 0:
        xs, ys = xs[::-1], ys[::-1]

    return xs, ys


def int_contour_boundary(G, xs, ys) -> float:
    """
    Compute  ∬_Ω f(x,y) dxdy  via Green's theorem using a piecewise-linear
    zero contour as the boundary, with the midpoint rule on each edge.

        ∬_Ω f dxdy  =  -∮_∂Ω G(x,y) dx
                     ≈  -∑_i  G(x_mid_i, y_mid_i) · Δx_i

    where the sum runs over all edges of the polygon, x_mid and y_mid are the
    edge midpoints, and Δx = x_{i+1} - x_i.

    Parameters
    ----------
    G    : callable(x, y) -> ndarray  – y-antiderivative of f, i.e. ∂G/∂y = f
    xs   : array-like, shape (N+1,)   – x-coordinates of closed polygon vertices
    ys   : array-like, shape (N+1,)   – y-coordinates of closed polygon vertices
                                        (first and last points must coincide)

    Returns
    -------
    integral : float
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x0, x1 = xs[:-1], xs[1:]
    y0, y1 = ys[:-1], ys[1:]

    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)
    dx    = x1 - x0

    return -float(np.sum(G(x_mid, y_mid) * dx))


def poloidal_circum(xs, ys) -> float:
    """
    Approximate the poloidal circumference of a closed polygon.

    Sums the Euclidean lengths of all edges of the polygon, which should be
    the zero contour of psi obtained from extract_zero_contour.

    Parameters
    ----------
    xs : array-like, shape (N+1,)  – x-coordinates of closed polygon vertices
    ys : array-like, shape (N+1,)  – y-coordinates of closed polygon vertices

    Returns
    -------
    circumference : float
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]

    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def volume_average(G, xs, ys) -> float:
    """
    Volume-averaged value over the plasma cross-section,

        <f> = (∬_Ω x f dxdy) / (∬_Ω x dxdy),

    where both integrals are evaluated by Green's theorem on the boundary
    polygon (xs, ys).  ``G`` is the y-antiderivative of ``x f``.
    """
    num   = int_contour_boundary(G, xs, ys)
    denom = int_contour_boundary(_volume_integrand, xs, ys)
    return num / denom


def int_masking(psi, epsilon, kappa, A, N) -> float:
    """
    Volume-averaged pressure by grid masking (no contour extraction).

    Evaluates psi on an N×N grid over the plasma extent, masks the interior
    (psi <= 0), and forms the toroidal volume average of the pressure
    ``p = -(1-A)·psi``:

        <p> = (∑ x p · 1_Ω · dA) / (∑ x · 1_Ω · dA).
    """
    x = np.linspace(1 - epsilon, 1 + epsilon, N)
    y = np.linspace(-kappa * epsilon, kappa * epsilon, N)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dA = dx * dy

    X, Y = np.meshgrid(x, y)
    PSI = psi(X, Y)

    # 1 inside the plasma, 0 outside
    indicator = (PSI <= 0).astype(float)

    # physical pressure (∝ -(1-A)·psi)
    P = -(1 - A) * PSI

    numerator   = dA * np.sum(X * P * indicator)   # ∬_Ω x p dA
    denominator = dA * np.sum(X * indicator)        # ∬_Ω x dA

    return numerator / denominator


def _compute_psi_min(psi, epsilon, kappa, N: int = 500) -> float:
    """
    Return the minimum (most negative) value of psi inside the zero contour.

    Evaluates psi on an NxN grid over the plasma extent with 0.1 padding and
    returns the minimum of all grid points where psi <= 0.
    """
    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    x_g = np.linspace(x_lim[0], x_lim[1], N)
    y_g = np.linspace(y_lim[0], y_lim[1], N)
    PSI_g = psi(*np.meshgrid(x_g, y_g))
    interior = PSI_g[PSI_g <= 0]
    if interior.size == 0:
        raise ValueError("No interior points found; check domain bounds.")
    return float(np.min(interior))


# ── pressure / beta diagnostics ───────────────────────────────────────────────
@accepts_shape_params
def get_vol_av_p_from_params(epsilon, kappa, delta, A: float = DEFAULT_A,
                             method: str = "contour", **kwargs) -> float:
    """
    Volume-averaged pressure <p> = <-(1-A)·psi> for the equilibrium defined by
    the shape parameters.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).

    Parameters
    ----------
    epsilon, kappa, delta : float or array – plasma shape parameters
    A      : float – Solov'ev profile parameter (default DEFAULT_A)
    method : str   – 'contour' (Green's theorem) or 'masking' (grid)
    **kwargs       – N (int, grid resolution, default 500)

    Returns
    -------
    p_avg : float
    """
    N = _validate_N(kwargs.get("N", 500))
    psi, c, _ = make_psi(epsilon, kappa, delta, A)

    if method == "contour":
        x_lim, y_lim = _plasma_domain(epsilon, kappa)
        xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
        G = lambda x, y: G_total(x, y, A, c)
        return -(1 - A) * volume_average(G, xs, ys)
    elif method == "masking":
        return int_masking(psi, epsilon, kappa, A, N)
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'contour' or 'masking'.")


@accepts_shape_params
def normalized_psi_pressure(epsilon, kappa, delta, A: float = DEFAULT_A,
                            method: str = "contour", **kwargs) -> float:
    """
    Volume-averaged normalised pressure: same as get_vol_av_p_from_params but
    integrates  psi / psi_min  instead of  psi, where psi_min is the minimum
    (most negative) value of psi inside the zero contour.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).

    Parameters
    ----------
    epsilon, kappa, delta : float or array – plasma shape parameters
    A      : float – Solov'ev profile parameter (default DEFAULT_A)
    method : str   – 'contour' or 'masking'
    **kwargs       – N (int, grid resolution, default 500)

    Returns
    -------
    p_norm : float
    """
    N = _validate_N(kwargs.get("N", 500))
    psi, c, _ = make_psi(epsilon, kappa, delta, A)
    psi_min = _compute_psi_min(psi, epsilon, kappa, N=N)

    if method == "contour":
        x_lim, y_lim = _plasma_domain(epsilon, kappa)
        xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
        G_norm = lambda x, y: G_total(x, y, A, c) / psi_min
        return volume_average(G_norm, xs, ys)
    elif method == "masking":
        # int_masking returns <-(1-A)·psi>; divide by the same factors to get
        # <psi / psi_min> = <-(1-A)·psi> / (-(1-A)·psi_min).
        return int_masking(psi, epsilon, kappa, A, N) / (-(1 - A) * psi_min)
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'contour' or 'masking'.")


@accepts_shape_params
def integral_multiplier(epsilon, kappa, delta, A: float = DEFAULT_A,
                        N: int = 500) -> float:
    """
    Compute  ∬_Ω [A + (1-A)·x²] / x  dx dy  via the Green's-theorem contour
    method.  The y-antiderivative of the integrand is ``y·[A/x + (1-A)x]``.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).
    """
    psi, _, _ = make_psi(epsilon, kappa, delta, A)
    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)
    return int_contour_boundary(_q_factor_integrand(A), xs, ys)


@accepts_shape_params
def inv_q_star(epsilon, kappa, delta, A: float = DEFAULT_A,
               R_0: float = 1.0, N: int = 500) -> float:
    """
    Compute the inverse cylindrical safety factor 1/q*.

        psi_0_squared = -R_0^4 / ((1-A) * psi_min)
        1/q*          = -(sqrt(psi_0_squared) / (epsilon * R_0^2 * B_0)) * (1/C_p) * integral_factor

    where:
      psi_min         = minimum (magnetic-axis) value of psi inside the zero contour
      C_p             = poloidal circumference of the plasma boundary
      integral_factor = ∬_Ω [A + (1-A)x²]/x dxdy
      B_0             = 2 (fixed normalisation)

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).

    Parameters
    ----------
    epsilon, kappa, delta : float or array – plasma shape parameters
    A   : float – Solov'ev profile parameter (default DEFAULT_A)
    R_0 : float – major radius normalization (default 1.0, cancels in the formula)
    N   : int   – grid resolution (default 500)

    Returns
    -------
    inv_q : float
    """
    psi, _, _ = make_psi(epsilon, kappa, delta, A)

    psi_min       = _compute_psi_min(psi, epsilon, kappa, N=N)
    psi_0_squared = -R_0**4 / ((1 - A) * psi_min)

    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)

    C_p             = poloidal_circum(xs, ys)
    integral_factor = integral_multiplier(epsilon, kappa, delta, A, N=N)

    B_0 = 2
    return -(np.sqrt(psi_0_squared) / (epsilon * R_0**2 * B_0)) * (1 / C_p) * integral_factor


@accepts_shape_params
def beta_t_alternative(epsilon, kappa, delta, A: float = DEFAULT_A,
                       N: int = 500) -> float:
    """
    Alternative toroidal beta built from the shape-only diagnostics:

        beta_t = epsilon^2 * beta_poloidal(...) * inv_q_star(...)^2

    With the B_0 = 2 normalisation used in ``inv_q_star`` this matches
    ``normalized_psi_pressure``.  See ``beta_toroidal`` (uses a prescribed
    safety factor ``q``) and ``beta_toroidal_updated`` (from physical
    ``R_0, I, B_0``) for the other two toroidal-beta definitions.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).
    """
    bp = beta_poloidal(epsilon, kappa, delta, A, N)
    iq = inv_q_star(epsilon, kappa, delta, A, N=N)
    return epsilon**2 * bp * iq**2


@accepts_shape_params
def beta_poloidal(epsilon, kappa, delta, A: float = DEFAULT_A,
                  N: int = 500) -> float:
    """
    Poloidal beta from plasma shape, via the Green's-theorem contour method:

        beta_p = -2 (1-A) (C_p² / V) · psi_integral / factor²

    where C_p is the poloidal circumference, V = ∬_Ω x dxdy, psi_integral is
    ∬_Ω x·psi dxdy, and factor = ∬_Ω [A + (1-A)x²]/x dxdy.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).
    """
    psi, c, _ = make_psi(epsilon, kappa, delta, A)

    x_lim, y_lim = _plasma_domain(epsilon, kappa)
    xs, ys = extract_zero_contour(psi, x_lim, y_lim, n=N)

    circum       = poloidal_circum(xs, ys)
    volume       = int_contour_boundary(_volume_integrand, xs, ys)
    psi_integral = int_contour_boundary(lambda x, y: G_total(x, y, A, c), xs, ys)
    factor       = int_contour_boundary(_q_factor_integrand(A), xs, ys)

    return -2 * (1 - A) * (circum**2 / volume) * psi_integral * factor**(-2)


@accepts_shape_params
def beta_toroidal(epsilon, kappa, delta, A: float = DEFAULT_A,
                  q: float = 2, N: int = 500) -> float:
    """
    Toroidal beta from plasma shape and a prescribed safety factor ``q``:

        beta_t = epsilon^2 * beta_poloidal(...) / q^2

    See ``beta_t_alternative`` (uses inv_q_star instead of a prescribed q) and
    ``beta_toroidal_updated`` (from physical ``R_0, I, B_0``) for the other two
    toroidal-beta definitions.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring).
    """
    bp = beta_poloidal(epsilon, kappa, delta, A, N)
    return epsilon**2 * bp / q**2


# ── physical-parameter API ────────────────────────────────────────────────────
@accepts_shape_params
def beta_poloidal_updated(epsilon, kappa, delta, R_0: float, I: float, P_0:float,
                          A: float = DEFAULT_A, N: int = 500) -> float:
    """
    Poloidal beta from physical parameters:

        beta_p = (4π² ε² R_0² (1 + κ²) / (μ_0 I²)) * <p>

    where <p> = normalized_psi_pressure(ε, κ, δ, A).

    Accepts scalars, arrays, or a single Nx3 array (see module docstring); when
    passing an Nx3 array, pass R_0 and I as keywords.

    Parameters
    ----------
    epsilon, kappa, delta : float or array – plasma shape parameters
    R_0 : float – major radius [m]
    I   : float – plasma current [A]
    A   : float – Solov'ev profile parameter (default DEFAULT_A)
    N   : int   – grid resolution (default 500)
    P_0 : float - Pressure scaling

    Returns
    -------
    beta_p : float
    """
    p_avg = P_0*normalized_psi_pressure(epsilon, kappa, delta, A=A, N=N)
    prefactor = (4 * np.pi**2 * epsilon**2 * R_0**2 * (1 + kappa**2)) / (MU_0 * I**2)
    return prefactor * p_avg


@accepts_shape_params
def q_star_updated(epsilon, kappa, delta, R_0: float, I: float,
                   B_0: float) -> float:
    """
    Cylindrical safety factor q* from physical parameters:

        q* = 2π ε² R_0² B_0 (1 + κ²) / (2 μ_0 R_0 I)

    ``delta`` is unused in this formula (kept for a consistent signature).
    Accepts scalars, arrays, or a single Nx3 array (see module docstring); when
    passing an Nx3 array, pass R_0, I and B_0 as keywords.

    Parameters
    ----------
    epsilon, kappa, delta : float or array – plasma shape parameters
    R_0 : float – major radius [m]
    I   : float – plasma current [A]
    B_0 : float – toroidal magnetic field at the magnetic axis [T]

    Returns
    -------
    q_star : float
    """
    return (2 * np.pi * epsilon**2 * R_0**2 * B_0 * (1 + kappa**2)) / (2 * MU_0 * R_0 * I)


@accepts_shape_params
def beta_toroidal_updated(epsilon, kappa, delta, R_0: float, I: float, P_0: float,
                          B_0: float, A: float = DEFAULT_A, N: int = 500) -> float:
    """
    Toroidal beta from physical parameters:

        beta_t = (ε² * beta_p_updated / q*_updated²) * (1 + κ²) / 2

    See ``beta_toroidal`` (shape-only, prescribed q) and ``beta_t_alternative``
    (via inv_q_star) for the other two toroidal-beta definitions.

    Accepts scalars, arrays, or a single Nx3 array (see module docstring); when
    passing an Nx3 array, pass R_0, I and B_0 as keywords.

    Parameters
    ----------
    epsilon, kappa, delta : float or array – plasma shape parameters
    R_0 : float – major radius [m]
    I   : float – plasma current [A]
    B_0 : float – toroidal magnetic field at the magnetic axis [T]
    A   : float – Solov'ev profile parameter (default DEFAULT_A)
    N   : int   – grid resolution for pressure integral (default 500)

    Returns
    -------
    beta_t : float
    """
    beta_p = beta_poloidal_updated(epsilon, kappa, delta, R_0, I,P_0, A=A, N=N)
    q_star = q_star_updated(epsilon, kappa, delta, R_0, I, B_0)
    return (epsilon**2 * beta_p / q_star**2) * (1 + kappa**2) / 2


def plot_plasma_profile(epsilon, kappa, delta, A: float = DEFAULT_A, N: int = 500,
                        n_levels: int = 30, colorbar: bool = True,
                        title: bool = True, ylabel: bool = True, ax=None):
    """
    Plot psi contours inside the plasma boundary (psi = 0) using the 'plasma'
    colormap.

    Parameters
    ----------
    epsilon  : float – inverse aspect ratio
    kappa    : float – elongation
    delta    : float – triangularity
    A        : float – Solov'ev profile parameter (default DEFAULT_A)
    N        : int   – grid resolution (default 500)
    n_levels : int   – number of contour levels (default 30)
    colorbar : bool  – show continuous colourbar on the right (default True)
    title    : bool  – show shape-parameter title above the plot (default True)
    ax       : matplotlib Axes or None – draw into existing axes if provided

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    psi, _, _ = make_psi(epsilon, kappa, delta, A)
    x_lim, y_lim = _plasma_domain(epsilon, kappa)

    x = np.linspace(*x_lim, N)
    y = np.linspace(*y_lim, N)
    X, Y = np.meshgrid(x, y)
    PSI = psi(X, Y)

    interior = np.where(PSI <= 0, PSI, np.nan)
    levels = np.linspace(-0.05, 0, n_levels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.3))
    else:
        fig = ax.get_figure()

    # Fix the main axes position so it is identical with or without a colorbar.
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.11)

    ax.contour(X, Y, interior, levels=levels, cmap='plasma')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$R/R_0$')
    if ylabel:
        ax.set_ylabel(r'$Z/R_0$')
    else:
        ax.tick_params(axis='y', left=False, labelleft=False)

    if title:
        ax.set_title(rf'$\epsilon={epsilon:.3f},\;\kappa={kappa:.3f},\;\delta={delta:.3f}$',
                     fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))

    if colorbar:
        norm = mcolors.Normalize(vmin=-0.05, vmax=0)
        sm   = cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cax = fig.add_axes((0.79, 0.11, 0.03, 0.82))
        fig.colorbar(sm, cax=cax, label=r'$\psi$')

    return ax
