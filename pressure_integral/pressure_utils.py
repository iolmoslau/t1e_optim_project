import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))

import numpy as np
from contourpy import contour_generator
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


def make_psi(epsilon, kappa, delta, A=-0.05):
    """
    Solve the Solovev boundary-value problem for the given shape parameters
    and return a callable psi(x, y) that evaluates the poloidal flux function
    at arbitrary points.

    Parameters
    ----------
    epsilon : float  – inverse aspect ratio
    kappa   : float  – elongation
    delta   : float  – triangularity  (negative values are valid)
    A       : float  – pressure/current profile parameter (default -0.05)

    Returns
    -------
    psi : callable(x, y) -> float or ndarray
        Evaluates ψ at the point(s) (x, y).  x and y may be scalars or
        NumPy arrays of any shape.
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

    return psi


def theta_from_x(epsilon, delta, x_target, upper=True, tol=1e-12, max_iter=50):
    """
    Invert  x = 1 + epsilon * cos(theta + arcsin(delta) * sin(theta))
    for theta using the secant method.

    Parameters
    ----------
    epsilon  : float – inverse aspect ratio
    delta    : float – triangularity
    x_target : float – target x value; must be in [1-epsilon, 1+epsilon]
    upper    : bool  – True  → search theta in [0,   pi]  (upper boundary, y >= 0)
                       False → search theta in [pi, 2*pi] (lower boundary, y <= 0)
    tol      : float – convergence tolerance on |f(theta)|
    max_iter : int   – maximum secant iterations

    Returns
    -------
    theta : float

    Note
    ----
    kappa does not appear in the x-equation so it is not a parameter here.
    The two solutions in [0, 2*pi] correspond to the upper/lower halves of
    the boundary; use the `upper` flag to select which one.
    """
    alpha = np.arcsin(delta)

    def f(theta):
        return 1.0 + epsilon * np.cos(theta + alpha * np.sin(theta)) - x_target

    # Zeroth-order approximation (ignores the alpha correction) as first guess
    ratio = np.clip((x_target - 1.0) / epsilon, -1.0, 1.0)
    theta1 = np.arccos(ratio)          # in [0, pi]
    if not upper:
        theta1 = 2.0 * np.pi - theta1  # mirror into [pi, 2*pi]

    # Second guess: small step away from theta1
    step = 0.05
    theta0 = theta1 + step if theta1 - step < (0 if upper else np.pi) else theta1 - step

    f0 = f(theta0)
    f1 = f(theta1)

    for _ in range(max_iter):
        denom = f1 - f0
        if abs(denom) < 1e-15:
            break
        theta2 = theta1 - f1 * (theta1 - theta0) / denom
        theta0, f0 = theta1, f1
        theta1, f1 = theta2, f(theta2)
        if abs(f1) < tol:
            break

    return theta1




