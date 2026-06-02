"""
psi_anti_deriv_exact.py

Exact y-antiderivatives G_n(x, y) of x * psi_n(x, y), where psi_n are the
basis functions from Cerfon & Freidberg (2010), Eq. (8).

These are used in the Green's theorem reduction of the volume integral:

    iint_Omega x * psi(x, y) dx dy = oint_{partial Omega} G(x, y) dx

where dG/dy = x * psi and G = G_P_base + A * G_P_A + sum_{n=1}^{7} c_n * G_psi_n.

The full flux function is:
    psi(x, y) = P_base + A * P_A + sum_{n=1}^{7} c_n * psi_n(x, y)

where:
    L[P_base] = x^2
    L[P_A]    = 1 - x^2
    L[psi_n]  = 0   (homogeneous GS operator)

so that L[psi] = x^2 + A(1 - x^2) = 1 - Ax^2 + A, matching Eq. (5).

All functions accept numpy arrays x, y and return arrays of the same shape.
"""

import numpy as np


def G_P_base(x, y):
    """G for x * (x^4 / 8);  L[P_base] = x^2"""
    return x**5 * y / 8


def G_P_A(x, y):
    """G for x * (x^2 * ln(x) / 2 - x^4 / 8);  L[P_A] = 1 - x^2"""
    return x**3 * y * (-x**2 + 4 * np.log(x)) / 8


def G_psi1(x, y):
    """G for x * 1"""
    return x * y


def G_psi2(x, y):
    """G for x * x^2"""
    return x**3 * y


def G_psi3(x, y):
    """G for x * (y^2 - x^2 * ln(x))"""
    return x * y * (-x**2 * np.log(x) + y**2 / 3)


def G_psi4(x, y):
    """G for x * (x^4 - 4 * x^2 * y^2)"""
    return x**3 * y * (x**2 - 4 * y**2 / 3)


def G_psi5(x, y):
    """G for x * (2y^4 - 9x^2*y^2 + 3x^4*ln(x) - 12x^2*y^2*ln(x))"""
    return x * y * (
        15 * x**4 * np.log(x)
        + x**2 * y**2 * (-20 * np.log(x) - 15)
        + 2 * y**4
    ) / 5


def G_psi6(x, y):
    """G for x * (x^6 - 12*x^4*y^2 + 8*x^2*y^4)"""
    return x**3 * y * (x**4 - 4 * x**2 * y**2 + 8 * y**4 / 5)


def G_psi7(x, y):
    """G for x * (8y^6 - 140x^2*y^4 + 75x^4*y^2
                  - 15x^6*ln(x) + 180x^4*y^2*ln(x) - 120x^2*y^4*ln(x))"""
    return x * y * (
        -105 * x**6 * np.log(x)
        + x**4 * y**2 * (420 * np.log(x) + 175)
        + x**2 * y**4 * (-168 * np.log(x) - 196)
        + 8 * y**6
    ) / 7


def G_total(x, y, A, c):
    """
    Full y-antiderivative of x * psi(x, y).

    Parameters
    ----------
    x, y : array-like
    A : float
        Solov'ev profile parameter.
    c : array-like, length 7
        Coefficients c[0] ... c[6] for psi_1 ... psi_7.

    Returns
    -------
    G : ndarray  such that dG/dy = x * psi(x, y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)

    return (G_P_base(x, y)
            + A * G_P_A(x, y)
            + c[0] * G_psi1(x, y)
            + c[1] * G_psi2(x, y)
            + c[2] * G_psi3(x, y)
            + c[3] * G_psi4(x, y)
            + c[4] * G_psi5(x, y)
            + c[5] * G_psi6(x, y)
            + c[6] * G_psi7(x, y))