"""
test_pressure_integral.py

Tests the contour method for the integral

    iint_Omega  [A + (1-A)*x^2] / x  dx dy

The antiderivative w.r.t. y of the integrand f(x,y) = A/x + (1-A)*x is:

    G(x, y) = y * [A/x + (1-A)*x]        (since f has no y-dependence)

Test case: circular boundary, eps=0.5, kap=1, dlt=0.
Exact value from scipy.integrate: 0.782578710843
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pressure_utils import (make_psi, extract_zero_contour, int_contour_boundary)

EPS, KAP, DLT, A = 0.5, 1.0, 0.0, -0.05
EXACT = 0.782578710843   # scipy.integrate reference

def G(x, y):
    return y * (A / x + (1 - A) * x)

psi, _, _ = make_psi(EPS, KAP, DLT, A)

print(f"Integral of [A+(1-A)x^2]/x over circle (eps={EPS}, kap={KAP}, dlt={DLT}, A={A})")
print(f"Exact (scipy) = {EXACT:.12f}\n")

print(f"{'Method':<24} {'Resolution':<12} {'Result':>14} {'Error':>12}")
print('-' * 64)

# ── contour method ────────────────────────────────────────────────────────────
for N in [100, 200, 500, 1000, 2000]:
    xs, ys = extract_zero_contour(psi, x_lim=(0.4, 1.6), y_lim=(-0.6, 0.6), n=N)
    val = int_contour_boundary(G, xs, ys)
    print(f"{'contour':<24} {'N='+str(N):<12} {val:>14.10f} {abs(val-EXACT):>12.2e}")
