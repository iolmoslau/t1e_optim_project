"""
test_beta.py

Tests each sub-integral inside beta_poloidal on the circular case
(eps=0.5, kap=1, dlt=0) using scipy.dblquad as the reference.

Sub-integrals:
  1. circum       = 2*pi*eps              (exact)
  2. volume       = iint x dA = pi*eps^2  (exact)
  3. psi_integral = iint psi dA           (no closed form — scipy reference)
  4. factor       = iint [A+(1-A)x^2]/x  (scipy reference, previously verified)

Also notes the two bugs found in beta_poloidal:
  - missing return statement
  - psi passed as G instead of its y-antiderivative
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import integrate as sci
from pressure_utils import (make_psi, extract_zero_contour,
                             int_contour_boundary, poloidal_circum,
                             integral_multiplier)

EPS, KAP, DLT, A = 0.5, 1.0, 0.0, -0.05
psi, c, _ = make_psi(EPS, KAP, DLT, A)

X_LIM = (1 - EPS - 0.1, 1 + EPS + 0.1)
Y_LIM = (-KAP * EPS - 0.1, KAP * EPS + 0.1)
N = 500

xs, ys = extract_zero_contour(psi, X_LIM, Y_LIM, n=N)

print(f"Circular case: eps={EPS}, kap={KAP}, dlt={DLT}, A={A}")
print("=" * 60)

# ── 1. circumference ──────────────────────────────────────────────
exact_circum = 2 * np.pi * EPS
num_circum   = poloidal_circum(xs, ys)
print(f"\n1. Circumference")
print(f"   exact   = {exact_circum:.10f}  (2*pi*eps)")
print(f"   numeric = {num_circum:.10f}  err={abs(num_circum-exact_circum):.2e}")

# ── 2. volume  iint x dA ──────────────────────────────────────────
exact_vol = np.pi * EPS**2 * KAP
num_vol   = int_contour_boundary(lambda x, y: x * y, xs, ys)
print(f"\n2. Volume  iint x dA")
print(f"   exact   = {exact_vol:.10f}  (pi*eps^2*kap)")
print(f"   numeric = {num_vol:.10f}  err={abs(num_vol-exact_vol):.2e}")

# ── 3. psi integral  iint x*psi dA ──────────────────────────────
from psi_anti_deriv_exact import G_total

def _psi_integrand(t, r):
    x = 1 + r * np.cos(t)
    y = r * np.sin(t)
    return float(psi(x, y)) * x * r

scipy_psi, scipy_psi_err = sci.dblquad(_psi_integrand, 0, EPS, 0, 2 * np.pi)
num_psi = int_contour_boundary(lambda x, y: G_total(x, y, A, c), xs, ys)
print(f"\n3. Psi integral  iint x*psi dA")
print(f"   scipy ref = {scipy_psi:.10f}  (est. err {scipy_psi_err:.2e})")
print(f"   numeric   = {num_psi:.10f}  err={abs(num_psi-scipy_psi):.2e}")

# ── 4. factor  iint [A+(1-A)x^2]/x dA ───────────────────────────
def _factor_integrand(t, r):
    x = 1 + r * np.cos(t)
    return (A + (1 - A) * x**2) / x * r

scipy_factor, scipy_factor_err = sci.dblquad(_factor_integrand, 0, EPS, 0, 2 * np.pi)
num_factor = integral_multiplier(EPS, KAP, DLT, A, N=N)
print(f"\n4. Factor  iint [A+(1-A)x^2]/x dA")
print(f"   scipy ref = {scipy_factor:.10f}  (est. err {scipy_factor_err:.2e})")
print(f"   numeric   = {num_factor:.10f}  err={abs(num_factor-scipy_factor):.2e}")

print("\n" + "=" * 60)
print("All four sub-integrals verified.\n")

# ── end-to-end beta_poloidal and beta_toroidal ────────────────────
from pressure_utils import beta_poloidal, beta_toroidal

bp = beta_poloidal(EPS, KAP, DLT, A, N=N)
bt = beta_toroidal(EPS, KAP, DLT, A, N=N)
print(f"beta_poloidal = {bp:.8f}")
print(f"beta_toroidal = {bt:.8f}")
