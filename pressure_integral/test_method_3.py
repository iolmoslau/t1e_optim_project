"""
test_method_3.py  —  int_contour_boundary tests (mirror of test_method_2)

Test 1: ITER — boundary from extract_zero_contour, varied grid resolution n.
Test 2: psi=1+x^4/8, eps=1, kap=1, dlt=0 — boundary built directly from the
        parametric unit-circle polygon (G_total has no log terms for c=[1,0,...],
        A=0, so eps=1 is safe here).
Test 3: f=1, G=xy, same unit-circle polygon.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pressure_integral.pressure_utils import (
    make_psi, extract_zero_contour, int_contour_boundary,
)
from pressure_integral.psi_anti_deriv_exact import G_total

EPS2, KAP2, DLT2 = 0.5, 1.0, 0.0   # circle radius 0.5 at (1,0); leftmost x=0.5, no log singularity


def _circle_polygon(N):
    """Counterclockwise boundary polygon for (EPS2, KAP2, DLT2) with N edges."""
    alpha = np.arcsin(DLT2)
    theta = np.linspace(0, 2 * np.pi, N + 1)
    xs = 1.0 + EPS2 * np.cos(theta + alpha * np.sin(theta))
    ys = EPS2 * KAP2 * np.sin(theta)
    return xs, ys   # naturally CCW for theta 0→2π


def collect_results():
    results = []

    # ── Test 1: ITER — zero contour from marching squares ────────────────────
    eps, kap, dlt, A = 0.32, 1.7, 0.33, -0.05
    psi, c, A = make_psi(eps, kap, dlt, A)
    G_iter = lambda x, y: G_total(x, y, A, c)

    for n in [200, 500, 1000, 2000]:
        xs, ys = extract_zero_contour(psi, x_lim=(0.5, 1.5), y_lim=(-0.7, 0.7), n=n)
        val = int_contour_boundary(G_iter, xs, ys)
        results.append(dict(case='ITER', method='Contour',
                            res=f'n={n}', value=val,
                            analytical=None, conv='O(n⁻²)'))

    # ── Test 2: psi=1+x^4/8, unit-circle polygon ─────────────────────────────
    c2 = np.array([1.0, 0, 0, 0, 0, 0, 0])
    G2 = lambda x, y: G_total(x, y, 0.0, c2)
    an2 = 1237.0 * np.pi / 4096.0   # pi/4 + (1/8)*(213*pi/512)

    for N in [100, 300, 1000, 3000]:
        xs, ys = _circle_polygon(N)
        val = int_contour_boundary(G2, xs, ys)
        results.append(dict(case='psi=1+x^4/8', method='Contour',
                            res=f'N={N}', value=val,
                            analytical=an2, conv='O(N⁻²)'))

    # ── Test 3: f=1, G=xy, unit-circle polygon ────────────────────────────────
    G3 = lambda x, y: x * y

    for N in [100, 300, 1000, 3000]:
        xs, ys = _circle_polygon(N)
        val = int_contour_boundary(G3, xs, ys)
        results.append(dict(case='f=1, G=xy', method='Contour',
                            res=f'N={N}', value=val,
                            analytical=np.pi * EPS2**2 * KAP2, conv='O(N⁻²)'))

    return results


if __name__ == '__main__':
    for r in collect_results():
        an = f"{r['analytical']:.10f}" if r['analytical'] is not None else '—'
        print(f"[{r['case']:<14}] {r['res']}  result={r['value']:.10f}  analytical={an}")
