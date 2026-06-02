"""
test_method_2.py  —  int_parametric_boundary tests

Test 1: ITER parameters (eps=0.32, kap=1.7, dlt=0.33, A=-0.05)
Test 2: c=[1,0,...], A=0, eps=1, kap=1, dlt=0  =>  psi = 1 + x^4/8
        analytical: iint x*(1+x^4/8) dA = 97*pi/64
        Note: log(x) terms blow up at x=0; midpoints landing exactly on
        theta=pi (x=0) give NaN at coarse h.
Test 3: f=1, G=xy, eps=1, kap=1, dlt=0
        analytical: iint x dA = pi  (centroid=1, area=pi)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pressure_integral.pressure_utils import make_psi, int_parametric_boundary
from pressure_integral.psi_anti_deriv_exact import G_total

EPS2, KAP2, DLT2 = 0.5, 1.0, 0.0   # circle radius 0.5 at (1,0); leftmost x=0.5, no log singularity


def collect_results():
    results = []

    # ── Test 1: ITER ──────────────────────────────────────────────────────────
    eps, kap, dlt, A = 0.32, 1.7, 0.33, -0.05
    _, c, A = make_psi(eps, kap, dlt, A)
    G_iter = lambda x, y: G_total(x, y, A, c)

    for h in [0.1, 0.01, 0.001]:
        val = int_parametric_boundary(G_iter, eps, kap, dlt, h)
        results.append(dict(case='ITER', method='Parametric',
                            res=f'h={h}', value=val,
                            analytical=None, conv='Spectral'))

    # ── Test 2: psi = 1 + x^4/8 ──────────────────────────────────────────────
    c2 = np.array([1.0, 0, 0, 0, 0, 0, 0])
    G2 = lambda x, y: G_total(x, y, 0.0, c2)
    an2 = 1237.0 * np.pi / 4096.0   # pi/4 + (1/8)*(213*pi/512)

    for h in [0.1, 0.01, 0.001]:
        val = int_parametric_boundary(G2, EPS2, KAP2, DLT2, h)
        results.append(dict(case='psi=1+x^4/8', method='Parametric',
                            res=f'h={h}', value=val,
                            analytical=an2, conv='Spectral'))

    # ── Test 3: f=1, G=xy ─────────────────────────────────────────────────────
    G3 = lambda x, y: x * y

    for h in [0.1, 0.01, 0.001]:
        val = int_parametric_boundary(G3, EPS2, KAP2, DLT2, h)
        results.append(dict(case='f=1, G=xy', method='Parametric',
                            res=f'h={h}', value=val,
                            analytical=np.pi * EPS2**2 * KAP2, conv='Spectral'))

    return results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    for r in collect_results():
        an = f"{r['analytical']:.10f}" if r['analytical'] is not None else '—'
        v  = f"{r['value']:.10f}" if not np.isnan(r['value']) else 'NaN'
        err = '' if r['analytical'] is None or np.isnan(r['value']) \
              else f"  err={abs(r['value']-r['analytical']):.2e}"
        print(f"[{r['case']:<14}] {r['res']}  result={v}  analytical={an}{err}")
