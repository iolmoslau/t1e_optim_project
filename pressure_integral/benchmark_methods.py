"""
benchmark_methods.py

Benchmarks the contour and masking integration methods against an exact
analytical result.

Integral: iint_Omega x dA  over the Solovev equilibrium domain.
Parameters: eps=0.5, kap=1.0, dlt=0.0  =>  circular cross-section, radius 0.5
            centred at (1, 0).
Exact answer: pi * eps^2 * kap = pi/4  (area * centroid x-coordinate).

Both methods are reduced to computing only this integral so the comparison
is apples-to-apples and the error is meaningful.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.dirname(__file__))

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from pressure_utils import make_psi, extract_zero_contour, int_contour_boundary

EPS, KAP, DLT, A = 0.5, 1.0, 0.0, -0.05
EXACT = np.pi * EPS**2 * KAP          # pi/4
GRID_SIZES = [50, 100, 200, 500, 1000, 2000]


def contour_x_integral(N):
    """iint x dA via Green's theorem: G(x,y) = xy, boundary from marching squares."""
    psi, _, _ = make_psi(EPS, KAP, DLT, A)
    xs, ys = extract_zero_contour(psi, x_lim=(0.4, 1.6), y_lim=(-0.6, 0.6), n=N)
    return int_contour_boundary(lambda x, y: x * y, xs, ys)


def masking_x_integral(N):
    """iint x dA via grid masking: sum X * (psi <= 0) * dA."""
    psi, _, _ = make_psi(EPS, KAP, DLT, A)
    x = np.linspace(1 - EPS - 0.05, 1 + EPS + 0.05, N)
    y = np.linspace(-KAP * EPS - 0.05, KAP * EPS + 0.05, N)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    indicator = (psi(X, Y) <= 0).astype(float)
    return dx * dy * np.sum(X * indicator)


# ── sweep ─────────────────────────────────────────────────────────────────────
print(f"Exact answer: pi/4 = {EXACT:.12f}\n")

rows = []
for N in GRID_SIZES:
    row = {'N': N}
    for label, fn in [('contour', contour_x_integral), ('masking', masking_x_integral)]:
        t0 = time.perf_counter()
        val = fn(N)
        elapsed = time.perf_counter() - t0
        row[f'{label}_val']  = val
        row[f'{label}_err']  = abs(val - EXACT)
        row[f'{label}_time'] = elapsed
        print(f"  N={N:5d}  {label:8s}  val={val:.10f}  "
              f"err={abs(val-EXACT):.2e}  t={elapsed:.3f}s", flush=True)
    rows.append(row)

# ── table ─────────────────────────────────────────────────────────────────────
print()
hdr = (f"{'N':>6} | {'contour val':>14} {'error':>10} {'time':>8} | "
       f"{'masking val':>14} {'error':>10} {'time':>8} | {'Δt (s)':>8}")
sep = '-' * len(hdr)
print(sep)
print(hdr)
print(sep)
for r in rows:
    dt = r['contour_time'] - r['masking_time']
    print(f"{r['N']:>6} | "
          f"{r['contour_val']:>14.10f} {r['contour_err']:>10.2e} {r['contour_time']:>7.3f}s | "
          f"{r['masking_val']:>14.10f} {r['masking_err']:>10.2e} {r['masking_time']:>7.3f}s | "
          f"{dt:>+8.3f}s")
print(sep)
print(f"  exact (pi/4) = {EXACT:.10f}")

# ── plot ──────────────────────────────────────────────────────────────────────
Ns     = [r['N']            for r in rows]
c_err  = [r['contour_err']  for r in rows]
m_err  = [r['masking_err']  for r in rows]
c_time = [r['contour_time'] for r in rows]
m_time = [r['masking_time'] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    r'Contour vs.\ Masking: $\iint_\Omega x\,dA$'
    f'  (ε={EPS}, κ={KAP}, δ={DLT})   exact = π/4', fontsize=12)

ax1.loglog(Ns, c_err, 'o-', color='steelblue', label='contour')
ax1.loglog(Ns, m_err, 's-', color='tomato',    label='masking')
ax1.set_xlabel('Grid size N', fontsize=12)
ax1.set_ylabel('|error|', fontsize=12)
ax1.set_title('Convergence', fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(True, which='both', alpha=0.3)

ax2.loglog(c_time, c_err, 'o-', color='steelblue', label='contour')
ax2.loglog(m_time, m_err, 's-', color='tomato',    label='masking')
for r in rows:
    ax2.annotate(f"N={r['N']}", (r['contour_time'], r['contour_err']),
                 textcoords='offset points', xytext=(5,  3), fontsize=7, color='steelblue')
    ax2.annotate(f"N={r['N']}", (r['masking_time'],  r['masking_err']),
                 textcoords='offset points', xytext=(5, -9), fontsize=7, color='tomato')
ax2.set_xlabel('Wall-clock time (s)', fontsize=12)
ax2.set_ylabel('|error|', fontsize=12)
ax2.set_title('Efficiency frontier  (lower-left = better)', fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'benchmark_methods.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved → {out}')
plt.show()
