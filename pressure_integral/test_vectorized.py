import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pressure_utils import get_vol_av_p_from_params

# ── scalar baseline ───────────────────────────────────────────────────────────
eps0, kap0, dlt0 = 0.32, 1.7, 0.33
scalar = get_vol_av_p_from_params(eps0, kap0, dlt0, method='parametric', h=0.01)
print(f"scalar                     eps={eps0}  kap={kap0}  dlt={dlt0}  →  {scalar:.8f}")

# ── 1-D array of epsilons ─────────────────────────────────────────────────────
epsilons = np.array([0.1, 0.2, 0.32, 0.5, 0.7])
result = get_vol_av_p_from_params(epsilons, kap0, dlt0, method='parametric', h=0.01)
print(f"\n1-D epsilon array (n={len(epsilons)}):")
for e, r in zip(epsilons, result):
    match = " ✓" if abs(e - eps0) < 1e-9 and abs(r - scalar) < 1e-6 else ""
    print(f"  eps={e:.2f}  →  {r:.8f}{match}")

# ── 1-D array of deltas ───────────────────────────────────────────────────────
deltas = np.array([-0.4, -0.2, 0.0, 0.2, 0.33, 0.5])
result = get_vol_av_p_from_params(eps0, kap0, deltas, method='parametric', h=0.01)
print(f"\n1-D delta array (n={len(deltas)}):")
for d, r in zip(deltas, result):
    match = " ✓" if abs(d - dlt0) < 1e-9 and abs(r - scalar) < 1e-6 else ""
    print(f"  dlt={d:+.2f}  →  {r:.8f}{match}")

# ── 1-D array of kappas ───────────────────────────────────────────────────────
kappas = np.array([0.8, 1.2, 1.7, 2.0])
result = get_vol_av_p_from_params(eps0, kappas, dlt0, method='parametric', h=0.01)
print(f"\n1-D kappa array (n={len(kappas)}):")
for k, r in zip(kappas, result):
    match = " ✓" if abs(k - kap0) < 1e-9 and abs(r - scalar) < 1e-6 else ""
    print(f"  kap={k:.1f}  →  {r:.8f}{match}")

# ── all three arrays simultaneously (must be same length) ────────────────────
epsilons3 = np.array([0.2,  0.32, 0.5 ])
kappas3   = np.array([1.2,  1.7,  2.0 ])
deltas3   = np.array([0.1,  0.33, 0.5 ])
result = get_vol_av_p_from_params(epsilons3, kappas3, deltas3, method='parametric', h=0.01)
print(f"\nAll three arrays simultaneously (n={len(epsilons3)}):")
for e, k, d, r in zip(epsilons3, kappas3, deltas3, result):
    print(f"  eps={e:.2f}  kap={k:.1f}  dlt={d:.2f}  →  {r:.8f}")

# ── 2-D grid: epsilon × delta ─────────────────────────────────────────────────
eps_grid = np.array([0.2, 0.32, 0.5])
dlt_grid = np.array([-0.2, 0.0, 0.33])
EPS, DLT = np.meshgrid(eps_grid, dlt_grid)
result2d = get_vol_av_p_from_params(EPS, kap0, DLT, method='parametric', h=0.01)
print(f"\n2-D grid  epsilon({len(eps_grid)}) × delta({len(dlt_grid)})  →  shape {result2d.shape}:")
print("          ", "  ".join(f"eps={e:.2f}" for e in eps_grid))
for i, d in enumerate(dlt_grid):
    row = "  ".join(f"{result2d[i,j]:.6f}" for j in range(len(eps_grid)))
    print(f"  dlt={d:+.2f}  {row}")
