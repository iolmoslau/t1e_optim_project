import sys
import os
sys.path.insert(0, os.path.dirname(__file__))          # for param_sweep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))

from param_sweep import compute_equilibrium

import numpy as np
import matplotlib.pyplot as plt

# ── Parameter sweep over negative and positive triangularity ─────────────────
# Fixed at ITER-like values; only delta is varied across [-0.6, +0.6].
EPSILON = 0.32
KAPPA   = 1.7
A       = -0.05

deltas = [-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6]

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
fig.suptitle(
    f'Solovev Equilibria — Triangularity Sweep  (ε={EPSILON}, κ={KAPPA})\n'
    r'Blue = $\psi=0$ boundary   |   Copper contours = interior flux surfaces',
    fontsize=12, y=1.01,
)

for ax, delta in zip(axes.flat, deltas):
    try:
        X, Y, Z, ysep = compute_equilibrium(EPSILON, KAPPA, delta, A)
    except (np.linalg.LinAlgError, ValueError) as e:
        ax.text(0.5, 0.5, f'Failed\nδ={delta}\n{e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f'δ = {delta:+.2f}', fontsize=10)
        continue

    levels = np.linspace(Z.min(), 0, 22)[:-1]
    sign_color = '#c0392b' if delta < 0 else ('#2c3e50' if delta == 0 else '#1a6b3c')

    ax.contour(X, Y, Z, levels=levels, cmap='copper_r')
    ax.contour(X, Y, Z, levels=[0.0], colors='steelblue', linewidths=1.5)
    ax.axvline(x=0.0, linestyle='--', color='black', linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$R/R_0$', fontsize=8)
    ax.set_ylabel(r'$Z/R_0$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f'δ = {delta:+.2f}', fontsize=11, color=sign_color, fontweight='bold')

# Row labels
row_text = ['Negative δ', 'Near-zero δ', 'Positive δ']
for row, label in enumerate(row_text):
    axes[row, 0].annotate(
        label, xy=(-0.38, 0.5), xycoords='axes fraction',
        fontsize=9, rotation=90, va='center', ha='center',
        annotation_clip=False,
    )

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'negative_delta_sweep.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.show()
