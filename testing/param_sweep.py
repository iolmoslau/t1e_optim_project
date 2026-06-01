import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))

import numpy as np
import matplotlib.pyplot as plt
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


def compute_equilibrium(epsilon, kappa, delta, A=-0.05, grid=400):
    """
    Compute the symmetric Solovev equilibrium flux function on a 2-D grid.

    Returns (X, Y, Z, ysep) or raises ValueError if the linear system is
    singular / ill-conditioned for the given parameters.
    """
    alpha = np.arcsin(delta)
    curv1 = -(1 + alpha)**2 / (epsilon * kappa**2)
    curv2 = -kappa / (epsilon * np.cos(alpha)**2)
    curv3 = (1 - alpha)**2 / (epsilon * kappa**2)

    # Boundary-condition matrix (7 symmetric homogeneous basis functions)
    def row(fn, fn_x, fn_y, fn_xx, fn_yy):
        return [
            fn(1 + epsilon, 0),
            fn(1 - epsilon, 0),
            fn(1 - epsilon*delta, kappa*epsilon),
            fn_x(1 - epsilon*delta, kappa*epsilon),
            curv1*fn_x(1 + epsilon, 0)     + fn_yy(1 + epsilon, 0),
            curv3*fn_x(1 - epsilon, 0)     + fn_yy(1 - epsilon, 0),
            curv2*fn_y(1 - epsilon*delta, kappa*epsilon) + fn_xx(1 - epsilon*delta, kappa*epsilon),
        ]

    fns = [
        (psi1, psi1x, psi1y, psi1xx, psi1yy),
        (psi2, psi2x, psi2y, psi2xx, psi2yy),
        (psi3, psi3x, psi3y, psi3xx, psi3yy),
        (psi4, psi4x, psi4y, psi4xx, psi4yy),
        (psi5, psi5x, psi5y, psi5xx, psi5yy),
        (psi6, psi6x, psi6y, psi6xx, psi6yy),
        (psi7, psi7x, psi7y, psi7xx, psi7yy),
    ]

    M = np.column_stack([row(*f) for f in fns])

    b = -np.array([
        A*psipart1(1 + epsilon, 0)                    + (1-A)*psipart2(1 + epsilon, 0),
        A*psipart1(1 - epsilon, 0)                    + (1-A)*psipart2(1 - epsilon, 0),
        A*psipart1(1 - epsilon*delta, kappa*epsilon)  + (1-A)*psipart2(1 - epsilon*delta, kappa*epsilon),
        A*psipart1x(1 - epsilon*delta, kappa*epsilon) + (1-A)*psipart2x(1 - epsilon*delta, kappa*epsilon),
        A*(curv1*psipart1x(1+epsilon,0) + psipart1yy(1+epsilon,0))
            + (1-A)*(curv1*psipart2x(1+epsilon,0) + psipart2yy(1+epsilon,0)),
        A*(curv3*psipart1x(1-epsilon,0) + psipart1yy(1-epsilon,0))
            + (1-A)*(curv3*psipart2x(1-epsilon,0) + psipart2yy(1-epsilon,0)),
        A*(curv2*psipart1y(1-epsilon*delta, kappa*epsilon) + psipart1xx(1-epsilon*delta, kappa*epsilon))
            + (1-A)*(curv2*psipart2y(1-epsilon*delta, kappa*epsilon) + psipart2xx(1-epsilon*delta, kappa*epsilon)),
    ])

    C = np.linalg.solve(M, b)
    C = np.concatenate([C, np.zeros(5)])   # pad for asymmetric basis (unused)
    ysep = -kappa * epsilon

    x = np.linspace(1 - epsilon - 0.05, 1 + epsilon + 0.1, grid)
    y = np.linspace(ysep - 0.05, kappa*epsilon + 0.025, grid)
    X, Y = np.meshgrid(x, y)

    Z = (C[0]*psi1(X,Y)  + C[1]*psi2(X,Y)  + C[2]*psi3(X,Y)  + C[3]*psi4(X,Y)
       + C[4]*psi5(X,Y)  + C[5]*psi6(X,Y)  + C[6]*psi7(X,Y)
       + C[7]*psi8(X,Y)  + C[8]*psi9(X,Y)  + C[9]*psi10(X,Y)
       + C[10]*psi11(X,Y) + C[11]*psi12(X,Y)
       + A*psipart1(X,Y) + (1-A)*psipart2(X,Y))

    return X, Y, Z, ysep


# ── Parameter sweep ──────────────────────────────────────────────────────────
# Each row varies one parameter while the other two are held at the baseline.
# All values are in [0, 1].

BASELINE = dict(epsilon=0.3, kappa=0.7, delta=0.3)

cases = [
    # Row 0 — vary epsilon
    dict(epsilon=0.10, kappa=0.7,  delta=0.3),
    dict(epsilon=0.30, kappa=0.7,  delta=0.3),
    dict(epsilon=0.55, kappa=0.7,  delta=0.3),
    # Row 1 — vary kappa
    dict(epsilon=0.3,  kappa=0.40, delta=0.3),
    dict(epsilon=0.3,  kappa=0.70, delta=0.3),
    dict(epsilon=0.3,  kappa=0.95, delta=0.3),
    # Row 2 — vary delta
    dict(epsilon=0.3,  kappa=0.7,  delta=0.05),
    dict(epsilon=0.3,  kappa=0.7,  delta=0.30),
    dict(epsilon=0.3,  kappa=0.7,  delta=0.65),
]

row_labels = [
    r'Varying $\epsilon$  (κ=0.7, δ=0.3)',
    r'Varying $\kappa$  (ε=0.3, δ=0.3)',
    r'Varying $\delta$  (ε=0.3, κ=0.7)',
]

fig, axes = plt.subplots(3, 3, figsize=(14, 13))
fig.suptitle('Solovev Equilibria — Parameter Comparison', fontsize=15, y=1.01)

for idx, (ax, p) in enumerate(zip(axes.flat, cases)):
    eps, kap, dlt = p['epsilon'], p['kappa'], p['delta']
    try:
        X, Y, Z, ysep = compute_equilibrium(eps, kap, dlt)
    except np.linalg.LinAlgError as e:
        ax.text(0.5, 0.5, f'Singular\n({e})', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.set_title(f'ε={eps}, κ={kap}, δ={dlt}', fontsize=9)
        continue

    # Levels from the magnetic-axis minimum up to the boundary (ψ=0)
    z_min = Z.min()
    levels = np.linspace(z_min, 0, 22)[:-1]   # drop the 0 endpoint to avoid open contour

    tau = np.linspace(0, 2 * np.pi, 500)
    alpha = np.arcsin(dlt)
    bx = 1 + eps * np.cos(tau + alpha * np.sin(tau))
    by = eps * kap * np.sin(tau)

    ax.contour(X, Y, Z, levels=levels, cmap='copper_r')
    ax.contour(X, Y, Z, levels=[0.0], colors='steelblue', linewidths=1.2)  # boundary
    ax.plot(bx, by, 'k:', linewidth=1.2, label='parametric boundary')
    ax.axvline(x=0.0, linestyle='--', color='black', linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$R/R_0$', fontsize=8)
    ax.set_ylabel(r'$Z/R_0$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f'ε={eps}, κ={kap}, δ={dlt}', fontsize=9)

# Add row labels on the left
for row, label in enumerate(row_labels):
    axes[row, 0].annotate(
        label, xy=(-0.35, 0.5), xycoords='axes fraction',
        fontsize=9, rotation=90, va='center', ha='center',
        annotation_clip=False,
    )

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'equilibria_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.show()
