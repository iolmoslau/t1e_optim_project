import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pressure_integral')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria')))

from fem_utils import create_mesh, build_function_spaces, dirichlet_bc, assemble_gs, solve_gs
from pressure_utils import make_psi

# ITER-like shape parameters
epsilon = 0.32
kappa   = 1.7
delta   = 0.33
A       = -0.05   # Solov'ev profile parameter (make_psi default)

# ── FEM solve ─────────────────────────────────────────────────────────────────
mesh        = create_mesh(epsilon, kappa, delta, n_points=100, resolution=64)
V, u, v     = build_function_spaces(mesh)
bc          = dirichlet_bc(V, 0.0)
K, f        = assemble_gs(mesh, V, u, v, A, bc)
psi_fem     = solve_gs(K, f, V)

coords          = mesh.coordinates()
psi_fem_vals    = psi_fem.compute_vertex_values(mesh)

# ── Solov'ev linear solution ───────────────────────────────────────────────────
psi_lin_fn, _, _ = make_psi(epsilon, kappa, delta, A)
psi_lin_vals     = psi_lin_fn(coords[:, 0], coords[:, 1])

# ── Error metrics ─────────────────────────────────────────────────────────────
err      = psi_fem_vals - psi_lin_vals
l2_err   = np.sqrt(np.mean(err**2))
linf_err = np.max(np.abs(err))

print(f"Vertices : {mesh.num_vertices()},  Cells : {mesh.num_cells()}")
print(f"psi_FEM    range : [{psi_fem_vals.min():.6f}, {psi_fem_vals.max():.6f}]")
print(f"psi_linear range : [{psi_lin_vals.min():.6f}, {psi_lin_vals.max():.6f}]")
print(f"L2  error : {l2_err:.4e}")
print(f"Linf error: {linf_err:.4e}")

# ── Plots ─────────────────────────────────────────────────────────────────────
cells  = mesh.cells()
triang = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)

vmin   = min(psi_fem_vals.min(), psi_lin_vals.min())
levels = np.linspace(vmin, 0.0, 16)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, vals, title in zip(
    axes[:2],
    [psi_fem_vals, psi_lin_vals],
    [r'FEM  $\psi_{\mathrm{FEM}}$', r'Linear  $\psi_{\mathrm{lin}}$'],
):
    tcf = ax.tricontourf(triang, vals, levels=levels, cmap='plasma')
    ax.tricontour(triang, vals, levels=levels, colors='k', linewidths=0.3)
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x  (R/R₀)')
axes[0].set_ylabel('y  (Z/R₀)')

# Signed error with diverging colormap centred at zero
emax  = np.max(np.abs(err))
tcf2  = axes[2].tricontourf(triang, err, levels=16, cmap='RdBu_r',
                             vmin=-emax, vmax=emax)
axes[2].tricontour(triang, err, levels=8, colors='k', linewidths=0.3)
fig.colorbar(tcf2, ax=axes[2], shrink=0.85)
axes[2].set_aspect('equal')
axes[2].set_title(
    f'Error  $\\psi_{{\\mathrm{{FEM}}}} - \\psi_{{\\mathrm{{lin}}}}$\n'
    f'$L_2$={l2_err:.2e},  $L_\\infty$={linf_err:.2e}'
)
axes[2].set_xlabel('x  (R/R₀)')

fig.suptitle(
    f"Solov'ev GS comparison  "
    f"($\\epsilon$={epsilon}, $\\kappa$={kappa}, $\\delta$={delta}, $A$={A})",
    fontsize=13
)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'gs_comparison.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out}")
