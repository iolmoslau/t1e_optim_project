import numpy as np
from mshr import Polygon, generate_mesh
from fenics import (Point, Mesh, FunctionSpace,
                    DirichletBC, Constant, SubDomain, near,
                    TrialFunction, TestFunction,
                    SpatialCoordinate, dot, grad, dx, Function, solve,
                    assemble_system as _assemble_system)


def create_mesh(epsilon: float, kappa: float, delta: float,
                n_points: int = 100, resolution: int = 32) -> Mesh:
    """
    Create a FEniCS mesh over the Miller-parameterized tokamak cross-section.

    Boundary curve (Miller parametrization):
        x(tau) = 1 + epsilon * cos(tau + alpha * sin(tau))
        y(tau) = epsilon * kappa * sin(tau),   tau in [0, 2*pi)
    where alpha = arcsin(delta).

    Parameters
    ----------
    epsilon    : inverse aspect ratio  (r/R0)
    kappa      : elongation
    delta      : triangularity  (sin(alpha) = delta, |delta| < 1)
    n_points   : number of polygon vertices used to approximate the boundary
    resolution : mshr mesh resolution (higher -> finer mesh)

    Returns
    -------
    mesh : dolfin.Mesh
    """
    if abs(delta) >= 1.0:
        raise ValueError(f"delta must satisfy |delta| < 1, got {delta}")

    alpha = np.arcsin(delta)
    tau = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    x = 1.0 + epsilon * np.cos(tau + alpha * np.sin(tau))
    y = epsilon * kappa * np.sin(tau)

    boundary = [Point(x[i], y[i]) for i in range(n_points)]
    domain = Polygon(boundary)
    mesh = generate_mesh(domain, resolution)

    return mesh


def build_function_spaces(mesh: Mesh):
    """
    Build the CG-1 function space required for the Grad-Shafranov FEM solve.

    Returns
    -------
    V : FunctionSpace   -- scalar CG-1 space for psi (poloidal flux)
    u : TrialFunction on V
    v : TestFunction  on V
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    return V, u, v


def assemble_gs(mesh: Mesh, V: FunctionSpace, u, v, A: float, bc) -> tuple:
    """
    Assemble the stiffness matrix and RHS vector for the Grad-Shafranov equation.

    Equation:
        x d/dx(1/x dpsi/dx) + d²psi/dy² = (1-A)x² + A

    Weak form (divide through by x, then integrate by parts with v=0 on boundary):
        a(psi, v) =  int_Omega  (1/x) grad(psi)·grad(v) dx
        L(v)      = -int_Omega  [(1-A)x + A/x] v dx

    Parameters
    ----------
    mesh : dolfin.Mesh
    V    : FunctionSpace  (CG-1 scalar)
    u    : TrialFunction on V
    v    : TestFunction  on V
    A    : float, Solov'ev free parameter
    bc   : DirichletBC or list of DirichletBC

    Returns
    -------
    K : dolfin.Matrix  (stiffness matrix, BCs applied)
    f : dolfin.Vector  (RHS vector, BCs applied)
    """
    x = SpatialCoordinate(mesh)[0]
    A_c = Constant(A)

    a = (1 / x) * dot(grad(u), grad(v)) * dx
    L = -((1 - A_c) * x + A_c / x) * v * dx

    bcs = bc if isinstance(bc, list) else [bc]
    K, f = _assemble_system(a, L, bcs)
    return K, f


def solve_gs(K, f, V: FunctionSpace) -> Function:
    """
    Solve the assembled linear system K*psi = f.

    Parameters
    ----------
    K : dolfin.Matrix
    f : dolfin.Vector
    V : FunctionSpace  (used to construct the output Function)

    Returns
    -------
    psi : dolfin.Function  (poloidal flux solution)
    """
    psi = Function(V)
    solve(K, psi.vector(), f)
    return psi


class _Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def dirichlet_bc(V: FunctionSpace, g=0.0) -> DirichletBC:
    """
    Define a Dirichlet boundary condition on the entire domain boundary.

    In the Grad-Shafranov context psi is typically set to zero on the last
    closed flux surface, so g defaults to 0.

    Parameters
    ----------
    V : FunctionSpace
        The function space on which the BC is imposed.
    g : float | dolfin.Constant | dolfin.Expression | dolfin.Function
        Boundary value.  A plain float is wrapped in a Constant automatically.

    Returns
    -------
    bc : dolfin.DirichletBC
    """
    if isinstance(g, (int, float)):
        g = Constant(g)
    return DirichletBC(V, g, _Boundary())


if __name__ == "__main__":
    # ITER-like parameters: epsilon=0.32, kappa=1.7, delta=0.33
    mesh = create_mesh(epsilon=0.32, kappa=1.7, delta=0.33)
    print(f"Mesh created: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")
