import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pressure_integral'))

import numpy as np
from pressure_utils import get_vol_av_p_from_params


def evaluate_population(params: np.ndarray, A: float = -0.5,
                         method: str = 'contour', **kwargs) -> np.ndarray:
    """
    Evaluate the volume-averaged pressure for a population of parameter sets.

    Parameters
    ----------
    params : ndarray, shape (N, 3)
        Each row is [epsilon, kappa, delta] for one individual.
    A      : float  – Solov'ev profile parameter (default -0.5)
    method : str    – 'parametric' or 'contour' (passed to get_vol_av_p_from_params)
    **kwargs        – forwarded to get_vol_av_p_from_params (e.g. h=0.01)

    Returns
    -------
    pressures : ndarray, shape (N,)
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError(f"params must be shape (N, 3), got {params.shape}")

    epsilon = params[:, 0]
    kappa   = params[:, 1]
    delta   = params[:, 2]

    return np.asarray(get_vol_av_p_from_params(epsilon, kappa, delta, A=A,
                                               method=method, **kwargs))


def select_parents(params: np.ndarray, fitnesses: np.ndarray,
                   k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Select N mating pairs via tournament selection.

    For each of the 2N slots, k individuals are sampled at random (without
    replacement) and the one with the highest fitness wins.  The 2N winners
    are split into two (N, 3) arrays so that row i of parents_a and row i of
    parents_b form one mating pair.

    Parameters
    ----------
    params    : ndarray, shape (N, 3)  – population parameter sets
    fitnesses : ndarray, shape (N,)    – fitness value for each individual
    k         : int                    – tournament size (2 <= k <= N)

    Returns
    -------
    parents_a : ndarray, shape (N, 3)
    parents_b : ndarray, shape (N, 3)
    """
    params    = np.asarray(params,    dtype=float)
    fitnesses = np.asarray(fitnesses, dtype=float).ravel()
    N = len(fitnesses)

    if params.shape != (N, 3):
        raise ValueError(f"params must be shape (N, 3), got {params.shape}")
    if not 2 <= k <= N:
        raise ValueError(f"k must satisfy 2 <= k <= N, got k={k}, N={N}")

    def _tournament():
        contestants = np.random.choice(N, size=k, replace=False)
        return contestants[np.argmax(fitnesses[contestants])]

    winners = np.array([_tournament() for _ in range(2 * N)])
    parents_a = params[winners[:N]]
    parents_b = params[winners[N:]]

    return parents_a, parents_b


def crossover(parents_a: np.ndarray, parents_b: np.ndarray,
              method: str = 'uniform', **kwargs) -> np.ndarray:
    """
    Produce N children from N mating pairs.

    Parameters
    ----------
    parents_a : ndarray, shape (N, 3)
    parents_b : ndarray, shape (N, 3)
    method    : 'uniform' | 'arithmetic' | 'blx_alpha'

    kwargs for 'blx_alpha'
    ----------------------
    alpha  : float – extension factor beyond the parents' interval (default 0.5)
    bounds : array-like, shape (3, 2) – [[eps_min, eps_max],
                                          [kap_min, kap_max],
                                          [dlt_min, dlt_max]]
             Children are clipped to these bounds after sampling.

    Returns
    -------
    children : ndarray, shape (N, 3)
    """
    a = np.asarray(parents_a, dtype=float)
    b = np.asarray(parents_b, dtype=float)

    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("parents_a and parents_b must both be shape (N, 3)")

    N = a.shape[0]

    if method == 'uniform':
        # Each gene is drawn independently from one of the two parents.
        mask = np.random.randint(0, 2, size=(N, 3)).astype(bool)
        children = np.where(mask, a, b)

    elif method == 'arithmetic':
        # Each child is a random convex combination of its two parents.
        # One λ per pair, broadcast across all 3 genes.
        lam = np.random.uniform(0, 1, size=(N, 1))
        children = lam * a + (1 - lam) * b

    elif method == 'blx_alpha':
        alpha  = float(kwargs.get('alpha', 0.5))
        bounds = kwargs.get('bounds', None)

        lo_parent = np.minimum(a, b)
        hi_parent = np.maximum(a, b)
        span      = hi_parent - lo_parent

        lo_sample = lo_parent - alpha * span
        hi_sample = hi_parent + alpha * span

        children = np.random.uniform(lo_sample, hi_sample)

        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)   # shape (3, 2)
            lo_bounds = bounds[:, 0]
            hi_bounds = bounds[:, 1]
            children = np.clip(children, lo_bounds, hi_bounds)

    else:
        raise ValueError(f"Unknown crossover method: '{method}'. "
                         "Choose 'uniform', 'arithmetic', or 'blx_alpha'.")

    return children
