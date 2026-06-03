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
