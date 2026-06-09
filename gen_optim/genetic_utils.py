import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ITER_Equilibria'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pressure_integral'))

import numpy as np
from pressure_utils import get_vol_av_p_from_params, beta_toroidal, normalized_psi_pressure



def initialize_population(N: int, epsilon_range: tuple, kappa_range: tuple, delta_range: tuple) -> np.ndarray:
    """
    Returns population : N x 3 array of members.

    Columns:
        0 -> epsilon
        1 -> kappa
        2 -> delta
    """
    if N <= 0:
        raise ValueError("Population size N must be positive.")

    epsilon_low, epsilon_high = epsilon_range
    kappa_low,   kappa_high   = kappa_range
    delta_low,   delta_high   = delta_range

    epsilons = np.random.uniform(epsilon_low, epsilon_high, N)
    kappas   = np.random.uniform(kappa_low,   kappa_high,   N)
    deltas   = np.random.uniform(delta_low,   delta_high,   N)

    return np.column_stack((epsilons, kappas, deltas))





def evaluate_population(params: np.ndarray, A: float = -0.5,
                         objective: str = 'pressure',
                         method: str = 'contour', **kwargs) -> np.ndarray:
    """
    Evaluate a fitness objective for a population of parameter sets.

    Parameters
    ----------
    params    : ndarray, shape (N, 3)
        Each row is [epsilon, kappa, delta] for one individual.
    A         : float  – Solov'ev profile parameter (default -0.5)
    objective : str    – 'pressure' uses get_vol_av_p_from_params;
                         'beta' uses beta_toroidal;
                         'normalized_psi' uses normalized_psi_pressure
    method    : str    – 'contour' or 'masking' (only used when objective='pressure'
                         or 'normalized_psi')
    **kwargs           – forwarded to the objective function (e.g. N=500, q=2)

    Returns
    -------
    fitnesses : ndarray, shape (N,)
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError(f"params must be shape (N, 3), got {params.shape}")

    if objective == 'pressure':
        return np.asarray(get_vol_av_p_from_params(params, A=A,
                                                   method=method, **kwargs))
    elif objective == 'beta':
        return np.asarray(beta_toroidal(params, A=A, **kwargs))
    elif objective == 'normalized_psi':
        return np.asarray(normalized_psi_pressure(params, A=A,
                                                  method=method, **kwargs))
    else:
        raise ValueError(f"Unknown objective: '{objective}'. "
                         "Choose 'pressure', 'beta', or 'normalized_psi'.")


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


def mutate(offspring: np.ndarray, sigma2: float,
           epsilon_range: tuple | None = None,
           kappa_range:   tuple | None = None,
           delta_range:   tuple | None = None) -> np.ndarray:
    """
    Apply random mutation to a population of offspring.

    For each individual, draw p ~ Uniform(0, 1) and add noise ~ N(0, sigma2)
    to parameter i if p falls in the i-th equal partition of (0, 1):
        i=0 (epsilon) : p in [0,   1/3)
        i=1 (kappa)   : p in [1/3, 2/3)
        i=2 (delta)   : p in [2/3, 1)

    Each individual gets exactly one parameter mutated, chosen with equal
    probability 1/3.  Mutated values are clipped to the supplied ranges.

    Parameters
    ----------
    offspring     : ndarray, shape (N, 3)
    sigma2        : float        – variance of the Gaussian noise
    epsilon_range : (lo, hi)     – allowed range for epsilon (column 0)
    kappa_range   : (lo, hi)     – allowed range for kappa   (column 1)
    delta_range   : (lo, hi)     – allowed range for delta   (column 2)

    Returns
    -------
    mutated : ndarray, shape (N, 3)  – copy of offspring with mutations applied
    """
    mutated = np.array(offspring, dtype=float)
    N = mutated.shape[0]

    p = np.random.uniform(0, 1, size=N)
    i = np.minimum((p * 3).astype(int), 2)
    noise = np.random.normal(0, np.sqrt(sigma2), size=N)

    mutated[np.arange(N), i] += noise

    for col, rng in enumerate([epsilon_range, kappa_range, delta_range]):
        if rng is not None:
            mutated[:, col] = np.clip(mutated[:, col], rng[0], rng[1])

    return mutated
