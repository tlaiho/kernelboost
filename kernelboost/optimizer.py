"""Simple search based optimization functions."""

import numpy as np
from typing import Callable, Tuple


def _validate_search_params(rounds: int, bounds: list) -> None:
    """Validate common search parameters."""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if bounds[0] <= 0:
        raise ValueError(f"lower bound must be > 0, got {bounds[0]}")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"lower bound must be < upper bound, got {bounds}")


def uniform_search(
        func: Callable,
        t_dependent: np.ndarray,
        t_features: np.ndarray,
        rounds: int,
        initial_precision: np.ndarray,
        bounds: tuple = (0.10, 35.00),
        rng=None,
        mean_y: float = 0.0,
        ) -> Tuple[np.ndarray, float]:
    """Perform random search using uniform distribution across bounds."""
    _validate_search_params(rounds, bounds)
    rng = rng or np.random.default_rng()
    best, best_eval = initial_precision, float(func(t_dependent, t_features, initial_precision, mean_y))

    for k in range(rounds):
        candidate = np.array([rng.uniform(bounds[0], bounds[1])])
        evaluation = func(t_dependent, t_features, candidate, mean_y)

        if evaluation < best_eval:
            best = candidate.copy()
            best_eval = evaluation

    return best, best_eval


def normal_search(
        func: Callable,
        t_dependent: np.ndarray,
        t_features: np.ndarray,
        rounds: int,
        initial_precision: np.ndarray,
        bounds: tuple = (0.10, 35.00),
        rng=None,
        mean_y: float = 0.0,
        ) -> Tuple[np.ndarray, float]:
    """Perform random search around current best using normal distribution."""
    _validate_search_params(rounds, bounds)
    rng = rng or np.random.default_rng()
    half_range = (bounds[1] - bounds[0]) / 2
    center = bounds[0] + half_range
    best, best_eval = initial_precision, float(func(t_dependent, t_features, initial_precision, mean_y))

    for k in range(rounds):
        mean_shift = (center - best) * 0.3
        while True:
            candidate = np.clip(
                np.atleast_1d(best) + mean_shift + rng.normal(0, half_range),
                bounds[0], bounds[1]
            )
            if not np.array_equal(candidate, best):
                break
        evaluation = func(t_dependent, t_features, candidate, mean_y)

        if evaluation < best_eval:
            best = candidate.copy()
            best_eval = evaluation

    return best, best_eval


def optimize_precision(
        func: Callable,
        t_dependent: np.ndarray,
        t_features: np.ndarray,
        optimization_parameters: dict,
        rng=None,
        mean_y: float = 0.0,
        ) -> np.ndarray:
    """Orchestrate precision optimization.

    Supports:
        - "search": LOO-CV with random search with given bounds (default)
        - "silverman": Silverman's rule-of-thumb (fast, no CV)
    """
    method = optimization_parameters.get("precision_method", "search")

    if method == "silverman":
        return silverman_precision(t_features)

    search_rounds = optimization_parameters["search_rounds"]
    bounds = optimization_parameters["bounds"]
    initial_precision = optimization_parameters["initial_precision"]

    if initial_precision == 0:
        init_val = np.atleast_1d((bounds[1] - bounds[0]) / 2)
        best, _ = uniform_search(
            func,
            t_dependent,
            t_features,
            search_rounds * 2,
            initial_precision=init_val,
            bounds=bounds,
            rng=rng,
            mean_y=mean_y,
        )
    else:
        best, _ = normal_search(
            func,
            t_dependent,
            t_features,
            search_rounds,
            initial_precision=np.atleast_1d(initial_precision),
            bounds=bounds,
            rng=rng,
            mean_y=mean_y,
        )

    return best


def grid_search(
        func: Callable,
        initial_values: np.ndarray,
        start: float,
        stop: float,
        step: float,
        **kwargs,
        ) -> Tuple[np.ndarray, float]:
    """Perform grid search by scaling initial values."""
    scale = start
    best = np.array(initial_values).copy()
    best_eval = func(best, **kwargs)

    while scale < stop:
        scale += step
        for index in range(len(initial_values)):
            current = best.copy()
            current[index] = initial_values[index] * scale
            evaluation = func(current, **kwargs)

            if evaluation < best_eval:
                best_eval = evaluation
                best = current.copy()

    return best, best_eval


def estimate_bounds(
        t_features: np.ndarray,
        t_dependent: np.ndarray,
        kernel_type: str,
        bounds: tuple = (0.10, 35.0),
        c_low: float = 0.2,
        c_high: float = 4.25,
        ) -> tuple:
    """Estimate bounds for precision using a polynomial pilot and AMISE formula."""

    y = np.asarray(t_dependent).flatten()
    n, d = t_features.shape

    if d > 75:
        return bounds   

    # fit 1D cubic model for each feature: y = b0 + sum_j(b1j*xj + b2j*xj^2 + b3j*xj^3)  
    design_matrix = np.column_stack([np.ones(n), t_features, t_features**2, t_features**3])
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    residuals = y - design_matrix @ coeffs
    sigma_sq = np.var(residuals)

    if sigma_sq < 1e-20:
        return bounds
    
    # kernel constants
    if kernel_type == "gaussian":
        RK, muK = (np.sqrt(np.pi / 2), np.sqrt(np.pi) / 2)
    elif kernel_type == "laplace":
        RK, muK = (1.0, 4.0)
    else:
        raise ValueError(f"kernel_type must be 'gaussian' or 'laplace', got '{kernel_type}'")

    # second derivatives
    b2 = coeffs[d+1 : 2*d+1]                                                                                       
    b3 = coeffs[2*d+1 : 3*d+1]
    second_deriv = 2 * b2 + 6 * b3 * t_features                                                          
    curvatures = np.clip(np.mean(second_deriv**2, axis=0), 1e-6, None)

    # inverse densities
    m = 50                                                                                                       
    q = np.quantile(t_features, np.linspace(0, 1, m + 1), axis=0) 
    gaps = np.diff(q, axis=0)                             
    inv_densities = m * np.sum(gaps**2, axis=0)   

    # AMISE bandwidths
    C = (RK * sigma_sq * inv_densities / (muK**2 * curvatures)) ** 0.2
    h_opts = C * n ** (-1.0 / (d + 4))
    h_opt = float(np.median(h_opts))

    # Convert bandwidth to precision
    if kernel_type == "gaussian":
        p = 1.0 / h_opt**2
    elif kernel_type == "laplace":
        p = 1.0 / h_opt

    lower = max(c_low * p, bounds[0])
    upper = min(c_high * p, bounds[1])

    if lower >= upper:
        return bounds

    return (lower, upper)



def silverman_precision(t_features: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Scaled Silverman's rule-of-thumb precision. Useful for testing, but
    tends to oversmooth."""
    n, d = t_features.shape
    sigma = np.mean(np.std(t_features, axis=0))
    h = scale * sigma * np.power(n, -1.0 / (d + 4))
    return np.atleast_1d(h)