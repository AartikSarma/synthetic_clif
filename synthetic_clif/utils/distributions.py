"""Statistical distributions for realistic synthetic data generation."""

from typing import Optional

import numpy as np
from scipy import stats


def log_normal_los(
    n: int,
    median_days: float = 5.0,
    sigma: float = 0.8,
    min_days: float = 0.5,
    max_days: float = 90.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate hospital length of stay values from a log-normal distribution.

    ICU LOS typically follows a log-normal distribution with a long right tail.

    Args:
        n: Number of values to generate
        median_days: Median length of stay in days
        sigma: Shape parameter (spread) of the log-normal
        min_days: Minimum LOS (clipped)
        max_days: Maximum LOS (clipped)
        rng: Random number generator

    Returns:
        Array of LOS values in days
    """
    if rng is None:
        rng = np.random.default_rng()

    # mu parameter for log-normal where median = exp(mu)
    mu = np.log(median_days)

    los = rng.lognormal(mu, sigma, n)
    return np.clip(los, min_days, max_days)


def truncated_normal(
    mean: float,
    std: float,
    lower: float,
    upper: float,
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate values from a truncated normal distribution.

    Useful for physiological values that have natural bounds.

    Args:
        mean: Mean of the distribution
        std: Standard deviation
        lower: Lower bound
        upper: Upper bound
        n: Number of values to generate
        rng: Random number generator

    Returns:
        Array of values
    """
    if rng is None:
        rng = np.random.default_rng()

    # Standardize bounds
    a = (lower - mean) / std
    b = (upper - mean) / std

    # Use scipy for truncated normal
    values = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=rng)
    return values


def autoregressive_series(
    n: int,
    initial_value: float,
    mean: float,
    phi: float = 0.8,
    sigma: float = 1.0,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an AR(1) autoregressive time series.

    Creates temporally correlated values that revert to a mean over time.
    Useful for physiological measurements that don't jump abruptly.

    Args:
        n: Number of values to generate
        initial_value: Starting value
        mean: Long-term mean to revert to
        phi: Autoregressive coefficient (0 < phi < 1, higher = more persistence)
        sigma: Innovation standard deviation
        lower: Optional lower bound
        upper: Optional upper bound
        rng: Random number generator

    Returns:
        Array of temporally correlated values
    """
    if rng is None:
        rng = np.random.default_rng()

    series = np.zeros(n)
    series[0] = initial_value

    for t in range(1, n):
        # AR(1) with mean reversion: x_t = mean + phi * (x_{t-1} - mean) + noise
        deviation = series[t - 1] - mean
        series[t] = mean + phi * deviation + rng.normal(0, sigma)

        # Apply bounds if specified
        if lower is not None:
            series[t] = max(series[t], lower)
        if upper is not None:
            series[t] = min(series[t], upper)

    return series


def bimodal_distribution(
    n: int,
    mode1_mean: float,
    mode1_std: float,
    mode2_mean: float,
    mode2_std: float,
    mode1_weight: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate values from a bimodal (mixture of two normals) distribution.

    Useful for variables like temperature that may have normal and fever states.

    Args:
        n: Number of values to generate
        mode1_mean: Mean of first mode
        mode1_std: Std of first mode
        mode2_mean: Mean of second mode
        mode2_std: Std of second mode
        mode1_weight: Probability of drawing from first mode
        rng: Random number generator

    Returns:
        Array of values
    """
    if rng is None:
        rng = np.random.default_rng()

    # Decide which mode for each sample
    from_mode1 = rng.random(n) < mode1_weight

    values = np.zeros(n)
    n_mode1 = from_mode1.sum()
    n_mode2 = n - n_mode1

    values[from_mode1] = rng.normal(mode1_mean, mode1_std, n_mode1)
    values[~from_mode1] = rng.normal(mode2_mean, mode2_std, n_mode2)

    return values


def categorical_with_weights(
    categories: list[str],
    weights: list[float],
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> list[str]:
    """Sample from categorical distribution with specified weights.

    Args:
        categories: List of category values
        weights: Probability weights (will be normalized)
        n: Number of samples
        rng: Random number generator

    Returns:
        List of sampled category values
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    indices = rng.choice(len(categories), size=n, p=weights)
    return [categories[i] for i in indices]
