"""UTC datetime utilities for CLIF timestamp generation."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def format_utc(dt: datetime) -> str:
    """Format datetime as CLIF-compliant UTC string (YYYY-MM-DD HH:MM:SS+00:00)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")


def random_datetime_in_range(
    start: datetime,
    end: datetime,
    rng: Optional[np.random.Generator] = None,
) -> datetime:
    """Generate a random datetime uniformly distributed between start and end."""
    if rng is None:
        rng = np.random.default_rng()

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    delta = (end - start).total_seconds()
    random_seconds = rng.uniform(0, delta)
    return start + timedelta(seconds=random_seconds)


def generate_irregular_timestamps(
    start: datetime,
    end: datetime,
    mean_interval_hours: float,
    cv: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> list[datetime]:
    """Generate irregularly spaced timestamps between start and end.

    Uses a gamma distribution for inter-arrival times to create realistic
    measurement timing patterns.

    Args:
        start: Start datetime
        end: End datetime
        mean_interval_hours: Mean time between measurements in hours
        cv: Coefficient of variation for interval timing (0.3 = 30% variability)
        rng: Random number generator

    Returns:
        List of datetime objects
    """
    if rng is None:
        rng = np.random.default_rng()

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    total_hours = (end - start).total_seconds() / 3600
    if total_hours <= 0:
        return []

    timestamps = []
    current_hours = 0

    # Gamma distribution parameters from mean and CV
    # shape = 1/cv^2, scale = mean * cv^2
    shape = 1 / (cv * cv) if cv > 0 else 100  # High shape = low variance
    scale = mean_interval_hours * cv * cv if cv > 0 else mean_interval_hours / 100

    while current_hours < total_hours:
        timestamps.append(start + timedelta(hours=current_hours))
        interval = rng.gamma(shape, scale)
        current_hours += max(interval, 0.1)  # Minimum 6 minutes between measurements

    return timestamps


def generate_ordered_timestamps(
    base_time: datetime,
    n_timestamps: int,
    min_gap_minutes: float = 5,
    max_gap_minutes: float = 60,
    rng: Optional[np.random.Generator] = None,
) -> list[datetime]:
    """Generate n ordered timestamps starting from base_time.

    Useful for generating sequences like order -> collect -> result times.

    Args:
        base_time: Starting datetime
        n_timestamps: Number of timestamps to generate
        min_gap_minutes: Minimum gap between timestamps
        max_gap_minutes: Maximum gap between timestamps
        rng: Random number generator

    Returns:
        List of ordered datetime objects
    """
    if rng is None:
        rng = np.random.default_rng()

    if base_time.tzinfo is None:
        base_time = base_time.replace(tzinfo=timezone.utc)

    timestamps = [base_time]
    current = base_time

    for _ in range(n_timestamps - 1):
        gap = rng.uniform(min_gap_minutes, max_gap_minutes)
        current = current + timedelta(minutes=gap)
        timestamps.append(current)

    return timestamps
