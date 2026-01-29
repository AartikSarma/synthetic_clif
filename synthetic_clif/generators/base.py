"""Abstract base generator with common utilities."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.config.mcide import MCIDELoader


class BaseGenerator(ABC):
    """Abstract base class for CLIF table generators.

    Provides common utilities for introducing realistic data artifacts:
    - Missing values with configurable mechanisms (MCAR, MAR, MNAR)
    - Physiologically plausible outliers
    - Irregular timestamp generation

    Subclasses must implement the generate() method.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        mcide: Optional[MCIDELoader] = None,
    ):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility
            mcide: mCIDE loader instance (shared across generators)
        """
        self.rng = np.random.default_rng(seed)
        self.mcide = mcide or MCIDELoader()
        self._seed = seed

    def _child_seed(self) -> int:
        """Generate a reproducible seed for child generators."""
        return int(self.rng.integers(0, 2**31))

    @abstractmethod
    def generate(self, **kwargs) -> pd.DataFrame:
        """Generate synthetic data for this table.

        Args:
            **kwargs: Table-specific generation parameters

        Returns:
            DataFrame with synthetic data matching CLIF schema
        """
        pass

    def add_missingness(
        self,
        df: pd.DataFrame,
        column: str,
        rate: float,
        mechanism: str = "MCAR",
        conditional_column: Optional[str] = None,
        conditional_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Introduce missing values with specified mechanism.

        Args:
            df: DataFrame to modify
            column: Column to add missingness to
            rate: Base rate of missingness (0-1)
            mechanism: Type of missingness:
                - 'MCAR': Missing completely at random
                - 'MAR': Missing at random (depends on conditional_column)
                - 'MNAR': Missing not at random (depends on own value)
            conditional_column: Column for MAR mechanism
            conditional_threshold: Threshold for conditional missingness

        Returns:
            DataFrame with missing values introduced
        """
        if column not in df.columns:
            return df

        df = df.copy()
        n = len(df)

        if mechanism == "MCAR":
            # Missing completely at random
            mask = self.rng.random(n) < rate

        elif mechanism == "MAR":
            # Missing depends on another observed variable
            if conditional_column is None or conditional_column not in df.columns:
                mask = self.rng.random(n) < rate
            else:
                cond_values = pd.to_numeric(df[conditional_column], errors="coerce")
                if conditional_threshold is None:
                    conditional_threshold = cond_values.median()

                # Higher rate of missingness above threshold
                above_threshold = cond_values > conditional_threshold
                mask = self.rng.random(n) < rate
                mask = mask | (above_threshold & (self.rng.random(n) < rate * 2))

        elif mechanism == "MNAR":
            # Missing depends on own (unobserved) value
            col_values = pd.to_numeric(df[column], errors="coerce")
            if col_values.notna().sum() > 0:
                threshold = col_values.quantile(0.8)
                # Extreme values more likely to be missing
                is_extreme = col_values > threshold
                mask = self.rng.random(n) < rate
                mask = mask | (is_extreme & (self.rng.random(n) < rate * 3))
            else:
                mask = self.rng.random(n) < rate

        else:
            mask = self.rng.random(n) < rate

        df.loc[mask, column] = None
        return df

    def add_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        rate: float,
        method: str = "iqr",
        multiplier: float = 3.0,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> pd.DataFrame:
        """Add physiologically plausible outliers.

        Args:
            df: DataFrame to modify
            column: Numeric column to add outliers to
            rate: Rate of outliers (0-1)
            method: Outlier generation method:
                - 'iqr': Values beyond IQR * multiplier
                - 'shift': Shift by multiplier * std
                - 'extreme': Use lower/upper bounds
            multiplier: Scale factor for outlier magnitude
            lower_bound: Minimum allowed value (physiological limit)
            upper_bound: Maximum allowed value (physiological limit)

        Returns:
            DataFrame with outliers introduced
        """
        if column not in df.columns:
            return df

        df = df.copy()
        col_values = pd.to_numeric(df[column], errors="coerce")
        valid_mask = col_values.notna()

        if valid_mask.sum() == 0:
            return df

        n_valid = valid_mask.sum()
        n_outliers = int(n_valid * rate)

        if n_outliers == 0:
            return df

        # Select indices to modify
        valid_indices = df.index[valid_mask].tolist()
        outlier_indices = self.rng.choice(valid_indices, size=n_outliers, replace=False)

        if method == "iqr":
            q1, q3 = col_values.quantile([0.25, 0.75])
            iqr = q3 - q1
            for idx in outlier_indices:
                if self.rng.random() > 0.5:
                    # High outlier
                    df.loc[idx, column] = q3 + iqr * multiplier * self.rng.uniform(1, 2)
                else:
                    # Low outlier
                    df.loc[idx, column] = q1 - iqr * multiplier * self.rng.uniform(1, 2)

        elif method == "shift":
            mean = col_values.mean()
            std = col_values.std()
            for idx in outlier_indices:
                if self.rng.random() > 0.5:
                    df.loc[idx, column] = mean + std * multiplier * self.rng.uniform(1, 2)
                else:
                    df.loc[idx, column] = mean - std * multiplier * self.rng.uniform(1, 2)

        elif method == "extreme":
            for idx in outlier_indices:
                if self.rng.random() > 0.5 and upper_bound is not None:
                    df.loc[idx, column] = upper_bound * self.rng.uniform(0.9, 1.0)
                elif lower_bound is not None:
                    df.loc[idx, column] = lower_bound * self.rng.uniform(1.0, 1.1)

        # Apply bounds
        if lower_bound is not None:
            df.loc[df[column] < lower_bound, column] = lower_bound
        if upper_bound is not None:
            df.loc[df[column] > upper_bound, column] = upper_bound

        return df

    def sample_category(
        self,
        category: str,
        n: int = 1,
        weights: Optional[list[float]] = None,
    ) -> list[str]:
        """Sample from an mCIDE category.

        Args:
            category: mCIDE category name
            n: Number of samples
            weights: Optional probability weights

        Returns:
            List of sampled category values
        """
        values = self.mcide.get_category(category)
        if not values:
            return ["Unknown"] * n

        if weights is None:
            indices = self.rng.integers(0, len(values), size=n)
        else:
            weights = np.array(weights[: len(values)], dtype=float)
            weights /= weights.sum()
            indices = self.rng.choice(len(values), size=n, p=weights)

        return [values[i] for i in indices]

    def generate_uuid(self) -> str:
        """Generate a UUID-format identifier."""
        return "-".join(
            [
                format(self.rng.integers(0, 0xFFFFFFFF), "08x"),
                format(self.rng.integers(0, 0xFFFF), "04x"),
                format(self.rng.integers(0, 0xFFFF) | 0x4000, "04x"),
                format(self.rng.integers(0, 0xFFFF) | 0x8000, "04x"),
                format(self.rng.integers(0, 0xFFFFFFFFFFFF), "012x"),
            ]
        )

    def generate_uuids(self, n: int) -> list[str]:
        """Generate n UUID-format identifiers."""
        return [self.generate_uuid() for _ in range(n)]
