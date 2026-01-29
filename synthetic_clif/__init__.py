"""Synthetic CLIF Dataset Generator.

Generate synthetic CLIF (Common Longitudinal ICU data Format) 2.1.0 datasets
with realistic properties including missingness, outliers, irregular measurement
frequency, temporal autocorrelation, and variable hospital length of stay.
"""

__version__ = "0.1.0"

from synthetic_clif.generators.dataset import SyntheticCLIFDataset

__all__ = ["SyntheticCLIFDataset"]
