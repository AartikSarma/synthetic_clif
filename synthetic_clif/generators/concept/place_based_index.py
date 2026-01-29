"""Place-based index generator."""

from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class PlaceBasedIndexGenerator(BaseGenerator):
    """Generate synthetic place-based index data.

    Creates place_based_index table with ADI, SVI, and other indices.
    """

    INDEX_TYPES = ["ADI", "SVI", "NDI"]

    def generate(
        self,
        patients_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate place-based index data.

        Args:
            patients_df: Patient table DataFrame

        Returns:
            DataFrame with place_based_index columns
        """
        records = []

        for _, patient in patients_df.iterrows():
            patient_id = patient["patient_id"]

            for index_type in self.INDEX_TYPES:
                # Generate correlated index values
                # Higher ADI correlates with higher SVI
                base_percentile = self.rng.uniform(0, 100)
                percentile = np.clip(
                    base_percentile + self.rng.normal(0, 15), 1, 100
                )

                # Index value (typically 1-100 for ADI, 0-1 for SVI)
                if index_type == "ADI":
                    value = round(percentile, 0)
                elif index_type == "SVI":
                    value = round(percentile / 100, 3)
                else:
                    value = round(percentile, 1)

                records.append(
                    {
                        "patient_id": patient_id,
                        "index_type": index_type,
                        "index_value": value,
                        "index_percentile": round(percentile, 1),
                        "geography_type": "Census Tract",
                        "geography_code": f"{self.rng.integers(1, 56):02d}"
                        f"{self.rng.integers(1, 999):03d}"
                        f"{self.rng.integers(100000, 999999):06d}",
                    }
                )

        return pd.DataFrame(records)
