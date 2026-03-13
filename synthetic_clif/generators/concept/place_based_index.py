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
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate place-based index data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with place_based_index columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]

            for index_name in self.INDEX_TYPES:
                # Generate correlated index values
                # Higher ADI correlates with higher SVI
                base_percentile = self.rng.uniform(0, 100)
                percentile = np.clip(
                    base_percentile + self.rng.normal(0, 15), 1, 100
                )

                # Index value (typically 1-100 for ADI, 0-1 for SVI)
                if index_name == "ADI":
                    value = round(percentile, 0)
                elif index_name == "SVI":
                    value = round(percentile / 100, 3)
                else:
                    value = round(percentile, 1)

                records.append(
                    {
                        "hospitalization_id": hosp_id,
                        "index_name": index_name,
                        "index_value": value,
                        "index_version": self.rng.choice(["2020", "2021"]),
                    }
                )

        return pd.DataFrame(records)
