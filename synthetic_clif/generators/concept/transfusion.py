"""Transfusion generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class TransfusionGenerator(BaseGenerator):
    """Generate synthetic transfusion data.

    Creates transfusion table with blood product administration.
    """

    PRODUCTS = {
        "Packed RBCs": {
            "probability": 0.20,
            "volume_ml": 300,
            "units": 1,
        },
        "Fresh Frozen Plasma": {
            "probability": 0.08,
            "volume_ml": 250,
            "units": 1,
        },
        "Platelets": {
            "probability": 0.06,
            "volume_ml": 300,
            "units": 1,
        },
        "Cryoprecipitate": {
            "probability": 0.02,
            "volume_ml": 100,
            "units": 5,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate transfusion data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with transfusion columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            los_hours = (discharge_time - admit_time).total_seconds() / 3600

            for product_name, params in self.PRODUCTS.items():
                if self.rng.random() > params["probability"]:
                    continue

                # Number of transfusions (usually 1-4)
                n_transfusions = self.rng.integers(1, 5)

                for i in range(n_transfusions):
                    # Transfusion timing
                    hours_from_admit = self.rng.uniform(0, los_hours * 0.9)
                    transfusion_time = admit_time + timedelta(hours=hours_from_admit)

                    if transfusion_time >= discharge_time:
                        continue

                    records.append(
                        {
                            "hospitalization_id": hosp_id,
                            "transfusion_dttm": transfusion_time,
                            "product_category": product_name,
                            "product_name": product_name,
                            "volume_ml": float(params["volume_ml"]),
                            "units": float(params["units"]),
                        }
                    )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["transfusion_dttm"] = pd.to_datetime(df["transfusion_dttm"], utc=True)

        return df
