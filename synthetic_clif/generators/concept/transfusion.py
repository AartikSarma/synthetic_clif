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
            "duration_hours": (1, 4),
        },
        "Fresh Frozen Plasma": {
            "probability": 0.08,
            "volume_ml": 250,
            "duration_hours": (0.5, 1),
        },
        "Platelets": {
            "probability": 0.06,
            "volume_ml": 300,
            "duration_hours": (0.5, 1),
        },
        "Cryoprecipitate": {
            "probability": 0.02,
            "volume_ml": 100,
            "duration_hours": (0.5, 1),
        },
    }

    PRODUCT_CODES = {
        "Packed RBCs": "E0027",
        "Fresh Frozen Plasma": "E0032",
        "Platelets": "E0033",
        "Cryoprecipitate": "E0034",
    }

    ATTRIBUTES = ["Irradiated", "Leukoreduced", "CMV Negative", None]

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
                    start_time = admit_time + timedelta(hours=hours_from_admit)

                    if start_time >= discharge_time:
                        continue

                    # Duration based on product type
                    dur_low, dur_high = params["duration_hours"]
                    duration_hours = self.rng.uniform(dur_low, dur_high)
                    end_time = start_time + timedelta(hours=duration_hours)

                    # Attribute (some products have special attributes)
                    attribute = self.rng.choice(self.ATTRIBUTES)

                    records.append(
                        {
                            "hospitalization_id": hosp_id,
                            "transfusion_start_dttm": start_time,
                            "transfusion_end_dttm": end_time,
                            "component_name": product_name,
                            "attribute_name": attribute,
                            "volume_transfused": float(params["volume_ml"]),
                            "volume_units": "mL",
                            "product_code": self.PRODUCT_CODES.get(product_name),
                        }
                    )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["transfusion_start_dttm"] = pd.to_datetime(df["transfusion_start_dttm"], utc=True)
            df["transfusion_end_dttm"] = pd.to_datetime(df["transfusion_end_dttm"], utc=True)

        return df
