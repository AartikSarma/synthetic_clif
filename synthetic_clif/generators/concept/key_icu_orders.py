"""Key ICU orders generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class KeyICUOrdersGenerator(BaseGenerator):
    """Generate synthetic key ICU orders data.

    Creates key_icu_orders table with PT/OT/consult orders.
    """

    ORDERS = {
        "PT Evaluation": {"probability": 0.6, "timing_hours": (12, 72)},
        "OT Evaluation": {"probability": 0.5, "timing_hours": (12, 72)},
        "Speech Evaluation": {"probability": 0.2, "timing_hours": (24, 96)},
        "Nutrition Consult": {"probability": 0.7, "timing_hours": (6, 48)},
        "Palliative Care Consult": {"probability": 0.15, "timing_hours": (48, 168)},
        "Social Work Consult": {"probability": 0.4, "timing_hours": (24, 72)},
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate key ICU orders.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with key_icu_orders columns
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

            for order_cat, params in self.ORDERS.items():
                if self.rng.random() > params["probability"]:
                    continue

                # Order timing
                timing_range = params["timing_hours"]
                max_hours = max(timing_range[0] + 1, min(timing_range[1], los_hours * 0.8))
                if max_hours <= timing_range[0]:
                    continue  # LOS too short for this order
                hours_from_admit = self.rng.uniform(timing_range[0], max_hours)
                order_time = admit_time + timedelta(hours=hours_from_admit)

                if order_time >= discharge_time:
                    continue

                records.append(
                    {
                        "hospitalization_id": hosp_id,
                        "order_dttm": order_time,
                        "order_category": order_cat,
                        "order_name": order_cat,
                        "order_status": self.rng.choice(
                            ["Completed", "Active", "Discontinued"],
                            p=[0.7, 0.25, 0.05],
                        ),
                    }
                )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)

        return df
