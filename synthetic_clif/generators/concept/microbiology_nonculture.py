"""Microbiology non-culture test generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class MicrobiologyNoncultureGenerator(BaseGenerator):
    """Generate synthetic non-culture microbiology data.

    Creates microbiology_nonculture table with PCR and rapid tests.
    """

    TESTS = {
        "COVID-19 PCR": {
            "probability": 0.8,
            "positive_rate": 0.15,
        },
        "Influenza PCR": {
            "probability": 0.4,
            "positive_rate": 0.10,
        },
        "RSV PCR": {
            "probability": 0.2,
            "positive_rate": 0.08,
        },
        "Respiratory Viral Panel": {
            "probability": 0.3,
            "positive_rate": 0.25,
        },
        "C. diff Toxin": {
            "probability": 0.25,
            "positive_rate": 0.12,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate non-culture microbiology data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with microbiology_nonculture columns
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

            for test_name, params in self.TESTS.items():
                if self.rng.random() > params["probability"]:
                    continue

                # Test timing (usually early in admission)
                hours_from_admit = self.rng.uniform(0, min(24, los_hours * 0.5))
                order_time = admit_time + timedelta(hours=hours_from_admit)
                collect_time = order_time + timedelta(minutes=int(self.rng.integers(15, 60)))
                result_time = collect_time + timedelta(hours=self.rng.uniform(2, 12))

                # Result
                is_positive = self.rng.random() < params["positive_rate"]
                result_cat = "Positive" if is_positive else "Negative"

                records.append(
                    {
                        "hospitalization_id": hosp_id,
                        "order_dttm": order_time,
                        "collect_dttm": collect_time,
                        "result_dttm": result_time,
                        "test_category": test_name,
                        "test_name": test_name,
                        "result_category": result_cat,
                        "result_value": result_cat,
                    }
                )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)
            df["collect_dttm"] = pd.to_datetime(df["collect_dttm"], utc=True)
            df["result_dttm"] = pd.to_datetime(df["result_dttm"], utc=True)

        return df
