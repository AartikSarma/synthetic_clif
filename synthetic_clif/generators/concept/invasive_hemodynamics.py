"""Invasive hemodynamics generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class InvasiveHemodynamicsGenerator(BaseGenerator):
    """Generate synthetic invasive hemodynamic data.

    Creates invasive_hemodynamics table with CVP, PA pressures, cardiac output.
    """

    # Hemodynamic parameters (mean, std, lower, upper)
    HEMO_PARAMS = {
        "CVP": (8, 4, 0, 20),
        "PA Systolic": (25, 8, 15, 60),
        "PA Diastolic": (12, 4, 5, 30),
        "PA Mean": (16, 5, 8, 40),
        "PCWP": (12, 4, 4, 30),
        "Cardiac Output": (5, 1.5, 2, 10),
        "Cardiac Index": (2.8, 0.8, 1.5, 5),
        "SVR": (1100, 300, 500, 2000),
        "PVR": (150, 50, 50, 400),
        "SvO2": (70, 8, 50, 85),
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        pa_catheter_rate: float = 0.08,
    ) -> pd.DataFrame:
        """Generate invasive hemodynamic data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            pa_catheter_rate: Proportion with PA catheters

        Returns:
            DataFrame with invasive_hemodynamics columns
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

            if self.rng.random() > pa_catheter_rate:
                continue

            hosp_hemo = self._generate_hospitalization_hemodynamics(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_hemo)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _generate_hospitalization_hemodynamics(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate hemodynamic data for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # PA catheter duration (typically 2-5 days)
        duration_hours = min(self.rng.uniform(48, 120), los_hours * 0.8)
        end_time = admit_time + timedelta(hours=duration_hours)

        # Measurements every 4-8 hours
        timestamps = generate_irregular_timestamps(
            admit_time,
            end_time,
            mean_interval_hours=6,
            cv=0.3,
            rng=self.rng,
        )

        for ts in timestamps:
            for hemo_cat, params in self.HEMO_PARAMS.items():
                mean, std, lower, upper = params
                value = np.clip(self.rng.normal(mean, std), lower, upper)

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "hemodynamic_category": hemo_cat,
                        "hemodynamic_value": round(value, 1),
                    }
                )

        return records
