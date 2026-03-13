"""ECMO/MCS (Mechanical Circulatory Support) generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class ECMOMCSGenerator(BaseGenerator):
    """Generate synthetic ECMO/MCS data.

    Creates ecmo_mcs table for patients on extracorporeal support.
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        ecmo_rate: float = 0.02,
    ) -> pd.DataFrame:
        """Generate ECMO/MCS data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            ecmo_rate: Proportion of hospitalizations with ECMO

        Returns:
            DataFrame with ecmo_mcs columns
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

            if self.rng.random() > ecmo_rate:
                continue

            hosp_ecmo = self._generate_hospitalization_ecmo(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_ecmo)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
            # Ensure string columns have correct dtype
            for col in ["device_name", "device_metric_name"]:
                if col in df.columns:
                    df[col] = df[col].astype("string")

        return df

    def _generate_hospitalization_ecmo(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate ECMO data for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # Device categories and their MCS groups per CLIFPy schema
        # Increased ECMO weights to ensure better coverage of sweep/fdO2 fields
        device_options = [
            ("VA_ECMO", "ECMO"),
            ("VV_ECMO", "ECMO"),
            ("IABP", "IABP"),
            ("Impella_CP", "temporary_LVAD"),
            ("Impella_5.5", "temporary_LVAD"),
            ("CentriMag_LV", "temporary_LVAD"),
        ]
        weights = [0.35, 0.35, 0.10, 0.08, 0.07, 0.05]
        idx = self.rng.choice(len(device_options), p=weights)
        device, mcs_group = device_options[idx]

        # ECMO start time
        start_time = admit_time + timedelta(
            hours=self.rng.uniform(0, min(48, los_hours * 0.3))
        )

        # Duration (typically 5-14 days for ECMO)
        duration_hours = min(
            self.rng.uniform(72, 336),
            (discharge_time - start_time).total_seconds() / 3600,
        )
        end_time = start_time + timedelta(hours=duration_hours)

        # Hourly recordings
        timestamps = generate_irregular_timestamps(
            start_time,
            end_time,
            mean_interval_hours=1,
            cv=0.2,
            rng=self.rng,
        )

        for ts in timestamps:
            record = {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": ts,
                "device_name": device.lower(),
                "device_category": device,
                "mcs_group": mcs_group,
                "device_metric_name": None,
                "device_rate": None,
                "sweep": None,
                "flow": None,
                "fdO2": None,
            }

            if mcs_group == "ECMO":
                record["flow"] = round(self.rng.uniform(3, 6), 1)
                record["sweep"] = round(self.rng.uniform(2, 8), 1)
                record["fdO2"] = round(self.rng.uniform(0.5, 1.0), 2)
                record["device_rate"] = round(self.rng.uniform(2500, 4000), 0)
            elif "Impella" in device or "CentriMag" in device:
                record["flow"] = round(self.rng.uniform(2, 5), 1)
                record["device_rate"] = round(self.rng.uniform(30000, 50000), 0)
            elif device == "IABP":
                record["flow"] = round(self.rng.uniform(0.5, 1.5), 1)

            records.append(record)

        return records
