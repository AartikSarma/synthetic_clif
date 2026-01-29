"""Intake/Output generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class IntakeOutputGenerator(BaseGenerator):
    """Generate synthetic intake/output data.

    Creates intake_output table with fluid balance tracking.
    """

    # IO categories and typical volumes
    IO_TYPES = {
        "intake": {
            "IV Fluids": {"volume_range": (50, 1000), "frequency_hours": 4},
            "Blood Products": {"volume_range": (200, 350), "frequency_hours": 24},
            "Enteral Nutrition": {"volume_range": (50, 100), "frequency_hours": 4},
            "PO Intake": {"volume_range": (100, 500), "frequency_hours": 8},
        },
        "output": {
            "Urine Output": {"volume_range": (50, 300), "frequency_hours": 1},
            "Drain Output": {"volume_range": (20, 200), "frequency_hours": 8},
            "Stool": {"volume_range": (100, 500), "frequency_hours": 24},
            "NG Output": {"volume_range": (50, 300), "frequency_hours": 8},
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate intake/output data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with intake_output columns
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

            hosp_io = self._generate_hospitalization_io(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_io)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _generate_hospitalization_io(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate IO data for one hospitalization."""
        records = []

        # Generate intake
        for io_cat, params in self.IO_TYPES["intake"].items():
            # Not all patients have all intake types
            if io_cat == "Blood Products" and self.rng.random() > 0.2:
                continue
            if io_cat == "PO Intake" and self.rng.random() > 0.5:
                continue

            timestamps = generate_irregular_timestamps(
                admit_time,
                discharge_time,
                mean_interval_hours=params["frequency_hours"],
                cv=0.4,
                rng=self.rng,
            )

            for ts in timestamps:
                volume = self.rng.uniform(*params["volume_range"])
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "io_category": io_cat,
                        "io_type": "intake",
                        "volume_ml": round(volume, 0),
                        "fluid_name": self._get_fluid_name(io_cat),
                    }
                )

        # Generate output
        for io_cat, params in self.IO_TYPES["output"].items():
            # Not all patients have all output types
            if io_cat == "Drain Output" and self.rng.random() > 0.2:
                continue
            if io_cat == "NG Output" and self.rng.random() > 0.15:
                continue

            timestamps = generate_irregular_timestamps(
                admit_time,
                discharge_time,
                mean_interval_hours=params["frequency_hours"],
                cv=0.4,
                rng=self.rng,
            )

            for ts in timestamps:
                volume = self.rng.uniform(*params["volume_range"])
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "io_category": io_cat,
                        "io_type": "output",
                        "volume_ml": round(volume, 0),
                        "fluid_name": None,
                    }
                )

        return records

    def _get_fluid_name(self, io_category: str) -> Optional[str]:
        """Get specific fluid name for intake category."""
        if io_category == "IV Fluids":
            return self.rng.choice([
                "Normal Saline",
                "Lactated Ringers",
                "D5W",
                "D5 1/2NS",
            ])
        elif io_category == "Blood Products":
            return self.rng.choice([
                "Packed RBCs",
                "Fresh Frozen Plasma",
                "Platelets",
            ])
        elif io_category == "Enteral Nutrition":
            return self.rng.choice([
                "Jevity",
                "Osmolite",
                "Nepro",
                "Promote",
            ])
        return None
