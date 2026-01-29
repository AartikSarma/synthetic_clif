"""Provider generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class ProviderGenerator(BaseGenerator):
    """Generate synthetic provider assignment data.

    Creates provider table with care team assignments.
    """

    ROLES = {
        "Attending": {"count_range": (1, 2), "probability": 1.0},
        "Resident": {"count_range": (1, 3), "probability": 0.7},
        "Fellow": {"count_range": (0, 1), "probability": 0.3},
        "NP": {"count_range": (0, 2), "probability": 0.5},
        "PA": {"count_range": (0, 1), "probability": 0.3},
        "RN": {"count_range": (2, 6), "probability": 1.0},
        "Pharmacist": {"count_range": (1, 2), "probability": 0.8},
        "Respiratory Therapist": {"count_range": (1, 2), "probability": 0.6},
        "Physical Therapist": {"count_range": (0, 1), "probability": 0.4},
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate provider assignment data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with provider columns
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

            for role, params in self.ROLES.items():
                if self.rng.random() > params["probability"]:
                    continue

                n_providers = self.rng.integers(
                    params["count_range"][0], params["count_range"][1] + 1
                )

                for i in range(n_providers):
                    provider_id = f"{role[:3].upper()}-{self.rng.integers(1000, 9999)}"

                    # Determine assignment duration
                    if role == "RN":
                        # Nurses rotate in shifts
                        shift_hours = 12
                        n_shifts = int(los_hours / shift_hours) + 1
                        for shift in range(min(n_shifts, 3)):  # Limit shifts per nurse
                            start = admit_time + timedelta(hours=shift * shift_hours)
                            end = min(start + timedelta(hours=shift_hours), discharge_time)

                            records.append(
                                {
                                    "hospitalization_id": hosp_id,
                                    "provider_id": f"{provider_id}-{shift}",
                                    "provider_role_category": role,
                                    "start_dttm": start,
                                    "end_dttm": end,
                                }
                            )
                    else:
                        # Other providers assigned for duration
                        records.append(
                            {
                                "hospitalization_id": hosp_id,
                                "provider_id": provider_id,
                                "provider_role_category": role,
                                "start_dttm": admit_time,
                                "end_dttm": discharge_time,
                            }
                        )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["start_dttm"] = pd.to_datetime(df["start_dttm"], utc=True)
            df["end_dttm"] = pd.to_datetime(df["end_dttm"], utc=True)

        return df
