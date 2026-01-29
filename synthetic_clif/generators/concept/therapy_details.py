"""Therapy details generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class TherapyDetailsGenerator(BaseGenerator):
    """Generate synthetic therapy session details.

    Creates therapy_details table with PT/OT/Speech sessions.
    """

    THERAPIES = {
        "Physical Therapy": {
            "probability": 0.5,
            "sessions_per_day": 1,
            "duration_range": (20, 45),
        },
        "Occupational Therapy": {
            "probability": 0.4,
            "sessions_per_day": 0.5,
            "duration_range": (20, 40),
        },
        "Speech Therapy": {
            "probability": 0.2,
            "sessions_per_day": 0.3,
            "duration_range": (15, 30),
        },
        "Respiratory Therapy": {
            "probability": 0.6,
            "sessions_per_day": 2,
            "duration_range": (10, 30),
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate therapy details.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with therapy_details columns
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

            los_days = (discharge_time - admit_time).total_seconds() / (24 * 3600)

            for therapy_type, params in self.THERAPIES.items():
                if self.rng.random() > params["probability"]:
                    continue

                # Calculate number of sessions
                n_sessions = int(los_days * params["sessions_per_day"])
                if n_sessions == 0 and self.rng.random() < 0.5:
                    n_sessions = 1

                for i in range(n_sessions):
                    # Session timing (usually during day)
                    day_offset = i / max(params["sessions_per_day"], 0.5)
                    session_time = admit_time + timedelta(
                        days=day_offset,
                        hours=self.rng.uniform(8, 16),
                    )

                    if session_time >= discharge_time:
                        continue

                    duration = self.rng.integers(
                        params["duration_range"][0], params["duration_range"][1]
                    )

                    records.append(
                        {
                            "hospitalization_id": hosp_id,
                            "therapy_dttm": session_time,
                            "therapy_category": therapy_type,
                            "therapy_type": self.rng.choice(
                                ["Evaluation", "Treatment", "Re-evaluation"],
                                p=[0.2, 0.7, 0.1] if i > 0 else [0.8, 0.15, 0.05],
                            ),
                            "duration_minutes": float(duration),
                            "provider_id": f"TH-{self.rng.integers(1000, 9999)}",
                        }
                    )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["therapy_dttm"] = pd.to_datetime(df["therapy_dttm"], utc=True)

        return df
