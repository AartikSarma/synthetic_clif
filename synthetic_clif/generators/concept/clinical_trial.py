"""Clinical trial generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class ClinicalTrialGenerator(BaseGenerator):
    """Generate synthetic clinical trial enrollment data.

    Creates clinical_trial table with ~5% of patients enrolled in trials.
    """

    # Sample clinical trials
    TRIALS = [
        {"name": "ARDS-NET Low Tidal Volume", "id_prefix": "ARDS"},
        {"name": "PROVENT Ventilator Weaning", "id_prefix": "PROV"},
        {"name": "VITAMINS Sepsis Treatment", "id_prefix": "VITA"},
        {"name": "CLASSIC Fluid Therapy", "id_prefix": "CLAS"},
        {"name": "ARISE Resuscitation", "id_prefix": "ARIS"},
    ]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        enrollment_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Generate clinical trial enrollment data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            enrollment_rate: Proportion enrolled in trials

        Returns:
            DataFrame with clinical_trial columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]

            if pd.isna(admit_time):
                continue

            if self.rng.random() > enrollment_rate:
                continue

            # Select trial
            trial = self.rng.choice(self.TRIALS)

            # Enrollment timing (usually within 48 hours of admission)
            enroll_time = admit_time + timedelta(
                hours=self.rng.uniform(4, 48)
            )

            records.append(
                {
                    "hospitalization_id": hosp_id,
                    "trial_id": f"{trial['id_prefix']}-{self.rng.integers(1000, 9999)}",
                    "trial_name": trial["name"],
                    "enrollment_dttm": enroll_time,
                    "enrollment_status": self.rng.choice(
                        ["Enrolled", "Screen Failure", "Withdrawn"],
                        p=[0.80, 0.15, 0.05],
                    ),
                }
            )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["enrollment_dttm"] = pd.to_datetime(df["enrollment_dttm"], utc=True)

        return df
