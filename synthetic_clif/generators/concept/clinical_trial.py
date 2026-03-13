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

            # Consent timing (usually within 48 hours of admission)
            consent_time = admit_time + timedelta(
                hours=self.rng.uniform(4, 48)
            )

            # Randomization happens 0-24 hours after consent
            randomized_time = consent_time + timedelta(
                hours=self.rng.uniform(0, 24)
            )

            # ~5% withdraw (those that would have been "Withdrawn")
            withdrawal_time = None
            if self.rng.random() < 0.05:
                withdrawal_time = randomized_time + timedelta(
                    days=self.rng.uniform(1, 14)
                )

            records.append(
                {
                    "hospitalization_id": hosp_id,
                    "trial_id": f"{trial['id_prefix']}-{self.rng.integers(1000, 9999)}",
                    "trial_name": trial["name"],
                    "arm_id": self.rng.choice(["Treatment", "Control", "Placebo"]),
                    "consent_dttm": consent_time,
                    "randomized_dttm": randomized_time,
                    "withdrawal_dttm": withdrawal_time,
                }
            )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["consent_dttm"] = pd.to_datetime(df["consent_dttm"], utc=True)
            df["randomized_dttm"] = pd.to_datetime(df["randomized_dttm"], utc=True)
            df["withdrawal_dttm"] = pd.to_datetime(df["withdrawal_dttm"], utc=True)

        return df
