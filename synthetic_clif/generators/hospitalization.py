"""Hospitalization table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.distributions import log_normal_los


class HospitalizationGenerator(BaseGenerator):
    """Generate synthetic hospitalizations.

    Creates hospitalization table with:
    - hospitalization_id (includes patient prefix for traceability)
    - patient_id (foreign key to patient table)
    - admission_dttm, discharge_dttm (LOS follows log-normal distribution)
    - age_at_admission (computed from patient birth_date)
    - admission_type_category, discharge_category (mCIDE categories)
    """

    def generate(
        self,
        patients_df: pd.DataFrame,
        n_hospitalizations: int,
        reference_date: Optional[datetime] = None,
        median_los_days: float = 5.0,
        los_sigma: float = 0.8,
    ) -> pd.DataFrame:
        """Generate hospitalizations linked to patients.

        Args:
            patients_df: Patient table DataFrame
            n_hospitalizations: Total number of hospitalizations to generate
            reference_date: Reference date for admission times (default: now - 1 year)
            median_los_days: Median length of stay in days
            los_sigma: Log-normal sigma parameter for LOS distribution

        Returns:
            DataFrame with hospitalization table columns
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc) - timedelta(days=365)

        patient_ids = patients_df["patient_id"].tolist()
        birth_dates = patients_df["birth_date"].tolist()
        death_dttms = patients_df["death_dttm"].tolist()

        n_patients = len(patient_ids)

        # Distribute hospitalizations across patients
        # Some patients have multiple admissions, most have 1
        hosp_counts = self._distribute_hospitalizations(n_patients, n_hospitalizations)

        records = []
        for pt_idx, n_hosp in enumerate(hosp_counts):
            if n_hosp == 0:
                continue

            patient_id = patient_ids[pt_idx]
            birth_date = birth_dates[pt_idx]
            death_dttm = death_dttms[pt_idx]

            # Generate admission times spread over 2 years before reference date
            admission_times = self._generate_admission_times(
                n_hosp, reference_date, spread_days=730
            )

            # Generate LOS for each hospitalization
            los_days = log_normal_los(
                n_hosp,
                median_days=median_los_days,
                sigma=los_sigma,
                rng=self.rng,
            )

            for hosp_idx, (admit_time, los) in enumerate(
                zip(admission_times, los_days)
            ):
                # Calculate discharge time
                discharge_time = admit_time + timedelta(days=float(los))

                # Determine discharge category
                is_terminal = (
                    death_dttm is not None
                    and pd.notna(death_dttm)
                    and admit_time <= death_dttm <= discharge_time
                )

                if is_terminal:
                    discharge_category = "Expired"
                    discharge_time = death_dttm
                else:
                    discharge_category = self._sample_discharge_category()

                # Calculate age at admission
                if pd.notna(birth_date):
                    age_at_admission = (
                        admit_time.date() - birth_date.date()
                    ).days / 365.25
                else:
                    age_at_admission = None

                # Generate hospitalization ID (includes patient prefix)
                hosp_id = f"{patient_id[:8]}-H{hosp_idx + 1:03d}"

                records.append(
                    {
                        "hospitalization_id": hosp_id,
                        "patient_id": patient_id,
                        "admission_dttm": admit_time,
                        "discharge_dttm": discharge_time,
                        "age_at_admission": age_at_admission,
                        "admission_type_category": self._sample_admission_type(),
                        "discharge_category": discharge_category,
                    }
                )

        df = pd.DataFrame(records)

        # Ensure datetime columns are UTC
        df["admission_dttm"] = pd.to_datetime(df["admission_dttm"], utc=True)
        df["discharge_dttm"] = pd.to_datetime(df["discharge_dttm"], utc=True)

        # Add missingness
        df = self.add_missingness(df, "age_at_admission", 0.01)
        df = self.add_missingness(df, "admission_type_category", 0.02)

        return df

    def _distribute_hospitalizations(
        self, n_patients: int, n_hospitalizations: int
    ) -> list[int]:
        """Distribute hospitalizations across patients.

        Most patients have 1 admission, some have multiple (readmissions).
        Uses a geometric-like distribution.
        """
        if n_hospitalizations <= n_patients:
            # Each selected patient gets exactly 1
            counts = [0] * n_patients
            selected = self.rng.choice(
                n_patients, size=n_hospitalizations, replace=False
            )
            for idx in selected:
                counts[idx] = 1
            return counts

        # Start with everyone getting 1
        counts = [1] * n_patients
        remaining = n_hospitalizations - n_patients

        # Distribute remaining hospitalizations (readmissions)
        # Probability of readmission decreases geometrically
        while remaining > 0:
            # 30% of patients have readmissions
            n_readmit = min(remaining, int(n_patients * 0.3))
            readmit_indices = self.rng.choice(n_patients, size=n_readmit, replace=True)
            for idx in readmit_indices:
                counts[idx] += 1
                remaining -= 1
                if remaining == 0:
                    break

        return counts

    def _generate_admission_times(
        self,
        n: int,
        reference_date: datetime,
        spread_days: int = 730,
    ) -> list[datetime]:
        """Generate admission times spread over a time period."""
        if reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=timezone.utc)

        admission_times = []
        for _ in range(n):
            days_ago = int(self.rng.integers(0, spread_days))
            hour = int(self.rng.integers(0, 24))
            minute = int(self.rng.integers(0, 60))
            admit_time = reference_date - timedelta(days=days_ago)
            admit_time = admit_time.replace(hour=hour, minute=minute, second=0)
            admission_times.append(admit_time)

        # Sort chronologically for same patient
        return sorted(admission_times)

    def _sample_admission_type(self) -> str:
        """Sample admission type with realistic weights."""
        weights = [0.50, 0.25, 0.20, 0.0, 0.03, 0.01, 0.01]
        return self.sample_category("admission_type", 1, weights)[0]

    def _sample_discharge_category(self) -> str:
        """Sample non-death discharge category."""
        # Weights for non-expired discharges
        weights = [0.55, 0.15, 0.0, 0.05, 0.08, 0.05, 0.02, 0.05, 0.03, 0.02]
        result = self.sample_category("discharge", 1, weights)[0]
        # Avoid "Expired" for non-terminal cases
        if result == "Expired":
            return "Home"
        return result
