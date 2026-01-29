"""Patient table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader


class PatientGenerator(BaseGenerator):
    """Generate synthetic patient demographics.

    Creates patient table with:
    - patient_id (UUID format)
    - sex_category, race_category, ethnicity_category (mCIDE categories)
    - birth_date (realistic age distribution 18-95)
    - death_dttm (~15% mortality, correlated with hospitalizations)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        mcide: Optional[MCIDELoader] = None,
    ):
        super().__init__(seed, mcide)
        self.faker = Faker()
        Faker.seed(seed)

    def generate(
        self,
        n_patients: int,
        mortality_rate: float = 0.15,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate patient demographics.

        Args:
            n_patients: Number of patients to generate
            mortality_rate: Proportion of patients who die (0-1)
            reference_date: Reference date for age/death calculations

        Returns:
            DataFrame with patient table columns
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)

        # Generate patient IDs
        patient_ids = self.generate_uuids(n_patients)

        # Generate demographics
        sex_weights = [0.48, 0.48, 0.02, 0.02]  # Female, Male, Other, Unknown
        sex_categories = self.sample_category("sex", n_patients, sex_weights)

        race_weights = [0.01, 0.06, 0.13, 0.01, 0.70, 0.05, 0.04]
        race_categories = self.sample_category("race", n_patients, race_weights)

        ethnicity_weights = [0.18, 0.78, 0.04]
        ethnicity_categories = self.sample_category("ethnicity", n_patients, ethnicity_weights)

        # Generate birth dates (age distribution typical for ICU)
        # Bimodal: younger trauma/surgical, older medical
        ages = self._generate_age_distribution(n_patients)
        birth_dates = [
            (reference_date - timedelta(days=int(age * 365.25))).date() for age in ages
        ]

        # Generate death dates for those who die
        n_deaths = int(n_patients * mortality_rate)
        death_indices = self.rng.choice(n_patients, size=n_deaths, replace=False)
        death_dttms = [None] * n_patients

        for idx in death_indices:
            # Death occurs within 0-90 days of reference date (will be linked to hospitalization)
            days_until_death = int(self.rng.integers(0, 90))
            death_dttms[idx] = reference_date - timedelta(days=days_until_death)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "sex_category": sex_categories,
                "race_category": race_categories,
                "ethnicity_category": ethnicity_categories,
                "birth_date": birth_dates,
                "death_dttm": death_dttms,
            }
        )

        # Convert datetime columns
        df["birth_date"] = pd.to_datetime(df["birth_date"])
        df["death_dttm"] = pd.to_datetime(df["death_dttm"], utc=True)

        # Add some missingness to demographics (rare)
        df = self.add_missingness(df, "race_category", 0.03)
        df = self.add_missingness(df, "ethnicity_category", 0.02)

        return df

    def _generate_age_distribution(self, n: int) -> np.ndarray:
        """Generate age distribution typical for ICU population.

        Uses mixture of distributions:
        - 20% younger (trauma, surgical): mean 35, std 12
        - 80% older (medical): mean 68, std 15

        Returns ages in years, bounded to [18, 95].
        """
        ages = np.zeros(n)

        # Young cohort (20%)
        n_young = int(n * 0.2)
        ages[:n_young] = self.rng.normal(35, 12, n_young)

        # Older cohort (80%)
        ages[n_young:] = self.rng.normal(68, 15, n - n_young)

        # Shuffle and bound
        self.rng.shuffle(ages)
        return np.clip(ages, 18, 95)
