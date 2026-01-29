"""Vitals table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.models.patient_state import PatientState
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class VitalsGenerator(BaseGenerator):
    """Generate synthetic vital signs.

    Creates vitals table with:
    - hospitalization_id (foreign key)
    - recorded_dttm (irregular timestamps, ~hourly in ICU, ~q4h elsewhere)
    - vital_category (mCIDE: temp_c, heart_rate, sbp, dbp, spo2, etc.)
    - vital_value (realistic ranges with temporal autocorrelation)
    - meas_site_category (measurement site)

    Features:
    - Temporal autocorrelation (values don't jump unrealistically)
    - Measurement frequency varies by location (ICU vs ward)
    - ~5% random missingness, ~1% outliers
    """

    # Vital sign parameters: (mean, std, lower_bound, upper_bound)
    VITAL_PARAMS = {
        "temp_c": (37.0, 0.5, 34.0, 42.0),
        "heart_rate": (80.0, 15.0, 30.0, 200.0),
        "sbp": (120.0, 20.0, 60.0, 250.0),
        "dbp": (75.0, 12.0, 30.0, 150.0),
        "spo2": (96.0, 3.0, 50.0, 100.0),
        "respiratory_rate": (16.0, 4.0, 6.0, 50.0),
        "map": (90.0, 15.0, 40.0, 160.0),
        "height_cm": (170.0, 10.0, 140.0, 210.0),
        "weight_kg": (80.0, 20.0, 40.0, 200.0),
    }

    # Measurement frequency in hours by location
    FREQUENCY_BY_LOCATION = {
        "icu": 1.0,
        "ed": 1.0,
        "stepdown": 2.0,
        "ward": 4.0,
        "other": 4.0,
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        adt_df: Optional[pd.DataFrame] = None,
        missingness_rate: float = 0.05,
        outlier_rate: float = 0.01,
    ) -> pd.DataFrame:
        """Generate vital signs for hospitalizations.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            adt_df: Optional ADT table for location-based frequency
            missingness_rate: Proportion of missing values
            outlier_rate: Proportion of outlier values

        Returns:
            DataFrame with vitals table columns
        """
        records = []

        # Build location timeline lookup
        location_lookup = self._build_location_lookup(adt_df)

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            # Generate vitals for this hospitalization
            hosp_vitals = self._generate_hospitalization_vitals(
                hosp_id, admit_time, discharge_time, location_lookup.get(hosp_id)
            )
            records.extend(hosp_vitals)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

            # Add missingness and outliers
            df = self.add_missingness(df, "vital_value", missingness_rate)
            df = self.add_outliers(df, "vital_value", outlier_rate)

        return df

    def _build_location_lookup(
        self, adt_df: Optional[pd.DataFrame]
    ) -> dict[str, list[tuple]]:
        """Build lookup for hospitalization locations over time."""
        if adt_df is None or len(adt_df) == 0:
            return {}

        lookup = {}
        for hosp_id in adt_df["hospitalization_id"].unique():
            hosp_adt = adt_df[adt_df["hospitalization_id"] == hosp_id]
            locations = []
            for _, row in hosp_adt.iterrows():
                locations.append(
                    (row["in_dttm"], row["out_dttm"], row["location_category"])
                )
            lookup[hosp_id] = sorted(locations, key=lambda x: x[0])

        return lookup

    def _get_location_at_time(
        self, time: datetime, locations: Optional[list[tuple]]
    ) -> str:
        """Get location at a specific time."""
        if locations is None:
            return "icu"

        for in_time, out_time, location in locations:
            if in_time <= time <= out_time:
                return location

        return "icu"

    def _generate_hospitalization_vitals(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        locations: Optional[list[tuple]],
    ) -> list[dict]:
        """Generate all vital signs for one hospitalization."""
        records = []

        # Initialize patient state
        acuity = self.rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.35, 0.15])
        state = PatientState.from_acuity(acuity, self.rng)

        # Get mean measurement interval based on acuity
        mean_interval = 1.0 if acuity <= 2 else 2.0

        # Generate measurement timestamps
        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=mean_interval,
            cv=0.3,
            rng=self.rng,
        )

        # Track last measurement time for state evolution
        last_time = admit_time

        for ts in timestamps:
            # Evolve patient state
            dt_hours = (ts - last_time).total_seconds() / 3600
            if dt_hours > 0:
                state = state.step(dt_hours, self.rng)
            last_time = ts

            # Adjust measurement frequency based on current location
            location = self._get_location_at_time(ts, locations)
            freq = self.FREQUENCY_BY_LOCATION.get(location, 4.0)

            # Not all vitals measured at every time point
            # Core vitals (HR, BP, SpO2, RR) measured frequently
            # Others less frequently

            # Core vitals
            if self.rng.random() < 0.95:  # 95% of time points
                records.extend(
                    self._generate_core_vitals(hospitalization_id, ts, state)
                )

            # Temperature less frequently
            if self.rng.random() < 0.3:  # 30% of time points
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "vital_category": "temp_c",
                        "vital_value": state.temperature,
                        "meas_site_category": self.rng.choice(
                            ["Oral", "Tympanic", "Temporal", "Axillary"]
                        ),
                    }
                )

            # Height/weight only on admission or infrequently
            if (ts - admit_time).total_seconds() < 3600:  # First hour
                records.extend(
                    self._generate_height_weight(hospitalization_id, ts)
                )

        return records

    def _generate_core_vitals(
        self,
        hospitalization_id: str,
        timestamp: datetime,
        state: PatientState,
    ) -> list[dict]:
        """Generate core vital signs from patient state."""
        vitals = []

        # Heart rate
        vitals.append(
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "heart_rate",
                "vital_value": round(state.heart_rate, 0),
                "meas_site_category": None,
            }
        )

        # Blood pressure
        vitals.append(
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "sbp",
                "vital_value": round(state.sbp, 0),
                "meas_site_category": self.rng.choice(
                    ["Arterial", None], p=[0.2, 0.8]
                ),
            }
        )

        vitals.append(
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "dbp",
                "vital_value": round(state.dbp, 0),
                "meas_site_category": self.rng.choice(
                    ["Arterial", None], p=[0.2, 0.8]
                ),
            }
        )

        # MAP (sometimes calculated, sometimes measured)
        if self.rng.random() < 0.7:
            vitals.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": timestamp,
                    "vital_category": "map",
                    "vital_value": round(state.map_value, 0),
                    "meas_site_category": None,
                }
            )

        # SpO2
        vitals.append(
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "spo2",
                "vital_value": round(state.spo2, 0),
                "meas_site_category": None,
            }
        )

        # Respiratory rate
        vitals.append(
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "respiratory_rate",
                "vital_value": round(state.respiratory_rate, 0),
                "meas_site_category": None,
            }
        )

        return vitals

    def _generate_height_weight(
        self,
        hospitalization_id: str,
        timestamp: datetime,
    ) -> list[dict]:
        """Generate height and weight measurements."""
        height = self.rng.normal(170, 10)
        height = np.clip(height, 140, 210)

        # Weight correlates with height (BMI typically 20-35)
        bmi = self.rng.normal(27, 5)
        bmi = np.clip(bmi, 18, 45)
        weight = bmi * (height / 100) ** 2

        return [
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "height_cm",
                "vital_value": round(height, 1),
                "meas_site_category": None,
            },
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": timestamp,
                "vital_category": "weight_kg",
                "vital_value": round(weight, 1),
                "meas_site_category": None,
            },
        ]
