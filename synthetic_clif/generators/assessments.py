"""Patient assessments generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class PatientAssessmentsGenerator(BaseGenerator):
    """Generate synthetic patient assessment data.

    Creates patient_assessments table with:
    - GCS (total and components)
    - RASS (sedation scale)
    - CAM-ICU (delirium screening)
    - Pain scores
    - Other ICU assessments
    """

    # Assessment parameters
    ASSESSMENT_PARAMS = {
        "gcs_total": {
            "range": (3, 15),
            "default": 15,
            "frequency_hours": 4,
        },
        "gcs_eye": {
            "range": (1, 4),
            "default": 4,
            "frequency_hours": 4,
        },
        "gcs_verbal": {
            "range": (1, 5),
            "default": 5,
            "frequency_hours": 4,
        },
        "gcs_motor": {
            "range": (1, 6),
            "default": 6,
            "frequency_hours": 4,
        },
        "rass": {
            "range": (-5, 4),
            "default": 0,
            "frequency_hours": 4,
        },
        "cam_icu": {
            "range": (0, 1),  # 0=negative, 1=positive
            "default": 0,
            "frequency_hours": 12,
        },
        "pain_score": {
            "range": (0, 10),
            "default": 0,
            "frequency_hours": 4,
        },
        "braden_score": {
            "range": (6, 23),
            "default": 18,
            "frequency_hours": 24,
        },
        "morse_fall_risk": {
            "range": (0, 125),
            "default": 25,
            "frequency_hours": 24,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate patient assessments.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            respiratory_df: Optional respiratory support for ventilation status

        Returns:
            DataFrame with patient_assessments columns
        """
        records = []

        # Build ventilation status lookup
        vent_lookup = self._build_ventilation_lookup(respiratory_df)

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            is_ventilated = vent_lookup.get(hosp_id, False)
            hosp_assessments = self._generate_hospitalization_assessments(
                hosp_id, admit_time, discharge_time, is_ventilated
            )
            records.extend(hosp_assessments)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _build_ventilation_lookup(
        self, respiratory_df: Optional[pd.DataFrame]
    ) -> dict[str, bool]:
        """Build lookup for ventilation status."""
        if respiratory_df is None or len(respiratory_df) == 0:
            return {}

        lookup = {}
        for hosp_id in respiratory_df["hospitalization_id"].unique():
            hosp_resp = respiratory_df[
                respiratory_df["hospitalization_id"] == hosp_id
            ]
            is_vent = (hosp_resp["device_category"] == "IMV").any()
            lookup[hosp_id] = is_vent

        return lookup

    def _generate_hospitalization_assessments(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_ventilated: bool,
    ) -> list[dict]:
        """Generate all assessments for one hospitalization."""
        records = []

        # Determine patient's baseline status
        is_sedated = is_ventilated and self.rng.random() < 0.8

        # GCS assessments
        records.extend(
            self._generate_gcs_assessments(
                hospitalization_id, admit_time, discharge_time, is_sedated
            )
        )

        # RASS for ICU patients
        if self.rng.random() < 0.7:
            records.extend(
                self._generate_rass_assessments(
                    hospitalization_id, admit_time, discharge_time, is_sedated
                )
            )

        # CAM-ICU
        if self.rng.random() < 0.6:
            records.extend(
                self._generate_camicu_assessments(
                    hospitalization_id, admit_time, discharge_time, is_sedated
                )
            )

        # Pain scores
        records.extend(
            self._generate_pain_assessments(
                hospitalization_id, admit_time, discharge_time, is_sedated
            )
        )

        # Nursing assessments (Braden, Morse) - less frequent
        if self.rng.random() < 0.8:
            records.extend(
                self._generate_nursing_assessments(
                    hospitalization_id, admit_time, discharge_time
                )
            )

        return records

    def _generate_gcs_assessments(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_sedated: bool,
    ) -> list[dict]:
        """Generate GCS assessments."""
        records = []

        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=4,
            cv=0.3,
            rng=self.rng,
        )

        for ts in timestamps:
            if is_sedated:
                # Sedated patients have lower GCS
                gcs_eye = self.rng.integers(1, 3)
                gcs_verbal = 1  # Usually intubated
                gcs_motor = self.rng.integers(1, 5)
            else:
                # Non-sedated
                gcs_eye = self.rng.choice([3, 4], p=[0.2, 0.8])
                gcs_verbal = self.rng.choice([4, 5], p=[0.3, 0.7])
                gcs_motor = self.rng.choice([5, 6], p=[0.2, 0.8])

            gcs_total = gcs_eye + gcs_verbal + gcs_motor

            # Add component scores
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "gcs_total",
                    "assessment_value": float(gcs_total),
                    "assessment_value_text": None,
                }
            )
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "gcs_eye",
                    "assessment_value": float(gcs_eye),
                    "assessment_value_text": None,
                }
            )
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "gcs_verbal",
                    "assessment_value": float(gcs_verbal),
                    "assessment_value_text": None,
                }
            )
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "gcs_motor",
                    "assessment_value": float(gcs_motor),
                    "assessment_value_text": None,
                }
            )

        return records

    def _generate_rass_assessments(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_sedated: bool,
    ) -> list[dict]:
        """Generate RASS (Richmond Agitation-Sedation Scale) assessments."""
        records = []

        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=4,
            cv=0.3,
            rng=self.rng,
        )

        for ts in timestamps:
            if is_sedated:
                # Target RASS -2 to -3 for sedation
                rass = self.rng.choice([-4, -3, -2, -1, 0], p=[0.1, 0.35, 0.35, 0.15, 0.05])
            else:
                # Alert patients mostly 0, some agitated
                rass = self.rng.choice([-1, 0, 1, 2], p=[0.1, 0.75, 0.10, 0.05])

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "rass",
                    "assessment_value": float(rass),
                    "assessment_value_text": None,
                }
            )

        return records

    def _generate_camicu_assessments(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_sedated: bool,
    ) -> list[dict]:
        """Generate CAM-ICU (delirium) assessments."""
        records = []

        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=12,
            cv=0.3,
            rng=self.rng,
        )

        # Delirium incidence ~30% in ICU
        has_delirium = self.rng.random() < 0.30

        for ts in timestamps:
            if is_sedated:
                # Cannot assess if deeply sedated
                result_text = "Unable to Assess"
                result_value = None
            elif has_delirium and self.rng.random() < 0.7:
                result_text = "Positive"
                result_value = 1.0
            else:
                result_text = "Negative"
                result_value = 0.0

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "cam_icu",
                    "assessment_value": result_value,
                    "assessment_value_text": result_text,
                }
            )

        return records

    def _generate_pain_assessments(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_sedated: bool,
    ) -> list[dict]:
        """Generate pain score assessments."""
        records = []

        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=4,
            cv=0.3,
            rng=self.rng,
        )

        for ts in timestamps:
            if is_sedated:
                # Use behavioral pain scale range for sedated
                pain = self.rng.integers(0, 4)
            else:
                # Standard 0-10 scale, most patients 0-4
                pain = self.rng.choice(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    p=[0.25, 0.15, 0.15, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
                )

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "pain_score",
                    "assessment_value": float(pain),
                    "assessment_value_text": None,
                }
            )

        return records

    def _generate_nursing_assessments(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate nursing assessments (Braden, Morse)."""
        records = []

        # Daily assessments
        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=24,
            cv=0.2,
            rng=self.rng,
        )

        for ts in timestamps:
            # Braden score (6-23, higher = lower risk)
            braden = self.rng.integers(12, 22)
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "braden_score",
                    "assessment_value": float(braden),
                    "assessment_value_text": None,
                }
            )

            # Morse fall risk (0-125)
            morse = self.rng.integers(15, 75)
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": ts,
                    "assessment_category": "morse_fall_risk",
                    "assessment_value": float(morse),
                    "assessment_value_text": None,
                }
            )

        return records
