"""Medication administration generators (continuous and intermittent)."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class MedicationContinuousGenerator(BaseGenerator):
    """Generate synthetic continuous medication administration data.

    Creates medication_admin_continuous table with:
    - Vasoactives triggered by low MAP
    - Sedation during mechanical ventilation
    - Insulin infusions
    - Dose titration patterns over time
    """

    # Medication parameters per CLIF 2.1.0 schema
    # med_group permissible values: vasoactives, sedation, diuretics, anticoagulation,
    # cardiac, paralytics, pulmonary vasodilators (IV), pulmonary vasodilators (inhaled),
    # gastrointestinal, Inhaled, endocrine, fluids_electrolytes, others
    MED_PARAMS = {
        "norepinephrine": {
            "dose_range": (0.01, 0.5),
            "unit": "mcg/kg/min",
            "titration": 0.02,
            "med_group": "vasoactives",
        },
        "epinephrine": {
            "dose_range": (0.01, 0.3),
            "unit": "mcg/kg/min",
            "titration": 0.01,
            "med_group": "vasoactives",
        },
        "vasopressin": {
            "dose_range": (0.01, 0.04),
            "unit": "units/min",
            "titration": 0.005,
            "med_group": "vasoactives",
        },
        "dopamine": {
            "dose_range": (2, 20),
            "unit": "mcg/kg/min",
            "titration": 2,
            "med_group": "vasoactives",
        },
        "dobutamine": {
            "dose_range": (2, 20),
            "unit": "mcg/kg/min",
            "titration": 2.5,
            "med_group": "vasoactives",
        },
        "phenylephrine": {
            "dose_range": (20, 200),
            "unit": "mcg/min",
            "titration": 20,
            "med_group": "vasoactives",
        },
        "milrinone": {
            "dose_range": (0.125, 0.75),
            "unit": "mcg/kg/min",
            "titration": 0.125,
            "med_group": "vasoactives",
        },
        "propofol": {
            "dose_range": (5, 80),
            "unit": "mcg/kg/min",
            "titration": 10,
            "med_group": "sedation",
        },
        "dexmedetomidine": {
            "dose_range": (0.2, 1.5),
            "unit": "mcg/kg/hr",
            "titration": 0.1,
            "med_group": "sedation",
        },
        "midazolam": {
            "dose_range": (0.5, 10),
            "unit": "mg/hr",
            "titration": 1,
            "med_group": "sedation",
        },
        "fentanyl": {
            "dose_range": (25, 200),
            "unit": "mcg/hr",
            "titration": 25,
            "med_group": "sedation",
        },
        "morphine": {
            "dose_range": (1, 10),
            "unit": "mg/hr",
            "titration": 1,
            "med_group": "sedation",
        },
        "heparin": {
            "dose_range": (500, 2000),
            "unit": "units/hr",
            "titration": 100,
            "med_group": "anticoagulation",
        },
        "insulin": {
            "dose_range": (0.5, 10),
            "unit": "units/hr",
            "titration": 0.5,
            "med_group": "endocrine",
        },
    }

    # MAR action categories per CLIF 2.1.0 schema
    # mar_action_category: dose_change, going, start, stop, verify, other
    # mar_action_group: administered, not_administered, other
    MAR_ACTION_CATEGORIES = ["dose_change", "going", "start", "stop", "verify", "other"]
    MAR_ACTION_WEIGHTS = [0.30, 0.35, 0.10, 0.10, 0.10, 0.05]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate continuous medication administration.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            respiratory_df: Optional respiratory support for ventilation status

        Returns:
            DataFrame with medication_admin_continuous columns
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
            hosp_meds = self._generate_hospitalization_meds(
                hosp_id, admit_time, discharge_time, is_ventilated
            )
            records.extend(hosp_meds)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["admin_dttm"] = pd.to_datetime(df["admin_dttm"], utc=True)

        return df

    def _build_ventilation_lookup(
        self, respiratory_df: Optional[pd.DataFrame]
    ) -> dict[str, bool]:
        """Build lookup for which hospitalizations have ventilation."""
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

    def _safe_uniform(self, low: float, high: float) -> float:
        """Generate uniform random value, handling edge cases where high <= low."""
        if high <= low:
            return low
        return self.rng.uniform(low, high)

    def _generate_hospitalization_meds(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_ventilated: bool,
    ) -> list[dict]:
        """Generate continuous meds for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # Skip very short stays
        if los_hours < 4:
            return records

        # Determine which medications this patient receives
        # Vasopressors: ~25% of ICU patients
        if self.rng.random() < 0.25 and los_hours >= 12:
            # Primary vasopressor (usually norepinephrine)
            records.extend(
                self._generate_infusion(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "norepinephrine",
                    duration_hours=self._safe_uniform(12, min(72, los_hours)),
                )
            )

            # Some need second vasopressor
            if self.rng.random() < 0.3 and los_hours >= 24:
                records.extend(
                    self._generate_infusion(
                        hospitalization_id,
                        admit_time + timedelta(hours=self._safe_uniform(2, min(12, los_hours / 2))),
                        discharge_time,
                        "vasopressin",
                        duration_hours=self._safe_uniform(12, min(48, los_hours)),
                    )
                )

        # Sedation: for ventilated patients
        if is_ventilated and los_hours >= 12:
            # Primary sedative
            sedative = self.rng.choice(["propofol", "dexmedetomidine"], p=[0.6, 0.4])
            records.extend(
                self._generate_infusion(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    sedative,
                    duration_hours=self._safe_uniform(min(12, los_hours), min(120, los_hours)),
                )
            )

            # Analgesia
            analgesic = self.rng.choice(["fentanyl", "morphine"], p=[0.7, 0.3])
            records.extend(
                self._generate_infusion(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    analgesic,
                    duration_hours=self._safe_uniform(min(12, los_hours), min(96, los_hours)),
                )
            )

        # Insulin: ~20% of patients
        if self.rng.random() < 0.2 and los_hours >= 12:
            start_offset = self._safe_uniform(0, min(24, los_hours / 2))
            records.extend(
                self._generate_infusion(
                    hospitalization_id,
                    admit_time + timedelta(hours=start_offset),
                    discharge_time,
                    "insulin",
                    duration_hours=self._safe_uniform(12, min(72, los_hours - start_offset)),
                )
            )

        # Heparin: ~15% of patients
        if self.rng.random() < 0.15 and los_hours >= 24:
            start_offset = self._safe_uniform(0, min(48, los_hours / 2))
            records.extend(
                self._generate_infusion(
                    hospitalization_id,
                    admit_time + timedelta(hours=start_offset),
                    discharge_time,
                    "heparin",
                    duration_hours=self._safe_uniform(24, min(120, los_hours - start_offset)),
                )
            )

        return records

    def _generate_infusion(
        self,
        hospitalization_id: str,
        start_time: datetime,
        end_time: datetime,
        medication: str,
        duration_hours: float,
    ) -> list[dict]:
        """Generate infusion records with titration."""
        records = []
        params = self.MED_PARAMS.get(medication)
        if params is None:
            return records

        order_id = str(uuid.uuid4())[:8]
        actual_end = min(start_time + timedelta(hours=duration_hours), end_time)

        # Generate dose changes over time (titration pattern)
        timestamps = generate_irregular_timestamps(
            start_time,
            actual_end,
            mean_interval_hours=1.0,
            cv=0.4,
            rng=self.rng,
        )

        dose_range = params["dose_range"]
        titration = params["titration"]
        current_dose = self.rng.uniform(dose_range[0], (dose_range[0] + dose_range[1]) / 2)

        for ts in timestamps:
            # Titration: small dose adjustments
            if self.rng.random() < 0.3:
                delta = self.rng.choice([-1, 0, 1]) * titration
                current_dose = np.clip(
                    current_dose + delta, dose_range[0], dose_range[1]
                )

            # Determine MAR action category and group per CLIF 2.1.0 schema
            mar_action_weights = np.array(self.MAR_ACTION_WEIGHTS, dtype=float)
            mar_action_weights /= mar_action_weights.sum()
            mar_action_category = self.rng.choice(self.MAR_ACTION_CATEGORIES, p=mar_action_weights)
            
            # Map to mar_action_group
            if mar_action_category in ["going", "start", "dose_change"]:
                mar_action_group = "administered"
            elif mar_action_category == "stop":
                mar_action_group = "not_administered"
            else:
                mar_action_group = "other"

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "med_order_id": order_id,
                    "admin_dttm": ts,
                    "med_category": medication,
                    "med_name": medication.replace("_", "-").title(),
                    "med_group": params["med_group"],
                    "med_dose": round(current_dose, 3),
                    "med_dose_unit": params["unit"],
                    "med_route_category": "iv",  # lowercase per schema
                    "mar_action_category": mar_action_category,
                    "mar_action_group": mar_action_group,
                }
            )

        return records


class MedicationIntermittentGenerator(BaseGenerator):
    """Generate synthetic intermittent medication administration data.

    Creates medication_admin_intermittent table with:
    - Scheduled medications (antibiotics, PPIs)
    - PRN medications
    - MAR action categories (given, held, refused)
    """

    # Common intermittent medications per CLIF 2.1.0 schema
    # med_group permissible values: CMS_sepsis_qualifying_antibiotics, analgesia, 
    # antipsychotic, anxiolytic, car_t, other, paralytics, sedation, steroid, vasopressor
    # med_route_category: enteral, im, iv, buccal_sublingual, intrapleural
    MED_SCHEDULES = {
        "vancomycin": {
            "dose_range": (1000, 1500),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 12,
            "med_group": "CMS_sepsis_qualifying_antibiotics",
        },
        "piperacillin_tazobactam": {
            "dose_range": (3375, 4500),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 6,
            "med_group": "CMS_sepsis_qualifying_antibiotics",
        },
        "cefepime": {
            "dose_range": (1000, 2000),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 8,
            "med_group": "CMS_sepsis_qualifying_antibiotics",
        },
        "meropenem": {
            "dose_range": (1000, 2000),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 8,
            "med_group": "CMS_sepsis_qualifying_antibiotics",
        },
        "pantoprazole": {
            "dose_range": (40, 40),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 24,
            "med_group": "other",
        },
        "metoprolol": {
            "dose_range": (25, 100),
            "unit": "mg",
            "route": "enteral",
            "frequency_hours": 12,
            "med_group": "other",
        },
        "lisinopril": {
            "dose_range": (5, 40),
            "unit": "mg",
            "route": "enteral",
            "frequency_hours": 24,
            "med_group": "other",
        },
        "aspirin": {
            "dose_range": (81, 325),
            "unit": "mg",
            "route": "enteral",
            "frequency_hours": 24,
            "med_group": "other",
        },
        "enoxaparin": {
            "dose_range": (40, 80),
            "unit": "mg",
            "route": "im",
            "frequency_hours": 12,
            "med_group": "other",
        },
        "fentanyl": {
            "dose_range": (25, 100),
            "unit": "mcg",
            "route": "iv",
            "frequency_hours": 4,
            "med_group": "analgesia",
        },
        "morphine": {
            "dose_range": (2, 10),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 4,
            "med_group": "analgesia",
        },
        "dexamethasone": {
            "dose_range": (4, 10),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 6,
            "med_group": "steroid",
        },
        "methylprednisolone": {
            "dose_range": (40, 125),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 6,
            "med_group": "steroid",
        },
        "haloperidol": {
            "dose_range": (2, 10),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 6,
            "med_group": "antipsychotic",
        },
        "lorazepam": {
            "dose_range": (0.5, 2),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 6,
            "med_group": "anxiolytic",
        },
        "rocuronium": {
            "dose_range": (50, 100),
            "unit": "mg",
            "route": "iv",
            "frequency_hours": 1,
            "med_group": "paralytics",
        },
    }

    # MAR action categories for intermittent per CLIF 2.1.0 schema
    # mar_action_category: given, bolus, not_given, other
    INTERMITTENT_MAR_CATEGORIES = ["given", "bolus", "not_given", "other"]
    INTERMITTENT_MAR_WEIGHTS = [0.85, 0.05, 0.07, 0.03]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate intermittent medication administration.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with medication_admin_intermittent columns
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

            hosp_meds = self._generate_hospitalization_meds(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_meds)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["admin_dttm"] = pd.to_datetime(df["admin_dttm"], utc=True)

        return df

    def _generate_hospitalization_meds(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate intermittent meds for one hospitalization."""
        records = []

        # PPI for most patients (stress ulcer prophylaxis)
        if self.rng.random() < 0.85:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "pantoprazole",
                )
            )

        # Antibiotics: ~50% of patients
        if self.rng.random() < 0.5:
            # Choose antibiotic regimen
            if self.rng.random() < 0.4:
                # Vancomycin + piperacillin-tazobactam (common broad spectrum)
                records.extend(
                    self._generate_scheduled_med(
                        hospitalization_id,
                        admit_time,
                        discharge_time,
                        "vancomycin",
                        duration_days=self.rng.uniform(5, 14),
                    )
                )
                records.extend(
                    self._generate_scheduled_med(
                        hospitalization_id,
                        admit_time,
                        discharge_time,
                        "piperacillin_tazobactam",
                        duration_days=self.rng.uniform(5, 14),
                    )
                )
            else:
                # Single agent
                abx = self.rng.choice(["cefepime", "meropenem"])
                records.extend(
                    self._generate_scheduled_med(
                        hospitalization_id,
                        admit_time,
                        discharge_time,
                        abx,
                        duration_days=self.rng.uniform(5, 10),
                    )
                )

        # DVT prophylaxis
        if self.rng.random() < 0.7:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "enoxaparin",
                )
            )

        # Cardiac meds for some patients
        if self.rng.random() < 0.3:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "metoprolol",
                )
            )
        if self.rng.random() < 0.2:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "aspirin",
                )
            )

        return records

    def _generate_scheduled_med(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        medication: str,
        duration_days: Optional[float] = None,
    ) -> list[dict]:
        """Generate scheduled medication doses."""
        records = []
        params = self.MED_SCHEDULES.get(medication)
        if params is None:
            return records

        order_id = str(uuid.uuid4())[:8]
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        if duration_days:
            end_time = min(
                admit_time + timedelta(days=duration_days), discharge_time
            )
        else:
            end_time = discharge_time

        freq_hours = params["frequency_hours"]
        dose_range = params["dose_range"]
        dose = self.rng.uniform(dose_range[0], dose_range[1])

        current_time = admit_time
        while current_time < end_time:
            # Determine MAR action per CLIF 2.1.0 schema
            mar_action_category, mar_action_group = self._sample_mar_action_intermittent()

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "med_order_id": order_id,
                    "admin_dttm": current_time,
                    "med_category": medication,
                    "med_name": medication.replace("_", "-").title(),
                    "med_group": params["med_group"],
                    "med_dose": round(dose, 0),
                    "med_dose_unit": params["unit"],
                    "med_route_category": params["route"],
                    "mar_action_category": mar_action_category,
                    "mar_action_group": mar_action_group,
                }
            )

            # Add jitter to scheduled time
            jitter = self.rng.uniform(-0.5, 0.5)
            current_time += timedelta(hours=freq_hours + jitter)

        return records

    def _sample_mar_action_intermittent(self) -> tuple[str, str]:
        """Sample MAR action with realistic distribution per CLIF 2.1.0 schema."""
        # mar_action_category: given, bolus, not_given, other
        weights = np.array(self.INTERMITTENT_MAR_WEIGHTS, dtype=float)
        weights /= weights.sum()
        mar_action_category = self.rng.choice(self.INTERMITTENT_MAR_CATEGORIES, p=weights)
        
        # Map to mar_action_group
        if mar_action_category in ["given", "bolus"]:
            mar_action_group = "administered"
        elif mar_action_category == "not_given":
            mar_action_group = "not_administered"
        else:
            mar_action_group = "other"
            
        return mar_action_category, mar_action_group
