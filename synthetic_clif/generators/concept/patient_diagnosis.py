"""Patient diagnosis (problem list/history) generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class PatientDiagnosisGenerator(BaseGenerator):
    """Generate synthetic patient diagnosis data.

    Creates patient_diagnosis table with problem list and medical history.
    """

    # Chronic conditions
    CONDITIONS = {
        "Hypertension": {"code": "I10", "probability": 0.55},
        "Type 2 Diabetes": {"code": "E11.9", "probability": 0.35},
        "Hyperlipidemia": {"code": "E78.5", "probability": 0.40},
        "Coronary Artery Disease": {"code": "I25.10", "probability": 0.20},
        "Heart Failure": {"code": "I50.9", "probability": 0.15},
        "Atrial Fibrillation": {"code": "I48.91", "probability": 0.12},
        "COPD": {"code": "J44.9", "probability": 0.15},
        "Asthma": {"code": "J45.909", "probability": 0.10},
        "Chronic Kidney Disease": {"code": "N18.9", "probability": 0.18},
        "Obesity": {"code": "E66.9", "probability": 0.30},
        "Depression": {"code": "F32.9", "probability": 0.15},
        "Anxiety": {"code": "F41.9", "probability": 0.12},
        "Hypothyroidism": {"code": "E03.9", "probability": 0.10},
        "GERD": {"code": "K21.0", "probability": 0.20},
        "Osteoarthritis": {"code": "M19.90", "probability": 0.15},
    }

    def generate(
        self,
        patients_df: pd.DataFrame,
        hospitalizations_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Generate patient diagnosis data.

        Args:
            patients_df: Patient table DataFrame
            hospitalizations_df: Optional hospitalization table DataFrame.
                If provided, diagnoses are assigned a hospitalization_id.

        Returns:
            DataFrame with patient_diagnosis columns
        """
        records = []

        # Build patient_id -> hospitalization_ids mapping
        patient_hosps = {}
        if hospitalizations_df is not None and len(hospitalizations_df) > 0:
            for _, hosp in hospitalizations_df.iterrows():
                pid = hosp.get("patient_id")
                if pid:
                    patient_hosps.setdefault(pid, []).append(hosp["hospitalization_id"])

        for _, patient in patients_df.iterrows():
            patient_id = patient["patient_id"]
            birth_date = patient.get("birth_date")

            for dx_name, params in self.CONDITIONS.items():
                if self.rng.random() > params["probability"]:
                    continue

                # Diagnosis date (sometime in past)
                if pd.notna(birth_date):
                    years_ago = self.rng.uniform(1, 20)
                    dx_date = datetime.now(timezone.utc) - timedelta(days=years_ago * 365)
                else:
                    dx_date = None

                # Assign hospitalization_id if available
                hosp_id = None
                if patient_id in patient_hosps:
                    hosp_id = self.rng.choice(patient_hosps[patient_id])

                # Most chronic conditions have no end_dttm
                end_dttm = None

                records.append(
                    {
                        "patient_id": patient_id,
                        "hospitalization_id": hosp_id,
                        "diagnosis_code": params["code"],
                        "diagnosis_code_format": "ICD-10-CM",
                        "start_dttm": dx_date,
                        "end_dttm": end_dttm,
                        "source_type": self.rng.choice(
                            ["Problem List", "Medical History", "Encounter"],
                            p=[0.5, 0.3, 0.2],
                        ),
                    }
                )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["start_dttm"] = pd.to_datetime(df["start_dttm"], utc=True)
            df["end_dttm"] = pd.to_datetime(df["end_dttm"], utc=True)

        return df
