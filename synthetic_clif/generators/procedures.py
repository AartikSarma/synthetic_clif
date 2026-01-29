"""Patient procedures and hospital diagnosis generators."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader


class PatientProceduresGenerator(BaseGenerator):
    """Generate synthetic patient procedure data.

    Creates patient_procedures table with:
    - ICU-relevant procedures (central lines, intubations, etc.)
    - ICD-10-PCS and CPT codes
    """

    # Common ICU procedures with codes
    PROCEDURES = {
        "Central Line Insertion": {
            "icd10_pcs": ["02HV33Z", "02HV43Z"],
            "cpt": ["36556", "36558"],
            "probability": 0.4,
        },
        "Arterial Line Insertion": {
            "icd10_pcs": ["02H633Z", "02H733Z"],
            "cpt": ["36620"],
            "probability": 0.35,
        },
        "Intubation": {
            "icd10_pcs": ["0BH17EZ"],
            "cpt": ["31500"],
            "probability": 0.15,
        },
        "Extubation": {
            "icd10_pcs": ["0BH18EZ"],
            "cpt": ["31500"],
            "probability": 0.12,
        },
        "Bronchoscopy": {
            "icd10_pcs": ["0BJ08ZZ"],
            "cpt": ["31622", "31623"],
            "probability": 0.08,
        },
        "Chest Tube Insertion": {
            "icd10_pcs": ["0W9930Z", "0W9940Z"],
            "cpt": ["32551"],
            "probability": 0.06,
        },
        "Paracentesis": {
            "icd10_pcs": ["0W9G30Z"],
            "cpt": ["49083"],
            "probability": 0.05,
        },
        "Thoracentesis": {
            "icd10_pcs": ["0W9B30Z"],
            "cpt": ["32555"],
            "probability": 0.04,
        },
        "Lumbar Puncture": {
            "icd10_pcs": ["009U3ZX"],
            "cpt": ["62270"],
            "probability": 0.03,
        },
        "Dialysis Catheter Insertion": {
            "icd10_pcs": ["02HV33Z"],
            "cpt": ["36558"],
            "probability": 0.08,
        },
        "PICC Line Insertion": {
            "icd10_pcs": ["02HV33Z"],
            "cpt": ["36569"],
            "probability": 0.15,
        },
        "Tracheostomy": {
            "icd10_pcs": ["0B110F4"],
            "cpt": ["31600"],
            "probability": 0.03,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate patient procedures.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with patient_procedures columns
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

            hosp_procedures = self._generate_hospitalization_procedures(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_procedures)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["procedure_dttm"] = pd.to_datetime(df["procedure_dttm"], utc=True)

        return df

    def _generate_hospitalization_procedures(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate procedures for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        for proc_name, proc_data in self.PROCEDURES.items():
            if self.rng.random() < proc_data["probability"]:
                # Procedure timing
                if proc_name in ["Intubation", "Central Line Insertion", "Arterial Line Insertion"]:
                    # Early procedures
                    hours_from_admit = self.rng.uniform(0, min(12, los_hours))
                elif proc_name in ["Extubation", "Tracheostomy"]:
                    # Later procedures
                    hours_from_admit = self.rng.uniform(
                        min(24, los_hours * 0.3), los_hours * 0.9
                    )
                else:
                    hours_from_admit = self.rng.uniform(0, los_hours * 0.8)

                proc_time = admit_time + timedelta(hours=hours_from_admit)

                # Select code type
                use_icd10 = self.rng.random() < 0.7
                if use_icd10:
                    code = self.rng.choice(proc_data["icd10_pcs"])
                    code_type = "ICD-10-PCS"
                else:
                    code = self.rng.choice(proc_data["cpt"])
                    code_type = "CPT"

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "procedure_dttm": proc_time,
                        "procedure_code": code,
                        "procedure_code_type": code_type,
                        "procedure_category": proc_name,
                    }
                )

        return records


class HospitalDiagnosisGenerator(BaseGenerator):
    """Generate synthetic hospital diagnosis data.

    Creates hospital_diagnosis table with:
    - ICD-10-CM diagnosis codes
    - Principal and secondary diagnoses
    - POA (present on admission) flags
    """

    # Common ICU diagnoses with ICD-10-CM codes
    DIAGNOSES = {
        # Respiratory
        "Acute respiratory failure": {
            "codes": ["J96.00", "J96.01", "J96.02"],
            "probability": 0.35,
            "is_principal": True,
        },
        "Pneumonia": {
            "codes": ["J18.9", "J15.9", "J13"],
            "probability": 0.25,
            "is_principal": True,
        },
        "ARDS": {
            "codes": ["J80"],
            "probability": 0.08,
            "is_principal": True,
        },
        "COPD exacerbation": {
            "codes": ["J44.1"],
            "probability": 0.12,
            "is_principal": True,
        },
        # Cardiovascular
        "Sepsis": {
            "codes": ["A41.9", "R65.20", "R65.21"],
            "probability": 0.30,
            "is_principal": True,
        },
        "Acute kidney injury": {
            "codes": ["N17.9", "N17.0", "N17.1"],
            "probability": 0.25,
            "is_principal": False,
        },
        "Acute MI": {
            "codes": ["I21.3", "I21.4", "I21.9"],
            "probability": 0.10,
            "is_principal": True,
        },
        "Heart failure": {
            "codes": ["I50.9", "I50.21", "I50.31"],
            "probability": 0.20,
            "is_principal": False,
        },
        "Atrial fibrillation": {
            "codes": ["I48.91", "I48.0", "I48.1"],
            "probability": 0.18,
            "is_principal": False,
        },
        # Neurologic
        "Stroke": {
            "codes": ["I63.9", "I61.9"],
            "probability": 0.08,
            "is_principal": True,
        },
        "Altered mental status": {
            "codes": ["R41.82"],
            "probability": 0.15,
            "is_principal": False,
        },
        # GI
        "GI bleed": {
            "codes": ["K92.2", "K92.0", "K92.1"],
            "probability": 0.08,
            "is_principal": True,
        },
        "Acute pancreatitis": {
            "codes": ["K85.9", "K85.0"],
            "probability": 0.05,
            "is_principal": True,
        },
        # Metabolic
        "DKA": {
            "codes": ["E11.10", "E10.10"],
            "probability": 0.05,
            "is_principal": True,
        },
        "Hyponatremia": {
            "codes": ["E87.1"],
            "probability": 0.12,
            "is_principal": False,
        },
        "Hyperkalemia": {
            "codes": ["E87.5"],
            "probability": 0.08,
            "is_principal": False,
        },
        # Comorbidities
        "Hypertension": {
            "codes": ["I10"],
            "probability": 0.50,
            "is_principal": False,
        },
        "Diabetes mellitus": {
            "codes": ["E11.9", "E10.9"],
            "probability": 0.35,
            "is_principal": False,
        },
        "Chronic kidney disease": {
            "codes": ["N18.3", "N18.4", "N18.5"],
            "probability": 0.20,
            "is_principal": False,
        },
        "Obesity": {
            "codes": ["E66.9", "E66.01"],
            "probability": 0.25,
            "is_principal": False,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate hospital diagnoses.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with hospital_diagnosis columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            hosp_diagnoses = self._generate_hospitalization_diagnoses(hosp_id)
            records.extend(hosp_diagnoses)

        return pd.DataFrame(records)

    def _generate_hospitalization_diagnoses(
        self,
        hospitalization_id: str,
    ) -> list[dict]:
        """Generate diagnoses for one hospitalization."""
        records = []
        has_principal = False

        # Shuffle diagnoses to randomize selection
        dx_items = list(self.DIAGNOSES.items())
        self.rng.shuffle(dx_items)

        for dx_name, dx_data in dx_items:
            if self.rng.random() < dx_data["probability"]:
                code = self.rng.choice(dx_data["codes"])

                # Determine diagnosis type
                if dx_data["is_principal"] and not has_principal:
                    dx_type = "Principal"
                    has_principal = True
                else:
                    dx_type = "Secondary"

                # POA status
                poa = self.rng.choice(
                    ["Yes", "No", "Unknown"],
                    p=[0.70, 0.20, 0.10],
                )

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "diagnosis_code": code,
                        "diagnosis_code_type": "ICD-10-CM",
                        "diagnosis_name": dx_name,
                        "diagnosis_type": dx_type,
                        "poa_category": poa,
                    }
                )

        # Ensure at least one principal diagnosis
        if records and not has_principal:
            records[0]["diagnosis_type"] = "Principal"

        return records
