"""Labs table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_ordered_timestamps


class LabsGenerator(BaseGenerator):
    """Generate synthetic laboratory results.

    Creates labs table with:
    - hospitalization_id (foreign key)
    - lab_order_dttm, lab_collect_dttm, lab_result_dttm (ordered timestamps)
    - lab_category (mCIDE categories)
    - lab_value, lab_value_numeric (with realistic ranges)
    - reference_unit (standard units)
    - lab_type_category (Routine, STAT, etc.)

    Features:
    - Daily routine labs, PRN based on clinical status
    - Proper timestamp ordering: order < collect < result
    - ~10% missingness for less common labs
    """

    # Lab panels (labs typically ordered together)
    LAB_PANELS = {
        "basic_metabolic": [
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "bun",
            "creatinine",
            "glucose",
        ],
        "comprehensive_metabolic": [
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "bun",
            "creatinine",
            "glucose",
            "calcium",
            "albumin",
            "total_protein",
            "ast",
            "alt",
            "alkaline_phosphatase",
            "bilirubin_total",
        ],
        "cbc": ["hemoglobin", "hematocrit", "wbc", "platelets"],
        "coagulation": ["inr", "pt", "ptt"],
        "abg": ["ph", "pco2", "po2", "base_excess"],
        "lactate": ["lactate"],
        "liver": [
            "ast",
            "alt",
            "alkaline_phosphatase",
            "bilirubin_total",
            "bilirubin_direct",
            "albumin",
        ],
        "cardiac": ["troponin", "bnp"],
        "inflammatory": ["crp", "procalcitonin"],
    }

    # Lab parameters: (mean, std, lower_bound, upper_bound)
    LAB_PARAMS = {
        "sodium": (140, 3, 120, 160),
        "potassium": (4.0, 0.5, 2.5, 7.0),
        "chloride": (102, 3, 90, 115),
        "bicarbonate": (24, 3, 10, 40),
        "bun": (15, 8, 5, 100),
        "creatinine": (1.0, 0.5, 0.3, 10.0),
        "glucose": (110, 40, 40, 500),
        "calcium": (9.0, 0.6, 6.0, 14.0),
        "magnesium": (2.0, 0.3, 1.0, 4.0),
        "phosphate": (3.5, 0.8, 1.5, 8.0),
        "albumin": (3.5, 0.6, 1.5, 5.5),
        "total_protein": (7.0, 0.8, 4.0, 10.0),
        "ast": (30, 20, 10, 500),
        "alt": (30, 20, 10, 500),
        "alkaline_phosphatase": (80, 30, 30, 500),
        "bilirubin_total": (0.8, 0.5, 0.1, 20.0),
        "bilirubin_direct": (0.2, 0.2, 0, 10.0),
        "lactate": (1.5, 1.0, 0.5, 15.0),
        "hemoglobin": (12, 2, 5, 18),
        "hematocrit": (38, 5, 15, 55),
        "wbc": (8, 4, 0.5, 50),
        "platelets": (220, 80, 10, 800),
        "inr": (1.1, 0.3, 0.8, 10.0),
        "pt": (12, 2, 9, 50),
        "ptt": (30, 5, 20, 120),
        "fibrinogen": (300, 80, 100, 800),
        "d_dimer": (300, 200, 0, 10000),
        "troponin": (0.02, 0.02, 0, 10),
        "bnp": (80, 100, 0, 5000),
        "crp": (5, 10, 0, 300),
        "procalcitonin": (0.3, 0.5, 0, 50),
        "ph": (7.40, 0.05, 7.0, 7.6),
        "pco2": (40, 5, 20, 80),
        "po2": (85, 15, 40, 500),
        "base_excess": (0, 3, -15, 15),
        "anion_gap": (10, 2, 3, 30),
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        missingness_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Generate laboratory results for hospitalizations.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            missingness_rate: Proportion of missing values

        Returns:
            DataFrame with labs table columns
        """
        records = []
        reference_units = self.mcide.get_lab_reference_units()

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            hosp_labs = self._generate_hospitalization_labs(
                hosp_id, admit_time, discharge_time, reference_units
            )
            records.extend(hosp_labs)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["lab_order_dttm"] = pd.to_datetime(df["lab_order_dttm"], utc=True)
            df["lab_collect_dttm"] = pd.to_datetime(df["lab_collect_dttm"], utc=True)
            df["lab_result_dttm"] = pd.to_datetime(df["lab_result_dttm"], utc=True)

            # Add missingness
            df = self.add_missingness(df, "lab_value_numeric", missingness_rate)
            df = self.add_missingness(df, "lab_order_dttm", 0.02)

        return df

    def _generate_hospitalization_labs(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        reference_units: dict[str, str],
    ) -> list[dict]:
        """Generate all labs for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600
        los_days = los_hours / 24

        # Admission labs (comprehensive)
        admission_labs = self._generate_admission_labs(
            hospitalization_id, admit_time, reference_units
        )
        records.extend(admission_labs)

        # Daily routine labs (BMP + CBC)
        current_day = 1
        while current_day < los_days:
            lab_time = admit_time + timedelta(days=current_day, hours=int(self.rng.integers(4, 8)))
            if lab_time >= discharge_time:
                break

            # BMP daily
            records.extend(
                self._generate_panel(
                    hospitalization_id,
                    lab_time,
                    "basic_metabolic",
                    "Routine",
                    reference_units,
                )
            )

            # CBC every 1-2 days
            if current_day % 2 == 0 or self.rng.random() < 0.5:
                records.extend(
                    self._generate_panel(
                        hospitalization_id,
                        lab_time + timedelta(minutes=5),
                        "cbc",
                        "Routine",
                        reference_units,
                    )
                )

            current_day += 1

        # PRN labs (lactate, ABG, coags)
        n_prn = int(los_days * self.rng.uniform(0.5, 2))
        for _ in range(n_prn):
            prn_time = admit_time + timedelta(hours=self.rng.uniform(0, los_hours))
            panel = self.rng.choice(["lactate", "abg", "coagulation", "cardiac"])
            lab_type = self.rng.choice(["STAT", "Point of Care"], p=[0.7, 0.3])

            records.extend(
                self._generate_panel(
                    hospitalization_id, prn_time, panel, lab_type, reference_units
                )
            )

        return records

    def _generate_admission_labs(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        reference_units: dict[str, str],
    ) -> list[dict]:
        """Generate comprehensive admission labs."""
        records = []

        # Comprehensive metabolic panel
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time,
                "comprehensive_metabolic",
                "STAT",
                reference_units,
            )
        )

        # CBC
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time + timedelta(minutes=2),
                "cbc",
                "STAT",
                reference_units,
            )
        )

        # Coagulation
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time + timedelta(minutes=4),
                "coagulation",
                "STAT",
                reference_units,
            )
        )

        # ABG (for ICU admits, ~50%)
        if self.rng.random() < 0.5:
            records.extend(
                self._generate_panel(
                    hospitalization_id,
                    admit_time + timedelta(minutes=10),
                    "abg",
                    "STAT",
                    reference_units,
                )
            )

        # Lactate
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time + timedelta(minutes=6),
                "lactate",
                "STAT",
                reference_units,
            )
        )

        return records

    def _generate_panel(
        self,
        hospitalization_id: str,
        order_time: datetime,
        panel_name: str,
        lab_type: str,
        reference_units: dict[str, str],
    ) -> list[dict]:
        """Generate a lab panel."""
        records = []
        labs = self.LAB_PANELS.get(panel_name, [panel_name])

        # Generate ordered timestamps: order -> collect -> result
        collect_delay = int(self.rng.integers(5, 30))  # minutes
        result_delay = int(self.rng.integers(30, 180))  # minutes

        collect_time = order_time + timedelta(minutes=collect_delay)
        result_time = collect_time + timedelta(minutes=result_delay)

        for lab_cat in labs:
            if lab_cat not in self.LAB_PARAMS:
                continue

            mean, std, lower, upper = self.LAB_PARAMS[lab_cat]
            value = self.rng.normal(mean, std)
            value = np.clip(value, lower, upper)

            # Format value string
            if lab_cat in ["ph"]:
                value_str = f"{value:.2f}"
                value = round(value, 2)
            elif lab_cat in ["troponin", "procalcitonin"]:
                value_str = f"{value:.3f}"
                value = round(value, 3)
            else:
                value_str = f"{value:.1f}"
                value = round(value, 1)

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "lab_order_dttm": order_time,
                    "lab_collect_dttm": collect_time,
                    "lab_result_dttm": result_time,
                    "lab_category": lab_cat,
                    "lab_value": value_str,
                    "lab_value_numeric": value,
                    "reference_unit": reference_units.get(lab_cat, ""),
                    "lab_type_category": lab_type,
                }
            )

        return records
