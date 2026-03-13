"""Other beta table generators: code_status, position, crrt_therapy."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class CodeStatusGenerator(BaseGenerator):
    """Generate synthetic code status data per CLIF 2.1.0.

    Creates code_status table with:
    - patient_id (not hospitalization_id per schema)
    - start_dttm (not recorded_dttm)
    - code_status_category (Full, DNR, DNR/DNI, etc.)
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate code status changes."""
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            patient_id = hosp.get("patient_id", hosp["hospitalization_id"])
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]
            discharge_category = hosp.get("discharge_category", "")

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            is_terminal = discharge_category == "Expired"
            hosp_codes = self._generate_hospitalization_codes(
                patient_id, admit_time, discharge_time, is_terminal
            )
            records.extend(hosp_codes)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["start_dttm"] = pd.to_datetime(df["start_dttm"], utc=True)

        return df

    def _generate_hospitalization_codes(
        self,
        patient_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_terminal: bool,
    ) -> list[dict]:
        """Generate code status changes for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # CLIF 2.1.0 code_status_category values:
        # DNR, DNAR, UDNR, DNR/DNI, DNAR/DNI, AND, Full, Presume Full, Other
        if is_terminal:
            initial_status = self.rng.choice(
                ["Full", "DNR", "DNR/DNI"],
                p=[0.6, 0.2, 0.2],
            )
        else:
            initial_status = self.rng.choice(
                ["Full", "Presume Full", "DNR", "DNR/DNI"],
                p=[0.80, 0.10, 0.06, 0.04],
            )

        records.append(
            {
                "patient_id": patient_id,
                "start_dttm": admit_time,
                "code_status_category": initial_status,
                "code_status_name": self.name_from_category(initial_status),
            }
        )

        current_status = initial_status

        if is_terminal and current_status in ["Full", "Presume Full"]:
            transition_time = admit_time + timedelta(
                hours=self.rng.uniform(los_hours * 0.5, los_hours * 0.9)
            )

            if self.rng.random() < 0.5:
                records.append(
                    {
                        "patient_id": patient_id,
                        "start_dttm": transition_time,
                        "code_status_category": "DNR/DNI",
                        "code_status_name": self.name_from_category("DNR/DNI"),
                    }
                )
                transition_time += timedelta(hours=self.rng.uniform(2, 24))

            records.append(
                {
                    "patient_id": patient_id,
                    "start_dttm": transition_time,
                    "code_status_category": "Other",
                    "code_status_name": self.name_from_category("Other"),
                }
            )

        elif not is_terminal and current_status not in ["Full", "Presume Full"]:
            if los_hours >= 48 and self.rng.random() < 0.2:
                upper_bound = max(25, los_hours * 0.5)
                records.append(
                    {
                        "patient_id": patient_id,
                        "start_dttm": admit_time
                        + timedelta(hours=self.rng.uniform(24, upper_bound)),
                        "code_status_category": "Full",
                        "code_status_name": self.name_from_category("Full"),
                    }
                )

        return records


class PositionGenerator(BaseGenerator):
    """Generate synthetic patient position data per CLIF 2.1.0.

    Creates position table with position_category: prone, not_prone
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate patient position data."""
        records = []

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
            hosp_positions = self._generate_hospitalization_positions(
                hosp_id, admit_time, discharge_time, is_ventilated
            )
            records.extend(hosp_positions)

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

    def _generate_hospitalization_positions(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_ventilated: bool,
    ) -> list[dict]:
        """Generate positions for one hospitalization."""
        records = []

        # Determine if patient receives prone positioning (~10% of ventilated)
        has_prone = is_ventilated and self.rng.random() < 0.10

        if has_prone:
            timestamps = generate_irregular_timestamps(
                admit_time,
                discharge_time,
                mean_interval_hours=8,
                cv=0.3,
                rng=self.rng,
            )

            is_prone = False
            for ts in timestamps:
                position = "prone" if not is_prone else "not_prone"
                is_prone = not is_prone

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "position_category": position,
                        "position_name": self.name_from_category(position),
                    }
                )
        else:
            timestamps = generate_irregular_timestamps(
                admit_time,
                discharge_time,
                mean_interval_hours=2,
                cv=0.4,
                rng=self.rng,
            )

            for ts in timestamps:
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "position_category": "not_prone",
                        "position_name": self.name_from_category("not_prone"),
                    }
                )

        return records


class CRRTTherapyGenerator(BaseGenerator):
    """Generate synthetic CRRT data per CLIF 2.1.0 schema."""

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        crrt_rate: float = 0.08,
    ) -> pd.DataFrame:
        """Generate CRRT therapy data."""
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            if self.rng.random() > crrt_rate:
                continue

            hosp_crrt = self._generate_hospitalization_crrt(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_crrt)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _generate_hospitalization_crrt(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate CRRT data for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        if los_hours < 48:
            return records

        start_max = max(12, min(72, los_hours * 0.3))
        crrt_start = admit_time + timedelta(
            hours=self.rng.uniform(12, start_max) if start_max > 12 else 12
        )

        remaining_hours = (discharge_time - crrt_start).total_seconds() / 3600
        if remaining_hours < 24:
            return records

        crrt_duration = min(
            self.rng.uniform(48, 168), remaining_hours * 0.9
        )
        crrt_end = crrt_start + timedelta(hours=crrt_duration)

        timestamps = generate_irregular_timestamps(
            crrt_start,
            crrt_end,
            mean_interval_hours=1,
            cv=0.2,
            rng=self.rng,
        )

        # CLIF 2.1.0 mode values: scuf, cvvh, cvvhd, cvvhdf, avvh
        mode = self.rng.choice(["cvvh", "cvvhd", "cvvhdf"], p=[0.3, 0.2, 0.5])

        dialysis_machines = ["Prismaflex", "PrisMax", "NxStage", "Aquarius"]

        for ts in timestamps:
            record = {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": ts,
                "device_id": f"CRRT-{self.rng.integers(1000, 9999)}",
                "crrt_mode_category": mode,
                "crrt_mode_name": self.name_from_category(mode),
                "dialysis_machine_name": self.rng.choice(dialysis_machines),
                "blood_flow_rate": round(self.rng.uniform(150, 250), 0),
                "pre_filter_replacement_fluid_rate": None,
                "post_filter_replacement_fluid_rate": None,
                "dialysate_flow_rate": None,
                "ultrafiltration_out": round(self.rng.uniform(50, 200), 0),
            }

            if mode in ["cvvhd", "cvvhdf"]:
                record["dialysate_flow_rate"] = round(self.rng.uniform(1000, 2000), 0)

            if mode in ["cvvh", "cvvhdf"]:
                record["pre_filter_replacement_fluid_rate"] = round(
                    self.rng.uniform(500, 1500), 0
                )
                record["post_filter_replacement_fluid_rate"] = round(
                    self.rng.uniform(500, 1500), 0
                )

            records.append(record)

        return records
