"""Other beta table generators: code_status, position, crrt_therapy."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class CodeStatusGenerator(BaseGenerator):
    """Generate synthetic code status data.

    Creates code_status table with:
    - DNR/DNI transitions over hospitalization
    - Comfort care transitions for terminal patients
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate code status changes.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with code_status columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]
            discharge_category = hosp.get("discharge_category", "")

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            is_terminal = discharge_category == "Expired"
            hosp_codes = self._generate_hospitalization_codes(
                hosp_id, admit_time, discharge_time, is_terminal
            )
            records.extend(hosp_codes)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _generate_hospitalization_codes(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        is_terminal: bool,
    ) -> list[dict]:
        """Generate code status changes for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # Initial code status (admission)
        if is_terminal:
            # Terminal patients may start full code and transition
            initial_status = self.rng.choice(
                ["Full Code", "DNR", "DNR/DNI"],
                p=[0.6, 0.2, 0.2],
            )
        else:
            # Most patients are full code
            initial_status = self.rng.choice(
                ["Full Code", "DNR", "DNR/DNI", "Unknown"],
                p=[0.85, 0.08, 0.05, 0.02],
            )

        records.append(
            {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": admit_time,
                "code_status_category": initial_status,
            }
        )

        # Code status transitions
        current_status = initial_status

        if is_terminal and current_status == "Full Code":
            # Transition to DNR/comfort care before death
            transition_time = admit_time + timedelta(
                hours=self.rng.uniform(los_hours * 0.5, los_hours * 0.9)
            )

            # May transition through DNR before comfort care
            if self.rng.random() < 0.5:
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": transition_time,
                        "code_status_category": "DNR/DNI",
                    }
                )
                transition_time += timedelta(hours=self.rng.uniform(2, 24))

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "recorded_dttm": transition_time,
                    "code_status_category": "Comfort Care",
                }
            )

        elif not is_terminal and current_status != "Full Code":
            # Some patients may return to full code (only if LOS is long enough)
            if los_hours >= 48 and self.rng.random() < 0.2:
                upper_bound = max(25, los_hours * 0.5)
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": admit_time
                        + timedelta(hours=self.rng.uniform(24, upper_bound)),
                        "code_status_category": "Full Code",
                    }
                )

        return records


class PositionGenerator(BaseGenerator):
    """Generate synthetic patient position data.

    Creates position table with:
    - Prone positioning for ARDS patients
    - Regular position changes
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate patient position data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            respiratory_df: Optional respiratory support data

        Returns:
            DataFrame with position columns
        """
        records = []

        # Build ventilation lookup for prone positioning
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
            # Generate prone/supine cycles
            timestamps = generate_irregular_timestamps(
                admit_time,
                discharge_time,
                mean_interval_hours=8,  # ~16 hours prone, 8 hours supine cycles
                cv=0.3,
                rng=self.rng,
            )

            is_prone = False
            for ts in timestamps:
                if is_prone:
                    position = "Supine"
                else:
                    position = "Prone"
                is_prone = not is_prone

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "position_category": position,
                    }
                )
        else:
            # Regular position changes (q2h turns)
            timestamps = generate_irregular_timestamps(
                admit_time,
                discharge_time,
                mean_interval_hours=2,
                cv=0.4,
                rng=self.rng,
            )

            positions = ["Supine", "Left Lateral", "Right Lateral", "Semi-Fowler"]

            for ts in timestamps:
                position = self.rng.choice(positions)
                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "recorded_dttm": ts,
                        "position_category": position,
                    }
                )

        return records


class CRRTTherapyGenerator(BaseGenerator):
    """Generate synthetic CRRT (continuous renal replacement therapy) data.

    Creates crrt_therapy table for patients with acute kidney injury.
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        crrt_rate: float = 0.08,
    ) -> pd.DataFrame:
        """Generate CRRT therapy data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            crrt_rate: Proportion of hospitalizations receiving CRRT

        Returns:
            DataFrame with crrt_therapy columns
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

            # Determine if patient receives CRRT
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

        # CRRT requires minimum LOS to develop AKI and initiate
        if los_hours < 48:
            return records

        # CRRT typically starts after admission (AKI develops)
        start_max = max(12, min(72, los_hours * 0.3))
        crrt_start = admit_time + timedelta(
            hours=self.rng.uniform(12, start_max) if start_max > 12 else 12
        )

        # CRRT duration (typically 2-7 days)
        remaining_hours = (discharge_time - crrt_start).total_seconds() / 3600
        if remaining_hours < 24:
            return records

        crrt_duration = min(
            self.rng.uniform(48, 168), remaining_hours * 0.9
        )
        crrt_end = crrt_start + timedelta(hours=crrt_duration)

        # Hourly CRRT recordings
        timestamps = generate_irregular_timestamps(
            crrt_start,
            crrt_end,
            mean_interval_hours=1,
            cv=0.2,
            rng=self.rng,
        )

        # CRRT mode
        mode = self.rng.choice(["CVVH", "CVVHD", "CVVHDF"], p=[0.3, 0.2, 0.5])

        for ts in timestamps:
            record = {
                "hospitalization_id": hospitalization_id,
                "recorded_dttm": ts,
                "crrt_mode_category": mode,
                "blood_flow_rate": round(self.rng.uniform(150, 250), 0),
                "dialysate_flow_rate": None,
                "replacement_flow_rate": None,
                "ultrafiltration_rate": round(self.rng.uniform(50, 200), 0),
                "effluent_flow_rate": round(self.rng.uniform(1500, 3000), 0),
            }

            if mode in ["CVVHD", "CVVHDF"]:
                record["dialysate_flow_rate"] = round(self.rng.uniform(1000, 2000), 0)

            if mode in ["CVVH", "CVVHDF"]:
                record["replacement_flow_rate"] = round(self.rng.uniform(1000, 2500), 0)

            records.append(record)

        return records
