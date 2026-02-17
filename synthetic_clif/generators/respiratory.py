"""Respiratory support table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.models.patient_state import PatientState
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class RespiratoryGenerator(BaseGenerator):
    """Generate synthetic respiratory support data.

    Creates respiratory_support table with:
    - hospitalization_id (foreign key)
    - recorded_dttm (hourly in ICU, less frequent elsewhere)
    - device_category (IMV, NIPPV, CPAP, High Flow NC, etc.)
    - mode_category (Volume Control, Pressure Support, etc.)
    - Ventilator settings (fio2, peep, tidal_volume, etc.)
    - tracheostomy flag

    Features:
    - Device escalation/de-escalation correlated with SpO2
    - Mode categories linked to device type
    - Realistic parameter ranges
    """

    # CLIF 2.1.0 mode_category values
    # "Assist Control-Volume Control", "Pressure Control",
    # "Pressure-Regulated Volume Control", "SIMV",
    # "Pressure Support/CPAP", "Volume Support", "Blow by", "Other"

    VENT_BRANDS = ["Puritan Bennett 840", "Servo-i", "Drager V500", "Hamilton G5", "Avea"]

    # Device-specific settings ranges
    DEVICE_SETTINGS = {
        "IMV": {
            "modes": [
                "Assist Control-Volume Control",
                "Pressure Control",
                "Pressure-Regulated Volume Control",
                "SIMV",
                "Pressure Support/CPAP",
            ],
            "fio2_range": (0.3, 1.0),
            "peep_range": (5, 20),
            "tidal_volume_range": (300, 600),
            "resp_rate_range": (12, 28),
            "pressure_support_range": (5, 20),
            "pressure_control_range": (15, 35),
        },
        "NIPPV": {
            "modes": ["BiPAP", "CPAP"],
            "fio2_range": (0.3, 0.8),
            "peep_range": (5, 12),
            "pressure_support_range": (8, 20),
        },
        "CPAP": {
            "modes": ["CPAP"],
            "fio2_range": (0.3, 0.6),
            "peep_range": (5, 10),
        },
        "High Flow NC": {
            "modes": [None],
            "fio2_range": (0.3, 1.0),
            "flow_rate_range": (20, 60),
        },
        "Face Mask": {
            "modes": [None],
            "fio2_range": (0.28, 0.5),
            "lpm_range": (6, 15),
        },
        "Nasal Cannula": {
            "modes": [None],
            "fio2_range": (0.24, 0.44),
            "lpm_range": (1, 6),
        },
        "Trach Collar": {
            "modes": [None],
            "fio2_range": (0.28, 0.5),
        },
        "Room Air": {
            "modes": [None],
            "fio2_range": (0.21, 0.21),
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        vitals_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate respiratory support data for hospitalizations.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            vitals_df: Optional vitals table for SpO2 correlation

        Returns:
            DataFrame with respiratory_support table columns
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

            hosp_resp = self._generate_hospitalization_respiratory(
                hosp_id, admit_time, discharge_time
            )
            records.extend(hosp_resp)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _generate_hospitalization_respiratory(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate respiratory support for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # Determine initial respiratory status
        # ~40% need some oxygen, ~15% need mechanical ventilation
        initial_status = self.rng.choice(
            ["room_air", "nasal_cannula", "high_flow", "nippv", "imv"],
            p=[0.45, 0.25, 0.10, 0.08, 0.12],
        )

        device_map = {
            "room_air": "Room Air",
            "nasal_cannula": "Nasal Cannula",
            "high_flow": "High Flow NC",
            "nippv": "NIPPV",
            "imv": "IMV",
        }
        current_device = device_map[initial_status]

        # Determine if patient has tracheostomy
        has_trach = initial_status == "imv" and los_hours > 168 and self.rng.random() < 0.3
        trach_time = None
        if has_trach:
            trach_time = admit_time + timedelta(hours=self.rng.uniform(120, 240))

        # Generate timestamps (hourly for ventilated, less frequent otherwise)
        mean_interval = 1.0 if initial_status in ["imv", "nippv"] else 4.0
        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=mean_interval,
            cv=0.25,
            rng=self.rng,
        )

        # Track device trajectory
        current_fio2 = self._get_initial_fio2(current_device)
        improving = self.rng.random() < 0.7  # 70% improve over stay

        for ts in timestamps:
            # Check for tracheostomy
            trach = has_trach and trach_time and ts >= trach_time

            # Evolve respiratory status
            current_device, current_fio2 = self._evolve_respiratory_status(
                current_device, current_fio2, improving, trach
            )

            record = self._generate_respiratory_record(
                hospitalization_id, ts, current_device, trach
            )
            records.append(record)

        return records

    def _get_initial_fio2(self, device: str) -> float:
        """Get initial FiO2 for a device."""
        settings = self.DEVICE_SETTINGS.get(device, {})
        fio2_range = settings.get("fio2_range", (0.21, 0.21))
        return self.rng.uniform(fio2_range[0], fio2_range[1])

    def _evolve_respiratory_status(
        self,
        current_device: str,
        current_fio2: float,
        improving: bool,
        has_trach: bool,
    ) -> tuple[str, float]:
        """Evolve respiratory status with small probability of change."""
        # Device escalation/de-escalation hierarchy
        device_hierarchy = [
            "Room Air",
            "Nasal Cannula",
            "Face Mask",
            "High Flow NC",
            "NIPPV",
            "IMV",
        ]

        if current_device not in device_hierarchy:
            current_device = "Room Air"

        current_idx = device_hierarchy.index(current_device)

        # Small probability of change per time step
        if self.rng.random() < 0.02:  # 2% chance
            if improving:
                # Wean (go down the hierarchy)
                if current_idx > 0:
                    current_device = device_hierarchy[current_idx - 1]
                    current_fio2 = max(0.21, current_fio2 - 0.1)
            else:
                # Escalate (go up the hierarchy)
                if current_idx < len(device_hierarchy) - 1:
                    current_device = device_hierarchy[current_idx + 1]
                    current_fio2 = min(1.0, current_fio2 + 0.1)

        # Trach patients stay on trach collar or IMV
        if has_trach and current_device not in ["IMV", "Trach Collar"]:
            current_device = "Trach Collar"

        # Small FiO2 adjustments
        if self.rng.random() < 0.1:
            delta = self.rng.uniform(-0.05, 0.05)
            if improving:
                delta -= 0.02
            else:
                delta += 0.02
            current_fio2 = np.clip(current_fio2 + delta, 0.21, 1.0)

        return current_device, current_fio2

    def _generate_respiratory_record(
        self,
        hospitalization_id: str,
        timestamp: datetime,
        device: str,
        has_trach: bool,
    ) -> dict:
        """Generate a single respiratory support record per CLIF 2.1.0 schema."""
        settings = self.DEVICE_SETTINGS.get(device, self.DEVICE_SETTINGS["Room Air"])

        record = {
            "hospitalization_id": hospitalization_id,
            "recorded_dttm": timestamp,
            "device_category": device,
            "vent_brand_name": None,
            "mode_category": None,
            "tracheostomy": int(has_trach),
            "fio2_set": None,
            "lpm_set": None,
            "tidal_volume_set": None,
            "resp_rate_set": None,
            "pressure_control_set": None,
            "pressure_support_set": None,
            "flow_rate_set": None,
            "peak_inspiratory_pressure_set": None,
            "inspiratory_time_set": None,
            "peep_set": None,
            # Observed values
            "tidal_volume_obs": None,
            "resp_rate_obs": None,
            "plateau_pressure_obs": None,
            "peak_inspiratory_pressure_obs": None,
            "peep_obs": None,
            "minute_vent_obs": None,
            "mean_airway_pressure_obs": None,
        }

        # Mode
        modes = settings.get("modes", [None])
        record["mode_category"] = self.rng.choice(modes) if modes[0] else None

        # FiO2
        if "fio2_range" in settings:
            record["fio2_set"] = round(
                self.rng.uniform(*settings["fio2_range"]), 2
            )

        # LPM (for nasal cannula, mask)
        if "lpm_range" in settings:
            record["lpm_set"] = round(self.rng.uniform(*settings["lpm_range"]), 0)

        # Flow rate (for high flow)
        if "flow_rate_range" in settings:
            record["flow_rate_set"] = round(
                self.rng.uniform(*settings["flow_rate_range"]), 0
            )

        # PEEP
        if "peep_range" in settings:
            record["peep_set"] = round(self.rng.uniform(*settings["peep_range"]), 0)

        # Ventilator-specific settings
        if device == "IMV":
            record["vent_brand_name"] = self.rng.choice(self.VENT_BRANDS)

            record["tidal_volume_set"] = round(
                self.rng.uniform(*settings["tidal_volume_range"]), 0
            )
            record["resp_rate_set"] = round(
                self.rng.uniform(*settings["resp_rate_range"]), 0
            )
            record["inspiratory_time_set"] = round(
                self.rng.uniform(0.8, 1.5), 2
            )

            mode = record["mode_category"]
            if mode in ["Pressure Control"]:
                record["pressure_control_set"] = round(
                    self.rng.uniform(*settings["pressure_control_range"]), 0
                )
            if mode in ["Pressure Support/CPAP", "SIMV"]:
                record["pressure_support_set"] = round(
                    self.rng.uniform(*settings["pressure_support_range"]), 0
                )

            # Set PIP
            record["peak_inspiratory_pressure_set"] = round(
                self.rng.uniform(15, 40), 0
            )

            # Observed values (slight variation from set values)
            tv_set = record["tidal_volume_set"]
            record["tidal_volume_obs"] = round(tv_set + self.rng.normal(0, 30), 0)
            rr_set = record["resp_rate_set"]
            record["resp_rate_obs"] = round(rr_set + self.rng.normal(2, 3), 0)
            record["peak_inspiratory_pressure_obs"] = round(
                record["peak_inspiratory_pressure_set"] + self.rng.normal(0, 2), 0
            )
            if self.rng.random() < 0.7:
                record["plateau_pressure_obs"] = round(
                    self.rng.uniform(12, 30), 0
                )
            peep_set = record["peep_set"]
            if peep_set is not None:
                record["peep_obs"] = round(peep_set + self.rng.normal(0, 0.5), 0)
            record["minute_vent_obs"] = round(self.rng.uniform(5, 15), 1)
            record["mean_airway_pressure_obs"] = round(self.rng.uniform(8, 25), 0)

        elif device == "NIPPV":
            record["pressure_support_set"] = round(
                self.rng.uniform(*settings["pressure_support_range"]), 0
            )

        return record
