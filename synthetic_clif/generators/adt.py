"""ADT (Admit/Discharge/Transfer) table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader


class ADTGenerator(BaseGenerator):
    """Generate synthetic ADT events.

    Creates adt table with location transfers during hospitalization:
    - hospitalization_id (foreign key)
    - in_dttm, out_dttm (contiguous and within hospitalization bounds)
    - location_category (mCIDE category)

    Typical flow: ED → ICU → Stepdown → Ward → Discharge
    """

    # Common location flow patterns
    FLOW_PATTERNS = {
        "emergency_to_icu": ["ed", "icu", "stepdown", "ward"],
        "direct_icu": ["icu", "stepdown", "ward"],
        "stepdown_only": ["ed", "stepdown", "ward"],
        "ward_only": ["ed", "ward"],
        "short_icu": ["icu", "ward"],
    }

    FLOW_WEIGHTS = [0.35, 0.25, 0.15, 0.15, 0.10]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate ADT events for each hospitalization.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with ADT table columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            # Handle missing discharge time
            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            # Ensure timezone
            if admit_time.tzinfo is None:
                admit_time = admit_time.replace(tzinfo=timezone.utc)
            if discharge_time.tzinfo is None:
                discharge_time = discharge_time.replace(tzinfo=timezone.utc)

            # Generate location sequence
            adt_events = self._generate_location_sequence(
                hosp_id, admit_time, discharge_time
            )
            records.extend(adt_events)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["in_dttm"] = pd.to_datetime(df["in_dttm"], utc=True)
            df["out_dttm"] = pd.to_datetime(df["out_dttm"], utc=True)

        return df

    def _generate_location_sequence(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate sequence of location transfers."""
        total_hours = (discharge_time - admit_time).total_seconds() / 3600

        if total_hours <= 0:
            return [
                {
                    "hospitalization_id": hospitalization_id,
                    "in_dttm": admit_time,
                    "out_dttm": discharge_time,
                    "location_category": "icu",
                }
            ]

        # Select flow pattern based on weights
        pattern_names = list(self.FLOW_PATTERNS.keys())
        pattern_idx = self.rng.choice(len(pattern_names), p=self.FLOW_WEIGHTS)
        locations = self.FLOW_PATTERNS[pattern_names[pattern_idx]].copy()

        # Adjust pattern based on LOS
        if total_hours < 24:
            # Very short stay - single location
            locations = [locations[0] if locations else "icu"]
        elif total_hours < 72:
            # Short stay - max 2 locations
            locations = locations[:2] if len(locations) > 2 else locations

        n_locations = len(locations)

        # Distribute time across locations
        # ICU gets more time early, ward gets more time late
        time_weights = self._get_time_weights(locations, total_hours)
        location_hours = [w * total_hours for w in time_weights]

        # Generate ADT events
        events = []
        current_time = admit_time

        for i, (location, hours) in enumerate(zip(locations, location_hours)):
            end_time = current_time + timedelta(hours=hours)

            # Last location ends at discharge
            if i == n_locations - 1:
                end_time = discharge_time

            # Add some randomness to transfer times
            if i > 0 and i < n_locations - 1:
                jitter_hours = self.rng.uniform(-0.5, 0.5)
                end_time += timedelta(hours=jitter_hours)
                end_time = min(end_time, discharge_time)

            events.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "in_dttm": current_time,
                    "out_dttm": end_time,
                    "location_category": location,
                }
            )

            current_time = end_time

        return events

    def _get_time_weights(
        self, locations: list[str], total_hours: float
    ) -> list[float]:
        """Calculate time weights for each location."""
        n = len(locations)
        weights = []

        for i, loc in enumerate(locations):
            if loc == "ed":
                # ED: short stay (2-8 hours)
                weights.append(max(2, min(8, total_hours * 0.05)) / total_hours)
            elif loc == "icu":
                # ICU: variable, usually substantial portion
                weights.append(0.4 if n > 2 else 0.6)
            elif loc == "stepdown":
                weights.append(0.25)
            elif loc == "ward":
                weights.append(0.3)
            else:
                weights.append(1.0 / n)

        # Normalize
        total = sum(weights)
        return [w / total for w in weights]
