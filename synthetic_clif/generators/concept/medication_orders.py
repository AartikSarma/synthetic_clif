"""Medication orders generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator


class MedicationOrdersGenerator(BaseGenerator):
    """Generate synthetic medication order data.

    Creates medication_orders table linking to admin tables.
    """

    # Mapping from med_category to med_group (mirrors medications.py generators)
    MED_GROUP_CONTINUOUS = {
        "norepinephrine": "vasoactives",
        "vasopressin": "vasoactives",
        "epinephrine": "vasoactives",
        "phenylephrine": "vasoactives",
        "dobutamine": "vasoactives",
        "milrinone": "vasoactives",
        "propofol": "sedation",
        "dexmedetomidine": "sedation",
        "midazolam": "sedation",
        "fentanyl": "sedation",
        "ketamine": "sedation",
        "heparin": "anticoagulation",
        "argatroban": "anticoagulation",
        "bivalirudin": "anticoagulation",
        "insulin": "endocrine",
    }

    MED_GROUP_INTERMITTENT = {
        "vancomycin": "CMS_sepsis_qualifying_antibiotics",
        "piperacillin_tazobactam": "CMS_sepsis_qualifying_antibiotics",
        "cefepime": "CMS_sepsis_qualifying_antibiotics",
        "meropenem": "CMS_sepsis_qualifying_antibiotics",
        "ceftriaxone": "CMS_sepsis_qualifying_antibiotics",
        "metronidazole": "CMS_sepsis_qualifying_antibiotics",
        "pantoprazole": "other",
        "metoprolol": "other",
        "enoxaparin": "other",
        "acetaminophen": "other",
        "ondansetron": "other",
    }

    # Intermittent medication frequencies
    INTERMITTENT_FREQUENCIES = {
        "vancomycin": "Q12H",
        "piperacillin_tazobactam": "Q6H",
        "cefepime": "Q8H",
        "meropenem": "Q8H",
        "ceftriaxone": "Q24H",
        "metronidazole": "Q8H",
        "pantoprazole": "Q24H",
        "metoprolol": "Q6H",
        "enoxaparin": "Q12H",
        "acetaminophen": "Q6H",
        "ondansetron": "Q8H",
    }

    # Common medication orders
    MED_ORDERS = {
        "norepinephrine": {"route": "IV", "dose": "0.1", "unit": "mcg/kg/min"},
        "propofol": {"route": "IV", "dose": "50", "unit": "mcg/kg/min"},
        "fentanyl": {"route": "IV", "dose": "100", "unit": "mcg/hr"},
        "vancomycin": {"route": "IV", "dose": "1500", "unit": "mg"},
        "piperacillin_tazobactam": {"route": "IV", "dose": "4500", "unit": "mg"},
        "pantoprazole": {"route": "IV", "dose": "40", "unit": "mg"},
        "heparin": {"route": "IV", "dose": "1000", "unit": "units/hr"},
        "insulin": {"route": "IV", "dose": "5", "unit": "units/hr"},
        "metoprolol": {"route": "PO", "dose": "50", "unit": "mg"},
        "enoxaparin": {"route": "SC", "dose": "40", "unit": "mg"},
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        med_continuous_df: Optional[pd.DataFrame] = None,
        med_intermittent_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate medication orders.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            med_continuous_df: Optional continuous med admin table
            med_intermittent_df: Optional intermittent med admin table

        Returns:
            DataFrame with medication_orders columns
        """
        records = []
        seen_orders = set()

        # Extract orders from continuous meds
        if med_continuous_df is not None and len(med_continuous_df) > 0:
            # For continuous meds, find order start/end by grouping on med_order_id
            for order_id, group in med_continuous_df.groupby("med_order_id"):
                if order_id in seen_orders:
                    continue
                seen_orders.add(order_id)

                first_row = group.iloc[0]
                med_cat = first_row.get("med_category", "")
                med_info = self.MED_ORDERS.get(med_cat, {})
                route = med_info.get("route") or first_row.get("med_route_category", "IV")

                order_start = group["admin_dttm"].min()
                order_end = group["admin_dttm"].max()
                # Add some time after last admin
                if pd.notna(order_end):
                    order_end = order_end + timedelta(hours=1)
                ordered_time = order_start - timedelta(minutes=self.rng.uniform(5, 30)) if pd.notna(order_start) else order_start

                med_group = self.MED_GROUP_CONTINUOUS.get(med_cat, "others")

                status_name = "Active" if pd.isna(order_end) else self.rng.choice(
                    ["Completed", "Active", "Discontinued"],
                    p=[0.6, 0.3, 0.1],
                )

                records.append(
                    {
                        "hospitalization_id": first_row["hospitalization_id"],
                        "med_order_id": order_id,
                        "order_start_dttm": order_start,
                        "order_end_dttm": order_end,
                        "ordered_dttm": ordered_time,
                        "med_category": med_cat,
                        "med_name": first_row.get("med_name"),
                        "med_group": med_group,
                        "med_dose": float(med_info.get("dose", 0)) or first_row.get("med_dose"),
                        "med_dose_unit": med_info.get("unit") or first_row.get("med_dose_unit"),
                        "med_order_status_name": status_name,
                        "med_order_status_category": status_name.lower(),
                        "med_route_name": self.name_from_category(route.lower()),
                        "med_frequency": "Continuous",
                        "prn": False,
                    }
                )

        # Extract orders from intermittent meds
        if med_intermittent_df is not None and len(med_intermittent_df) > 0:
            for order_id, group in med_intermittent_df.groupby("med_order_id"):
                if order_id in seen_orders:
                    continue
                seen_orders.add(order_id)

                first_row = group.iloc[0]
                med_cat = first_row.get("med_category", "")
                med_info = self.MED_ORDERS.get(med_cat, {})
                route = med_info.get("route") or first_row.get("med_route_category", "IV")

                order_start = group["admin_dttm"].min()
                # Estimate order end based on typical course length (3-7 days for antibiotics, shorter for others)
                course_days = self.rng.uniform(3, 7) if med_cat in self.MED_GROUP_INTERMITTENT and self.MED_GROUP_INTERMITTENT.get(med_cat) == "CMS_sepsis_qualifying_antibiotics" else self.rng.uniform(1, 5)
                order_end = order_start + timedelta(days=course_days) if pd.notna(order_start) else None
                ordered_time = order_start - timedelta(minutes=self.rng.uniform(5, 30)) if pd.notna(order_start) else order_start

                med_group = self.MED_GROUP_INTERMITTENT.get(med_cat, "other")
                frequency = self.INTERMITTENT_FREQUENCIES.get(med_cat, "Q8H")

                # PRN for analgesics/antiemetics
                is_prn = med_cat in ("acetaminophen", "ondansetron") and self.rng.random() < 0.5

                status_name = self.rng.choice(
                    ["Completed", "Active", "Discontinued"],
                    p=[0.7, 0.25, 0.05],
                )

                records.append(
                    {
                        "hospitalization_id": first_row["hospitalization_id"],
                        "med_order_id": order_id,
                        "order_start_dttm": order_start,
                        "order_end_dttm": order_end,
                        "ordered_dttm": ordered_time,
                        "med_category": med_cat,
                        "med_name": first_row.get("med_name"),
                        "med_group": med_group,
                        "med_dose": float(med_info.get("dose", 0)) or first_row.get("med_dose"),
                        "med_dose_unit": med_info.get("unit") or first_row.get("med_dose_unit"),
                        "med_order_status_name": status_name,
                        "med_order_status_category": status_name.lower(),
                        "med_route_name": self.name_from_category(route.lower()),
                        "med_frequency": frequency,
                        "prn": is_prn,
                    }
                )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_start_dttm"] = pd.to_datetime(df["order_start_dttm"], utc=True)
            df["order_end_dttm"] = pd.to_datetime(df["order_end_dttm"], utc=True)
            df["ordered_dttm"] = pd.to_datetime(df["ordered_dttm"], utc=True)

        return df
