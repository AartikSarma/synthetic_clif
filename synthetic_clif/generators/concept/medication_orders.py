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
            for _, row in med_continuous_df.iterrows():
                order_id = row.get("med_order_id")
                if order_id and order_id not in seen_orders:
                    seen_orders.add(order_id)
                    med_cat = row.get("med_category", "")
                    med_info = self.MED_ORDERS.get(med_cat, {})

                    records.append(
                        {
                            "hospitalization_id": row["hospitalization_id"],
                            "med_order_id": order_id,
                            "order_dttm": row["admin_dttm"],
                            "med_category": med_cat,
                            "med_name": row.get("med_name"),
                            "med_dose": float(med_info.get("dose", 0)) or row.get("med_dose"),
                            "med_dose_unit": med_info.get("unit") or row.get("med_dose_unit"),
                            "med_route_category": med_info.get("route") or row.get("med_route_category"),
                            "order_status": "Active",
                        }
                    )

        # Extract orders from intermittent meds
        if med_intermittent_df is not None and len(med_intermittent_df) > 0:
            for _, row in med_intermittent_df.iterrows():
                order_id = row.get("med_order_id")
                if order_id and order_id not in seen_orders:
                    seen_orders.add(order_id)
                    med_cat = row.get("med_category", "")
                    med_info = self.MED_ORDERS.get(med_cat, {})

                    records.append(
                        {
                            "hospitalization_id": row["hospitalization_id"],
                            "med_order_id": order_id,
                            "order_dttm": row["admin_dttm"],
                            "med_category": med_cat,
                            "med_name": row.get("med_name"),
                            "med_dose": float(med_info.get("dose", 0)) or row.get("med_dose"),
                            "med_dose_unit": med_info.get("unit") or row.get("med_dose_unit"),
                            "med_route_category": med_info.get("route") or row.get("med_route_category"),
                            "order_status": "Completed",
                        }
                    )

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)

        return df
