"""Tests for labs generator."""

import pytest
import pandas as pd

from synthetic_clif.generators.labs import LabsGenerator


class TestLabsGenerator:
    """Tests for LabsGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic labs generation."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert len(df) > 0
        assert "hospitalization_id" in df.columns
        assert "lab_order_dttm" in df.columns
        assert "lab_collect_dttm" in df.columns
        assert "lab_result_dttm" in df.columns
        assert "lab_category" in df.columns
        assert "lab_value" in df.columns
        assert "lab_value_numeric" in df.columns
        assert "reference_unit" in df.columns
        assert "lab_type_category" in df.columns

    def test_lab_categories_valid(self, hospitalizations_df, seed, mcide):
        """Test that lab categories are valid mCIDE values."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_labs = set(mcide.get_category("lab"))
        for lc in df["lab_category"].dropna():
            assert lc in valid_labs

    def test_timestamp_ordering(self, hospitalizations_df, seed, mcide):
        """Test that lab timestamps are properly ordered."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        for _, row in df.iterrows():
            if pd.notna(row["lab_order_dttm"]) and pd.notna(row["lab_collect_dttm"]):
                assert row["lab_collect_dttm"] >= row["lab_order_dttm"]
            if pd.notna(row["lab_collect_dttm"]) and pd.notna(row["lab_result_dttm"]):
                assert row["lab_result_dttm"] >= row["lab_collect_dttm"]

    def test_lab_values_reasonable(self, hospitalizations_df, seed, mcide):
        """Test that lab values are physiologically reasonable."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # Check some common labs
        bounds = {
            "sodium": (120, 160),
            "potassium": (2.5, 7.0),
            "creatinine": (0.3, 10.0),
            "hemoglobin": (5, 18),
            "ph": (7.0, 7.6),
        }

        for lab_cat, (lower, upper) in bounds.items():
            values = df[df["lab_category"] == lab_cat]["lab_value_numeric"].dropna()
            if len(values) > 0:
                assert values.min() >= lower
                assert values.max() <= upper

    def test_reference_units(self, hospitalizations_df, seed, mcide):
        """Test that reference units are appropriate."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        expected_units = mcide.get_lab_reference_units()

        for _, row in df.iterrows():
            lab_cat = row["lab_category"]
            if lab_cat in expected_units and pd.notna(row["reference_unit"]):
                assert row["reference_unit"] == expected_units[lab_cat]

    def test_lab_type_valid(self, hospitalizations_df, seed, mcide):
        """Test that lab types are valid mCIDE values."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_types = set(mcide.get_category("lab_type"))
        for lt in df["lab_type_category"].dropna():
            assert lt in valid_types

    def test_admission_labs(self, hospitalizations_df, seed, mcide):
        """Test that admission labs are generated."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # Each hospitalization should have labs early in admission
        hosp_lookup = hospitalizations_df.set_index("hospitalization_id")

        for hosp_id in df["hospitalization_id"].unique():
            hosp = hosp_lookup.loc[hosp_id]
            hosp_labs = df[df["hospitalization_id"] == hosp_id]

            # Check for labs within first few hours
            early_labs = hosp_labs[
                (hosp_labs["lab_result_dttm"] - hosp["admission_dttm"]).dt.total_seconds()
                < 6 * 3600
            ]
            assert len(early_labs) > 0, f"No early labs for {hosp_id}"
