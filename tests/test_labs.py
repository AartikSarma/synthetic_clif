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
        assert "lab_order_category" in df.columns

    def test_timestamp_ordering(self, hospitalizations_df, seed, mcide):
        """Test that lab timestamps are properly ordered."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        for _, row in df.iterrows():
            if pd.notna(row["lab_order_dttm"]) and pd.notna(row["lab_collect_dttm"]):
                assert row["lab_collect_dttm"] >= row["lab_order_dttm"]
            if pd.notna(row["lab_collect_dttm"]) and pd.notna(row["lab_result_dttm"]):
                assert row["lab_result_dttm"] >= row["lab_collect_dttm"]

    def test_blood_gas_arterial_venous_categories(self, hospitalizations_df, seed, mcide):
        """Test that blood gas categories use arterial/venous distinction."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        blood_gas_cats = df["lab_category"].unique()

        # Should not contain generic categories
        assert "pco2" not in blood_gas_cats, "Generic 'pco2' found; should be pco2_arterial/pco2_venous"
        assert "po2" not in blood_gas_cats, "Generic 'po2' found; should be po2_arterial/po2_venous"
        assert "ph" not in blood_gas_cats, "Generic 'ph' found; should be ph_arterial/ph_venous"

        # Should contain arterial variants (arterial draws are more common, so should always appear)
        assert "pco2_arterial" in blood_gas_cats
        assert "po2_arterial" in blood_gas_cats
        assert "ph_arterial" in blood_gas_cats

    def test_blood_gas_arterial_venous_ratio(self, hospitalizations_df, seed, mcide):
        """Test that ~70% of blood gas draws are arterial."""
        gen = LabsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        arterial_count = (df["lab_category"] == "pco2_arterial").sum()
        venous_count = (df["lab_category"] == "pco2_venous").sum()
        total = arterial_count + venous_count

        if total > 0:
            arterial_fraction = arterial_count / total
            # Allow wide tolerance: expect 50-90% arterial
            assert 0.5 <= arterial_fraction <= 0.9, (
                f"Arterial fraction {arterial_fraction:.2f} out of expected range [0.5, 0.9]"
            )

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
