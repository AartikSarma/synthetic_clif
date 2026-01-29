"""Tests for vitals generator."""

import pytest
import pandas as pd
import numpy as np

from synthetic_clif.generators.vitals import VitalsGenerator
from synthetic_clif.generators.adt import ADTGenerator


class TestVitalsGenerator:
    """Tests for VitalsGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic vitals generation."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert len(df) > 0
        assert "hospitalization_id" in df.columns
        assert "recorded_dttm" in df.columns
        assert "vital_category" in df.columns
        assert "vital_value" in df.columns
        assert "meas_site_category" in df.columns

    def test_vital_categories_valid(self, hospitalizations_df, seed, mcide):
        """Test that vital categories are valid mCIDE values."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_vitals = set(mcide.get_category("vital"))
        for vc in df["vital_category"].dropna():
            assert vc in valid_vitals

    def test_timestamps_within_hospitalization(self, hospitalizations_df, seed, mcide):
        """Test that vital timestamps are within hospitalization bounds."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        hosp_lookup = hospitalizations_df.set_index("hospitalization_id")

        for _, row in df.iterrows():
            hosp_id = row["hospitalization_id"]
            hosp = hosp_lookup.loc[hosp_id]
            assert row["recorded_dttm"] >= hosp["admission_dttm"]
            if pd.notna(hosp["discharge_dttm"]):
                assert row["recorded_dttm"] <= hosp["discharge_dttm"]

    def test_vital_values_reasonable(self, hospitalizations_df, seed, mcide):
        """Test that vital values are physiologically reasonable."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        bounds = {
            "heart_rate": (30, 200),
            "sbp": (60, 250),
            "dbp": (30, 150),
            "spo2": (50, 100),
            "respiratory_rate": (6, 50),
            "temp_c": (34, 42),
            "map": (40, 160),
        }

        for vital_cat, (lower, upper) in bounds.items():
            values = df[df["vital_category"] == vital_cat]["vital_value"].dropna()
            if len(values) > 0:
                assert values.min() >= lower
                assert values.max() <= upper

    def test_temporal_consistency(self, hospitalizations_df, seed, mcide):
        """Test that consecutive vitals don't jump unrealistically."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # Check heart rate doesn't jump more than 50 bpm between measurements
        for hosp_id in df["hospitalization_id"].unique():
            hr_values = (
                df[(df["hospitalization_id"] == hosp_id) & (df["vital_category"] == "heart_rate")]
                .sort_values("recorded_dttm")["vital_value"]
                .dropna()
            )

            if len(hr_values) > 1:
                diffs = hr_values.diff().dropna().abs()
                # Most changes should be small (< 30 bpm)
                assert (diffs < 30).mean() > 0.8

    def test_with_adt(self, hospitalizations_df, seed, mcide):
        """Test that ADT affects measurement frequency."""
        adt_gen = ADTGenerator(seed=seed, mcide=mcide)
        adt_df = adt_gen.generate(hospitalizations_df)

        vitals_gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = vitals_gen.generate(hospitalizations_df, adt_df)

        assert len(df) > 0

    def test_missingness(self, hospitalizations_df, seed, mcide):
        """Test that missingness is introduced."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df, missingness_rate=0.05)

        # Some values should be missing
        missing_rate = df["vital_value"].isna().mean()
        assert missing_rate > 0

    def test_outliers(self, hospitalizations_df, seed, mcide):
        """Test that outliers are introduced."""
        gen = VitalsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df, outlier_rate=0.01)

        # Check that there are some extreme values
        hr_values = df[df["vital_category"] == "heart_rate"]["vital_value"].dropna()
        if len(hr_values) > 100:
            # Should have some values outside typical range
            extreme_count = ((hr_values < 50) | (hr_values > 120)).sum()
            assert extreme_count > 0
