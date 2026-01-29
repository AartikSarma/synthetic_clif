"""Tests for patient generator."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from synthetic_clif.generators.patient import PatientGenerator


class TestPatientGenerator:
    """Tests for PatientGenerator."""

    def test_generate_basic(self, seed, mcide):
        """Test basic patient generation."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=10)

        assert len(df) == 10
        assert "patient_id" in df.columns
        assert "sex_category" in df.columns
        assert "race_category" in df.columns
        assert "ethnicity_category" in df.columns
        assert "birth_date" in df.columns
        assert "death_dttm" in df.columns

    def test_patient_ids_unique(self, seed, mcide):
        """Test that patient IDs are unique."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=100)

        assert df["patient_id"].nunique() == 100

    def test_patient_ids_uuid_format(self, seed, mcide):
        """Test that patient IDs are in UUID format."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=10)

        for pid in df["patient_id"]:
            parts = pid.split("-")
            assert len(parts) == 5
            assert len(parts[0]) == 8
            assert len(parts[1]) == 4
            assert len(parts[2]) == 4
            assert len(parts[3]) == 4
            assert len(parts[4]) == 12

    def test_sex_category_valid(self, seed, mcide):
        """Test that sex categories are valid mCIDE values."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=100)

        valid_sexes = set(mcide.get_category("sex"))
        for sex in df["sex_category"].dropna():
            assert sex in valid_sexes

    def test_race_category_valid(self, seed, mcide):
        """Test that race categories are valid mCIDE values."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=100)

        valid_races = set(mcide.get_category("race"))
        for race in df["race_category"].dropna():
            assert race in valid_races

    def test_ethnicity_category_valid(self, seed, mcide):
        """Test that ethnicity categories are valid mCIDE values."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=100)

        valid_ethnicities = set(mcide.get_category("ethnicity"))
        for eth in df["ethnicity_category"].dropna():
            assert eth in valid_ethnicities

    def test_birth_date_reasonable(self, seed, mcide, reference_date):
        """Test that birth dates produce reasonable ages."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=100, reference_date=reference_date)

        for _, row in df.iterrows():
            if pd.notna(row["birth_date"]):
                age = (reference_date.date() - row["birth_date"].date()).days / 365.25
                # Allow small tolerance for boundary cases
                assert 17.9 <= age <= 96

    def test_mortality_rate(self, seed, mcide):
        """Test that mortality rate is approximately correct."""
        gen = PatientGenerator(seed=seed, mcide=mcide)
        df = gen.generate(n_patients=1000, mortality_rate=0.15)

        death_rate = df["death_dttm"].notna().sum() / len(df)
        assert 0.10 <= death_rate <= 0.20  # Allow some variance

    def test_reproducibility(self, seed, mcide):
        """Test that same seed produces same results."""
        gen1 = PatientGenerator(seed=seed, mcide=mcide)
        df1 = gen1.generate(n_patients=10)

        gen2 = PatientGenerator(seed=seed, mcide=mcide)
        df2 = gen2.generate(n_patients=10)

        # Check non-datetime columns match exactly
        non_dt_cols = [c for c in df1.columns if c != "death_dttm"]
        pd.testing.assert_frame_equal(df1[non_dt_cols], df2[non_dt_cols])

        # Check death_dttm matches (accounting for potential timestamp precision issues)
        assert (df1["death_dttm"].isna() == df2["death_dttm"].isna()).all()
