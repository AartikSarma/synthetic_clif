"""Tests for hospitalization generator."""

import pytest
import pandas as pd
from datetime import timedelta

from synthetic_clif.generators.hospitalization import HospitalizationGenerator


class TestHospitalizationGenerator:
    """Tests for HospitalizationGenerator."""

    def test_generate_basic(self, patients_df, seed, mcide):
        """Test basic hospitalization generation."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=15)

        assert len(df) == 15
        assert "hospitalization_id" in df.columns
        assert "patient_id" in df.columns
        assert "admission_dttm" in df.columns
        assert "discharge_dttm" in df.columns
        assert "age_at_admission" in df.columns
        assert "admission_type_category" in df.columns
        assert "discharge_category" in df.columns

    def test_hospitalization_ids_unique(self, patients_df, seed, mcide):
        """Test that hospitalization IDs are unique."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=50)

        assert df["hospitalization_id"].nunique() == len(df)

    def test_patient_ids_exist(self, patients_df, seed, mcide):
        """Test that all patient IDs exist in patient table."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=20)

        valid_patient_ids = set(patients_df["patient_id"])
        for pid in df["patient_id"]:
            assert pid in valid_patient_ids

    def test_discharge_after_admission(self, patients_df, seed, mcide):
        """Test that discharge is after admission."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=50)

        for _, row in df.iterrows():
            if pd.notna(row["discharge_dttm"]):
                assert row["discharge_dttm"] >= row["admission_dttm"]

    def test_los_reasonable(self, patients_df, seed, mcide):
        """Test that length of stay is reasonable."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=100)

        for _, row in df.iterrows():
            if pd.notna(row["discharge_dttm"]):
                los_days = (
                    row["discharge_dttm"] - row["admission_dttm"]
                ).total_seconds() / (24 * 3600)
                assert 0.5 <= los_days <= 90

    def test_admission_type_valid(self, patients_df, seed, mcide):
        """Test that admission types are valid mCIDE values."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=50)

        valid_types = set(mcide.get_category("admission_type"))
        for at in df["admission_type_category"].dropna():
            assert at in valid_types

    def test_discharge_category_valid(self, patients_df, seed, mcide):
        """Test that discharge categories are valid mCIDE values."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=50)

        valid_categories = set(mcide.get_category("discharge"))
        for dc in df["discharge_category"].dropna():
            assert dc in valid_categories

    def test_age_at_admission_reasonable(self, patients_df, seed, mcide):
        """Test that age at admission is reasonable."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        df = gen.generate(patients_df, n_hospitalizations=50)

        for age in df["age_at_admission"].dropna():
            assert 18 <= age <= 100

    def test_readmissions(self, patients_df, seed, mcide):
        """Test that some patients have multiple hospitalizations."""
        gen = HospitalizationGenerator(seed=seed, mcide=mcide)
        # Generate more hospitalizations than patients
        df = gen.generate(patients_df, n_hospitalizations=25)

        hosp_counts = df.groupby("patient_id").size()
        assert hosp_counts.max() > 1  # At least one patient has multiple
