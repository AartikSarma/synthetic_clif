"""Tests for patient assessments generator."""

import pytest
import pandas as pd

from synthetic_clif.generators.assessments import PatientAssessmentsGenerator


class TestPatientAssessmentsGenerator:
    """Tests for PatientAssessmentsGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic assessments generation."""
        gen = PatientAssessmentsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert len(df) > 0
        assert "hospitalization_id" in df.columns
        assert "recorded_dttm" in df.columns
        assert "assessment_category" in df.columns
        assert "assessment_value" in df.columns
        assert "assessment_value_text" in df.columns

    def test_assessment_categories_valid(self, hospitalizations_df, seed, mcide):
        """Test that assessment categories are valid mCIDE values."""
        gen = PatientAssessmentsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_assessments = set(mcide.get_category("assessment"))
        for ac in df["assessment_category"].dropna():
            assert ac in valid_assessments

    def test_gcs_values_valid(self, hospitalizations_df, seed, mcide):
        """Test that GCS values are in valid ranges."""
        gen = PatientAssessmentsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # GCS total: 3-15
        gcs_total = df[df["assessment_category"] == "gcs_total"]["assessment_value"].dropna()
        if len(gcs_total) > 0:
            assert gcs_total.min() >= 3
            assert gcs_total.max() <= 15

        # GCS eye: 1-4
        gcs_eye = df[df["assessment_category"] == "gcs_eye"]["assessment_value"].dropna()
        if len(gcs_eye) > 0:
            assert gcs_eye.min() >= 1
            assert gcs_eye.max() <= 4

        # GCS verbal: 1-5
        gcs_verbal = df[df["assessment_category"] == "gcs_verbal"]["assessment_value"].dropna()
        if len(gcs_verbal) > 0:
            assert gcs_verbal.min() >= 1
            assert gcs_verbal.max() <= 5

        # GCS motor: 1-6
        gcs_motor = df[df["assessment_category"] == "gcs_motor"]["assessment_value"].dropna()
        if len(gcs_motor) > 0:
            assert gcs_motor.min() >= 1
            assert gcs_motor.max() <= 6

    def test_rass_values_valid(self, hospitalizations_df, seed, mcide):
        """Test that RASS values are in valid range."""
        gen = PatientAssessmentsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        rass = df[df["assessment_category"] == "rass"]["assessment_value"].dropna()
        if len(rass) > 0:
            assert rass.min() >= -5
            assert rass.max() <= 4

    def test_pain_score_valid(self, hospitalizations_df, seed, mcide):
        """Test that pain scores are in valid range."""
        gen = PatientAssessmentsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        pain = df[df["assessment_category"] == "pain_score"]["assessment_value"].dropna()
        if len(pain) > 0:
            assert pain.min() >= 0
            assert pain.max() <= 10

    def test_gcs_components_sum(self, hospitalizations_df, seed, mcide):
        """Test that GCS components sum to total."""
        gen = PatientAssessmentsGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # Group by hospitalization and timestamp
        for (hosp_id, ts), group in df.groupby(["hospitalization_id", "recorded_dttm"]):
            gcs_total = group[group["assessment_category"] == "gcs_total"]["assessment_value"]
            gcs_eye = group[group["assessment_category"] == "gcs_eye"]["assessment_value"]
            gcs_verbal = group[group["assessment_category"] == "gcs_verbal"]["assessment_value"]
            gcs_motor = group[group["assessment_category"] == "gcs_motor"]["assessment_value"]

            if len(gcs_total) > 0 and len(gcs_eye) > 0 and len(gcs_verbal) > 0 and len(gcs_motor) > 0:
                expected_total = gcs_eye.iloc[0] + gcs_verbal.iloc[0] + gcs_motor.iloc[0]
                assert gcs_total.iloc[0] == expected_total
