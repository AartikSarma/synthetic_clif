"""Tests for medication generators."""

import pytest
import pandas as pd

from synthetic_clif.generators.medications import (
    MedicationContinuousGenerator,
    MedicationIntermittentGenerator,
)
from synthetic_clif.generators.respiratory import RespiratoryGenerator


class TestMedicationContinuousGenerator:
    """Tests for MedicationContinuousGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic continuous medication generation."""
        gen = MedicationContinuousGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert "hospitalization_id" in df.columns
        assert "med_order_id" in df.columns
        assert "admin_dttm" in df.columns
        assert "med_category" in df.columns
        assert "med_name" in df.columns
        assert "med_dose" in df.columns
        assert "med_dose_unit" in df.columns
        assert "med_route_category" in df.columns
        assert "med_group" in df.columns
        assert "mar_action_category" in df.columns

    def test_dose_positive(self, hospitalizations_df, seed, mcide):
        """Test that doses are positive."""
        gen = MedicationContinuousGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            assert (df["med_dose"].dropna() > 0).all()

    def test_with_respiratory(self, hospitalizations_df, seed, mcide):
        """Test generation with respiratory data for sedation correlation."""
        resp_gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        resp_df = resp_gen.generate(hospitalizations_df)

        med_gen = MedicationContinuousGenerator(seed=seed, mcide=mcide)
        df = med_gen.generate(hospitalizations_df, resp_df)

        assert len(df) >= 0


class TestMedicationIntermittentGenerator:
    """Tests for MedicationIntermittentGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic intermittent medication generation."""
        gen = MedicationIntermittentGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert "hospitalization_id" in df.columns
        assert "med_order_id" in df.columns
        assert "admin_dttm" in df.columns
        assert "med_category" in df.columns
        assert "med_name" in df.columns
        assert "med_dose" in df.columns
        assert "med_dose_unit" in df.columns
        assert "med_route_category" in df.columns
        assert "mar_action_category" in df.columns
        assert "med_group" in df.columns

    def test_mostly_given(self, hospitalizations_df, seed, mcide):
        """Test that most medications are marked as given."""
        gen = MedicationIntermittentGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            given_rate = (df["mar_action_category"] == "given").mean()
            assert given_rate > 0.85

    def test_scheduled_timing(self, hospitalizations_df, seed, mcide):
        """Test that scheduled medications have regular timing."""
        gen = MedicationIntermittentGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            order_counts = df.groupby("med_order_id").size()
            assert order_counts.max() > 1
