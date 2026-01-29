"""Tests for microbiology generators."""

import pytest
import pandas as pd

from synthetic_clif.generators.microbiology import (
    MicrobiologyCultureGenerator,
    MicrobiologySusceptibilityGenerator,
)


class TestMicrobiologyCultureGenerator:
    """Tests for MicrobiologyCultureGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic culture generation."""
        gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert "hospitalization_id" in df.columns
        assert "culture_id" in df.columns
        assert "order_dttm" in df.columns
        assert "collect_dttm" in df.columns
        assert "result_dttm" in df.columns
        assert "fluid_category" in df.columns
        assert "organism_id" in df.columns
        assert "organism_category" in df.columns
        assert "organism_group" in df.columns

    def test_timestamp_ordering(self, hospitalizations_df, seed, mcide):
        """Test that culture timestamps are properly ordered."""
        gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            for _, row in df.iterrows():
                if pd.notna(row["order_dttm"]) and pd.notna(row["collect_dttm"]):
                    assert row["collect_dttm"] >= row["order_dttm"]
                if pd.notna(row["collect_dttm"]) and pd.notna(row["result_dttm"]):
                    assert row["result_dttm"] >= row["collect_dttm"]

    def test_fluid_category_valid(self, hospitalizations_df, seed, mcide):
        """Test that fluid categories are valid mCIDE values."""
        gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            valid_fluids = set(mcide.get_category("culture_fluid"))
            for fc in df["fluid_category"].dropna():
                assert fc in valid_fluids

    def test_organism_category_valid(self, hospitalizations_df, seed, mcide):
        """Test that organism categories are valid mCIDE values."""
        gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            valid_organisms = set(mcide.get_category("organism"))
            for oc in df["organism_category"].dropna():
                assert oc in valid_organisms

    def test_positive_cultures_have_organism(self, hospitalizations_df, seed, mcide):
        """Test that positive cultures have organism information."""
        gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        if len(df) > 0:
            positive = df[df["organism_id"].notna()]
            # Positive cultures should have organism category
            assert positive["organism_category"].notna().all()


class TestMicrobiologySusceptibilityGenerator:
    """Tests for MicrobiologySusceptibilityGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic susceptibility generation."""
        culture_gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        cultures_df = culture_gen.generate(hospitalizations_df)

        sus_gen = MicrobiologySusceptibilityGenerator(seed=seed, mcide=mcide)
        df = sus_gen.generate(cultures_df)

        assert "organism_id" in df.columns
        assert "antibiotic_name" in df.columns
        assert "antibiotic_category" in df.columns
        assert "susceptibility_category" in df.columns
        assert "mic_value" in df.columns

    def test_organism_id_valid(self, hospitalizations_df, seed, mcide):
        """Test that organism IDs reference valid cultures."""
        culture_gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        cultures_df = culture_gen.generate(hospitalizations_df)

        sus_gen = MicrobiologySusceptibilityGenerator(seed=seed, mcide=mcide)
        df = sus_gen.generate(cultures_df)

        if len(df) > 0:
            valid_organism_ids = set(cultures_df["organism_id"].dropna())
            for oid in df["organism_id"]:
                assert oid in valid_organism_ids

    def test_susceptibility_valid(self, hospitalizations_df, seed, mcide):
        """Test that susceptibility values are valid mCIDE values."""
        culture_gen = MicrobiologyCultureGenerator(seed=seed, mcide=mcide)
        cultures_df = culture_gen.generate(hospitalizations_df)

        sus_gen = MicrobiologySusceptibilityGenerator(seed=seed, mcide=mcide)
        df = sus_gen.generate(cultures_df)

        if len(df) > 0:
            valid_sus = set(mcide.get_category("susceptibility"))
            for sus in df["susceptibility_category"].dropna():
                assert sus in valid_sus
