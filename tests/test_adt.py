"""Tests for ADT generator."""

import pytest
import pandas as pd

from synthetic_clif.generators.adt import ADTGenerator
from synthetic_clif.config.schema import CLIFSchema


class TestADTGenerator:
    """Tests for ADTGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic ADT generation."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert len(df) > 0
        assert "hospitalization_id" in df.columns
        assert "in_dttm" in df.columns
        assert "out_dttm" in df.columns
        assert "location_category" in df.columns

    def test_hospital_id_column_present(self, hospitalizations_df, seed, mcide):
        """Test that hospital_id column is present per CLIF 2.1.0 spec."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert "hospital_id" in df.columns
        assert df["hospital_id"].notna().all()
        # Verify format
        for hid in df["hospital_id"]:
            assert hid.startswith("HOSP-")

    def test_location_type_column_present(self, hospitalizations_df, seed, mcide):
        """Test that location_type column is present per CLIF 2.1.0 spec."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert "location_type" in df.columns
        # ICU rows (specific ICU categories) should have a location_type; others should be null
        icu_categories = set(ADTGenerator.ICU_CATEGORIES)
        icu_rows = df[df["location_category"].isin(icu_categories)]
        non_icu_rows = df[~df["location_category"].isin(icu_categories)]

        if len(icu_rows) > 0:
            assert icu_rows["location_type"].notna().all()
            valid_types = set(ADTGenerator.LOCATION_TYPES)
            for lt in icu_rows["location_type"]:
                assert lt in valid_types
        if len(non_icu_rows) > 0:
            assert non_icu_rows["location_type"].isna().all()

    def test_schema_includes_hospital_id_and_location_type(self):
        """Test that the ADT schema definition includes hospital_id and location_type."""
        schema = CLIFSchema.ADT
        col_names = schema.column_names()

        assert "hospital_id" in col_names
        assert "location_type" in col_names

    def test_referential_integrity(self, hospitalizations_df, seed, mcide):
        """Test that all hospitalization_ids reference valid hospitalizations."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_hosp_ids = set(hospitalizations_df["hospitalization_id"])
        for hosp_id in df["hospitalization_id"]:
            assert hosp_id in valid_hosp_ids

    def test_timestamps_within_hospitalization(self, hospitalizations_df, seed, mcide):
        """Test that ADT timestamps are within hospitalization bounds."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        hosp_lookup = hospitalizations_df.set_index("hospitalization_id")

        for _, row in df.iterrows():
            hosp_id = row["hospitalization_id"]
            hosp = hosp_lookup.loc[hosp_id]
            assert row["in_dttm"] >= hosp["admission_dttm"]
            if pd.notna(hosp["discharge_dttm"]):
                assert row["out_dttm"] <= hosp["discharge_dttm"]

    def test_contiguous_transfers(self, hospitalizations_df, seed, mcide):
        """Test that ADT events are contiguous within each hospitalization."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        for hosp_id in df["hospitalization_id"].unique():
            hosp_adt = df[df["hospitalization_id"] == hosp_id].sort_values("in_dttm")
            if len(hosp_adt) > 1:
                for i in range(len(hosp_adt) - 1):
                    # Each event's out_dttm should equal the next event's in_dttm
                    assert hosp_adt.iloc[i]["out_dttm"] == hosp_adt.iloc[i + 1]["in_dttm"]

    def test_every_hospitalization_has_adt(self, hospitalizations_df, seed, mcide):
        """Test that every hospitalization with a valid admission has ADT events."""
        gen = ADTGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_hosps = hospitalizations_df[
            hospitalizations_df["admission_dttm"].notna()
        ]["hospitalization_id"]

        adt_hosp_ids = set(df["hospitalization_id"])
        for hosp_id in valid_hosps:
            assert hosp_id in adt_hosp_ids
