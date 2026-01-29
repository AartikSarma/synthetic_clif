"""Tests for respiratory support generator."""

import pytest
import pandas as pd

from synthetic_clif.generators.respiratory import RespiratoryGenerator


class TestRespiratoryGenerator:
    """Tests for RespiratoryGenerator."""

    def test_generate_basic(self, hospitalizations_df, seed, mcide):
        """Test basic respiratory support generation."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert len(df) > 0
        assert "hospitalization_id" in df.columns
        assert "recorded_dttm" in df.columns
        assert "device_category" in df.columns
        assert "mode_category" in df.columns
        assert "fio2_set" in df.columns
        assert "peep_set" in df.columns
        assert "tracheostomy" in df.columns

    def test_device_categories_valid(self, hospitalizations_df, seed, mcide):
        """Test that device categories are valid mCIDE values."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_devices = set(mcide.get_category("respiratory_device"))
        for dc in df["device_category"].dropna():
            assert dc in valid_devices

    def test_mode_categories_valid(self, hospitalizations_df, seed, mcide):
        """Test that mode categories are valid mCIDE values."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        valid_modes = set(mcide.get_category("respiratory_mode"))
        for mc in df["mode_category"].dropna():
            assert mc in valid_modes

    def test_fio2_range(self, hospitalizations_df, seed, mcide):
        """Test that FiO2 is in valid range."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        fio2_values = df["fio2_set"].dropna()
        if len(fio2_values) > 0:
            assert fio2_values.min() >= 0.21
            assert fio2_values.max() <= 1.0

    def test_peep_range(self, hospitalizations_df, seed, mcide):
        """Test that PEEP is in valid range."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        peep_values = df["peep_set"].dropna()
        if len(peep_values) > 0:
            assert peep_values.min() >= 0
            assert peep_values.max() <= 25

    def test_imv_has_ventilator_settings(self, hospitalizations_df, seed, mcide):
        """Test that IMV records have ventilator settings."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        imv_records = df[df["device_category"] == "IMV"]
        if len(imv_records) > 0:
            # Should have tidal volume and respiratory rate
            assert imv_records["tidal_volume_set"].notna().any()
            assert imv_records["resp_rate_set"].notna().any()

    def test_tracheostomy_flag(self, hospitalizations_df, seed, mcide):
        """Test that tracheostomy flag is boolean."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        assert df["tracheostomy"].dtype == bool

    def test_device_appropriate_settings(self, hospitalizations_df, seed, mcide):
        """Test that settings are appropriate for device type."""
        gen = RespiratoryGenerator(seed=seed, mcide=mcide)
        df = gen.generate(hospitalizations_df)

        # Room air should have FiO2 of 0.21
        room_air = df[df["device_category"] == "Room Air"]
        if len(room_air) > 0:
            assert (room_air["fio2_set"].dropna() == 0.21).all()

        # High flow NC should have flow rate
        hfnc = df[df["device_category"] == "High Flow NC"]
        if len(hfnc) > 0:
            assert hfnc["flow_rate_set"].notna().any()
