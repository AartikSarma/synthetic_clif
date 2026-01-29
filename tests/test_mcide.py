"""Tests for mCIDE loader."""

import pytest
from synthetic_clif.config.mcide import MCIDELoader


class TestMCIDELoader:
    """Tests for MCIDELoader class."""

    def test_init_default(self):
        """Test default initialization."""
        loader = MCIDELoader()
        assert loader.mcide_dir is None

    def test_get_category_vital(self, mcide):
        """Test getting vital categories."""
        vitals = mcide.get_category("vital")
        assert len(vitals) > 0
        assert "heart_rate" in vitals
        assert "sbp" in vitals
        assert "spo2" in vitals

    def test_get_category_sex(self, mcide):
        """Test getting sex categories."""
        sexes = mcide.get_category("sex")
        assert "Female" in sexes
        assert "Male" in sexes

    def test_get_category_discharge(self, mcide):
        """Test getting discharge categories."""
        discharges = mcide.get_category("discharge")
        assert "Home" in discharges
        assert "Expired" in discharges
        assert "SNF" in discharges

    def test_get_category_location(self, mcide):
        """Test getting location categories."""
        locations = mcide.get_category("location")
        assert "icu" in locations
        assert "ward" in locations
        assert "ed" in locations

    def test_get_category_respiratory_device(self, mcide):
        """Test getting respiratory device categories."""
        devices = mcide.get_category("respiratory_device")
        assert "IMV" in devices
        assert "NIPPV" in devices
        assert "Room Air" in devices

    def test_get_category_unknown(self, mcide):
        """Test getting unknown category returns empty list."""
        result = mcide.get_category("nonexistent_category")
        assert result == []

    def test_vital_categories_property(self, mcide):
        """Test vital_categories cached property."""
        vitals = mcide.vital_categories
        assert len(vitals) > 0
        # Should be same object on second access (cached)
        assert mcide.vital_categories is vitals

    def test_lab_categories_property(self, mcide):
        """Test lab_categories cached property."""
        labs = mcide.lab_categories
        assert len(labs) > 0
        assert "sodium" in labs
        assert "creatinine" in labs

    def test_get_lab_reference_units(self, mcide):
        """Test getting lab reference units."""
        units = mcide.get_lab_reference_units()
        assert units["sodium"] == "mEq/L"
        assert units["creatinine"] == "mg/dL"
        assert units["hemoglobin"] == "g/dL"

    def test_get_lab_reference_ranges(self, mcide):
        """Test getting lab reference ranges."""
        ranges = mcide.get_lab_reference_ranges()
        assert ranges["sodium"] == (136, 145)
        assert ranges["potassium"] == (3.5, 5.0)
        assert ranges["ph"] == (7.35, 7.45)
