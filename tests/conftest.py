"""Shared pytest fixtures for synthetic CLIF tests."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.generators.patient import PatientGenerator
from synthetic_clif.generators.hospitalization import HospitalizationGenerator
from synthetic_clif.generators.dataset import SyntheticCLIFDataset


@pytest.fixture
def mcide():
    """Shared mCIDE loader fixture."""
    return MCIDELoader()


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def small_dataset(seed):
    """Small synthetic dataset for testing."""
    dataset = SyntheticCLIFDataset(
        n_patients=5,
        n_hospitalizations=8,
        seed=seed,
        include_concept_tables=True,
    )
    return dataset.generate()


@pytest.fixture
def patients_df(seed, mcide):
    """Generated patient DataFrame for testing."""
    gen = PatientGenerator(seed=seed, mcide=mcide)
    return gen.generate(n_patients=10)


@pytest.fixture
def hospitalizations_df(patients_df, seed, mcide):
    """Generated hospitalization DataFrame for testing."""
    gen = HospitalizationGenerator(seed=seed, mcide=mcide)
    return gen.generate(patients_df, n_hospitalizations=15)


@pytest.fixture
def reference_date():
    """Fixed reference date for testing."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
