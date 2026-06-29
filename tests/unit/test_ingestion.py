"""
test_ingestion.py
Unit tests for the ZeraMatumizi ingestion pipeline.
"""

import pandas as pd
import pytest
import sys
import os

# Make sure Python can find our src folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zeramatumizi.ingestion.loader import create_sample_kdhs_data
from zeramatumizi.ingestion.validator import validate_kdhs, KDHS_SCHEMA


def test_sample_data_shape():
    """Dataset must have 1000 rows and 13 columns."""
    df = create_sample_kdhs_data()
    assert df.shape == (1000, 13)


def test_no_missing_values():
    """No missing values allowed in the dataset."""
    df = create_sample_kdhs_data()
    assert df.isnull().sum().sum() == 0


def test_age_range():
    """All ages must be between 18 and 65."""
    df = create_sample_kdhs_data()
    assert df["age"].between(18, 65).all()


def test_initiation_age_less_than_current_age():
    """Age of initiation must always be less than current age."""
    df = create_sample_kdhs_data()
    assert (df["age_of_initiation"] < df["age"]).all()


def test_valid_gender_values():
    """Gender must only be male or female."""
    df = create_sample_kdhs_data()
    assert set(df["gender"].unique()).issubset({"male", "female"})


def test_valid_substance_flags():
    """Substance use flags must be 0 or 1 only."""
    df = create_sample_kdhs_data()
    for col in ["alcohol_use", "cannabis_use", "khat_use"]:
        assert set(df[col].unique()).issubset({0, 1})


def test_valid_hiv_status():
    """HIV status must be positive, negative, or unknown."""
    df = create_sample_kdhs_data()
    assert set(df["hiv_status"].unique()).issubset(
        {"positive", "negative", "unknown"}
    )


def test_disorder_progression_is_binary():
    """Disorder progression target must be 0 or 1."""
    df = create_sample_kdhs_data()
    assert set(df["disorder_progression"].unique()).issubset({0, 1})


def test_validation_passes():
    """Full Pandera schema validation must pass on sample data."""
    df = create_sample_kdhs_data()
    validated = validate_kdhs(df)
    assert len(validated) == 1000