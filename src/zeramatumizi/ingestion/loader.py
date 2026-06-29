"""
loader.py
Downloads and saves raw data sources for ZeraMatumizi.
"""

import os
import numpy as np
import requests
import pandas as pd

# Define where raw data will be saved
RAW_DATA_PATH = os.path.join("data", "raw")


def ensure_directories():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    print(f"✓ Raw data directory ready: {RAW_DATA_PATH}")


def download_file(url: str, filename: str) -> str:
    """
    Download a file from a URL and save it to the raw data folder.
    Returns the full path of the saved file.
    """
    filepath = os.path.join(RAW_DATA_PATH, filename)

    if os.path.exists(filepath):
        print(f"✓ Already exists, skipping download: {filename}")
        return filepath

    print(f"Downloading {filename}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"✓ Saved: {filepath}")
    return filepath


def create_sample_kdhs_data() -> pd.DataFrame:
    """
    Creates a sample KDHS-style dataset for development and testing.
    In production this will be replaced with the real KDHS 2022 data.
    """
    np.random.seed(42)
    n = 1000

    counties = [
        "Kisumu", "Kakamega", "Siaya", "Homa Bay", "Migori",
        "Nairobi", "Mombasa", "Nakuru", "Kisii", "Nyamira"
    ]

    # Generate ages first so initiation age is always lower than current age
    ages = np.random.randint(18, 65, n)

    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "county": np.random.choice(counties, n),
        "age": ages,
        "gender": np.random.choice(["male", "female"], n),
        "education_level": np.random.choice(
            ["none", "primary", "secondary", "tertiary"], n,
            p=[0.10, 0.40, 0.35, 0.15]
        ),
        "wealth_index": np.random.choice(
            ["poorest", "poor", "middle", "rich", "richest"], n
        ),
        "alcohol_use": np.random.choice([0, 1], n, p=[0.70, 0.30]),
        "cannabis_use": np.random.choice([0, 1], n, p=[0.92, 0.08]),
        "khat_use": np.random.choice([0, 1], n, p=[0.88, 0.12]),
        "age_of_initiation": [np.random.randint(10, age) for age in ages],
        "hiv_status": np.random.choice(
            ["positive", "negative", "unknown"], n,
            p=[0.06, 0.74, 0.20]
        ),
        "employment_status": np.random.choice(
            ["employed", "unemployed", "student"], n,
            p=[0.35, 0.45, 0.20]
        ),
        "disorder_progression": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })

    return df


def save_sample_data():
    """Save the sample dataset to the raw data folder."""
    ensure_directories()
    df = create_sample_kdhs_data()
    filepath = os.path.join(RAW_DATA_PATH, "kdhs_sample.parquet")
    df.to_parquet(filepath, index=False)
    print(f"✓ Sample KDHS data saved: {filepath}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Counties: {df['county'].unique().tolist()}")
    return df


if __name__ == "__main__":
    save_sample_data()