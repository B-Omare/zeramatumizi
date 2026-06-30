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
    Disorder progression is generated from a realistic latent risk
    score combining substance use, age of initiation, HIV status,
    employment, and wealth - mirroring the causal pathway in the
    project brief, so downstream models have genuine signal to learn.
    In production this will be replaced with the real KDHS 2022 data.
    """
    np.random.seed(42)
    n = 4000

    counties = [
        "Kisumu", "Kakamega", "Siaya", "Homa Bay", "Migori",
        "Nairobi", "Mombasa", "Nakuru", "Kisii", "Nyamira"
    ]

    # Generate ages first so initiation age is always lower than current age
    ages = np.random.randint(18, 65, n)

    county_arr = np.random.choice(counties, n)
    gender_arr = np.random.choice(["male", "female"], n)
    education_arr = np.random.choice(
        ["none", "primary", "secondary", "tertiary"], n,
        p=[0.10, 0.40, 0.35, 0.15]
    )
    wealth_arr = np.random.choice(
        ["poorest", "poor", "middle", "rich", "richest"], n
    )
    alcohol_arr = np.random.choice([0, 1], n, p=[0.70, 0.30])
    cannabis_arr = np.random.choice([0, 1], n, p=[0.92, 0.08])
    khat_arr = np.random.choice([0, 1], n, p=[0.88, 0.12])
    initiation_arr = np.array([np.random.randint(10, age) for age in ages])
    hiv_arr = np.random.choice(
        ["positive", "negative", "unknown"], n, p=[0.06, 0.74, 0.20]
    )
    employment_arr = np.random.choice(
        ["employed", "unemployed", "student"], n, p=[0.35, 0.45, 0.20]
    )

    # --- Build a latent risk score from real signal ---
    wealth_penalty = {
        "poorest": 0.5, "poor": 0.3, "middle": 0.0, "rich": -0.3, "richest": -0.5
    }
    wealth_score = np.array([wealth_penalty[w] for w in wealth_arr])

    risk_score = (
        0.9 * alcohol_arr
        + 1.1 * cannabis_arr
        + 0.8 * khat_arr
        + 1.0 * (employment_arr == "unemployed").astype(int)
        + 0.7 * (hiv_arr == "positive").astype(int)
        + 0.6 * (initiation_arr < 15).astype(int)
        + wealth_score
        - 0.4 * (education_arr == "tertiary").astype(int)
        + np.random.normal(0, 0.35, n)  # noise so it's not perfectly separable
    )

    # Convert risk score to probability via logistic function
    disorder_prob = 1 / (1 + np.exp(-(risk_score - 1.5)))
    disorder_progression = np.random.binomial(1, disorder_prob)

    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "county": county_arr,
        "age": ages,
        "gender": gender_arr,
        "education_level": education_arr,
        "wealth_index": wealth_arr,
        "alcohol_use": alcohol_arr,
        "cannabis_use": cannabis_arr,
        "khat_use": khat_arr,
        "age_of_initiation": initiation_arr,
        "hiv_status": hiv_arr,
        "employment_status": employment_arr,
        "disorder_progression": disorder_progression,
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
    print(f"  Disorder progression rate: {df['disorder_progression'].mean():.1%}")
    return df


if __name__ == "__main__":
    save_sample_data()