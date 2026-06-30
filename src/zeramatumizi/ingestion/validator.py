"""
validator.py
Validates raw KDHS and NACADA data using Pandera schemas.
Ensures data quality before it enters the pipeline.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


# Define the schema — rules every row must follow
KDHS_SCHEMA = DataFrameSchema(
    columns={
        "respondent_id": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            description="Unique respondent identifier"
        ),
        "county": Column(
            str,
            checks=Check.isin([
                "Kisumu", "Kakamega", "Siaya", "Homa Bay", "Migori",
                "Nairobi", "Mombasa", "Nakuru", "Kisii", "Nyamira",
                "Kiambu", "Machakos", "Kilifi", "Kwale", "Meru",
                "Embu", "Nyeri", "Kirinyaga", "Muranga", "Nyandarua",
                "Laikipia", "Samburu", "Trans Nzoia", "Uasin Gishu",
                "Elgeyo Marakwet", "Nandi", "Baringo", "Kericho",
                "Bomet", "Narok", "Kajiado", "Makueni", "Kitui",
                "Tharaka Nithi", "Isiolo", "Marsabit", "Mandera",
                "Wajir", "Garissa", "Tana River", "Lamu", "Taita Taveta",
                "Bungoma", "Busia", "Vihiga", "Turkana", "West Pokot"
            ]),
            nullable=False,
            description="Kenya county name"
        ),
        "age": Column(
            int,
            checks=Check.in_range(15, 65),
            nullable=False,
            coerce=True,
            description="Respondent age (15-65 per KDHS eligibility)"
        ),
        "gender": Column(
            str,
            checks=Check.isin(["male", "female"]),
            nullable=False,
            description="Respondent gender"
        ),
        "education_level": Column(
            str,
            checks=Check.isin(["none", "primary", "secondary", "tertiary"]),
            nullable=False,
            description="Highest education level attained"
        ),
        "wealth_index": Column(
            str,
            checks=Check.isin(["poorest", "poor", "middle", "rich", "richest"]),
            nullable=False,
            description="DHS wealth index quintile"
        ),
        "alcohol_use": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Alcohol use flag (0=No, 1=Yes)"
        ),
        "cannabis_use": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Cannabis use flag (0=No, 1=Yes)"
        ),
        "khat_use": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Khat use flag (0=No, 1=Yes)"
        ),
        "age_of_initiation": Column(
            int,
            checks=Check.in_range(5, 65),
            nullable=False,
            coerce=True,
            description="Age at first substance use"
        ),
        "hiv_status": Column(
            str,
            checks=Check.isin(["positive", "negative", "unknown"]),
            nullable=False,
            description="HIV status"
        ),
        "employment_status": Column(
            str,
            checks=Check.isin(["employed", "unemployed", "student"]),
            nullable=False,
            description="Current employment status"
        ),
        "disorder_progression": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            coerce=True,
            description="Target variable: disorder progression (0=No, 1=Yes)"
        ),
    },
    checks=[
        # Global check: age of initiation must be less than current age
        Check(
            lambda df: (df["age_of_initiation"] < df["age"]).all(),
            error="age_of_initiation must be less than current age"
        )
    ],
    name="KDHS_Schema",
    description="Validation schema for KDHS 2022 substance use dataset"
)


def validate_kdhs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a KDHS dataframe against the schema.
    Returns the validated dataframe if all checks pass.
    Raises SchemaError with details if any check fails.
    """
    print("Running validation on KDHS dataset...")
    print(f"  Rows to validate: {len(df)}")

    try:
        validated_df = KDHS_SCHEMA.validate(df, lazy=True)
        print(f"✓ All validation checks passed!")
        print(f"  Columns validated: {len(validated_df.columns)}")
        print(f"  Rows validated: {len(validated_df)}")
        return validated_df

    except pa.errors.SchemaErrors as e:
        print(f"✗ Validation failed! Issues found:")
        print(e.failure_cases)
        raise


def run_validation_report(df: pd.DataFrame):
    """Prints a summary report of data quality."""
    print("\n--- Data Quality Report ---")
    print(f"Total rows:        {len(df)}")
    print(f"Total columns:     {len(df.columns)}")
    print(f"Missing values:    {df.isnull().sum().sum()}")
    print(f"Duplicate IDs:     {df['respondent_id'].duplicated().sum()}")
    print(f"\nSubstance use rates:")
    print(f"  Alcohol:  {df['alcohol_use'].mean():.1%}")
    print(f"  Cannabis: {df['cannabis_use'].mean():.1%}")
    print(f"  Khat:     {df['khat_use'].mean():.1%}")
    print(f"\nDisorder progression rate: {df['disorder_progression'].mean():.1%}")
    print(f"\nCounty distribution:")
    print(df['county'].value_counts().to_string())
    print("---------------------------\n")


if __name__ == "__main__":
    # Load the sample data and validate it
    df = pd.read_parquet("data/raw/kdhs_sample.parquet")
    validated_df = validate_kdhs(df)
    run_validation_report(validated_df)