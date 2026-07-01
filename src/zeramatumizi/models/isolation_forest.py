"""
isolation_forest.py
Isolation Forest anomaly detection for ZeraMatumizi.

Detects unusual high-risk individuals who don't fit the typical
disorder progression profile - these are often the most vulnerable
cases that standard supervised models miss because they fall outside
the training distribution.

Use cases:
1. Detecting individuals with atypical risk factor combinations
   (e.g. wealthy but polysubstance user with early initiation)
2. Flagging unusual county-level patterns in DHIS2 data
   (e.g. sudden spike in registrations suggesting outbreak)
3. Data quality checks - detecting likely data entry errors

Algorithm: Isolation Forest (Liu et al. 2008)
- Anomalies are isolated quickly in random trees
- Normal points require more splits to isolate
- No labels needed - fully unsupervised
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_and_engineer_features() -> pd.DataFrame:
    """Loads and engineers features - same as other D5 models."""
    df = pd.read_parquet("data/raw/kdhs_sample.parquet")

    df["gender_num"] = (df["gender"] == "male").astype(int)
    df["employed_num"] = (df["employment_status"] == "employed").astype(int)
    df["unemployed_num"] = (df["employment_status"] == "unemployed").astype(int)
    df["hiv_positive_num"] = (df["hiv_status"] == "positive").astype(int)

    education_map = {"none": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    df["education_num"] = df["education_level"].map(education_map)
    wealth_map = {"poorest": 0, "poor": 1, "middle": 2, "rich": 3, "richest": 4}
    df["wealth_num"] = df["wealth_index"].map(wealth_map)

    df["any_substance"] = (
        (df["alcohol_use"] + df["cannabis_use"] + df["khat_use"]) > 0
    ).astype(int)
    df["polysubstance"] = (
        (df["alcohol_use"] + df["cannabis_use"] + df["khat_use"]) >= 2
    ).astype(int)
    df["early_initiation"] = (df["age_of_initiation"] < 15).astype(int)
    df["years_since_initiation"] = df["age"] - df["age_of_initiation"]

    return df


def get_feature_columns():
    return [
        "age", "gender_num", "education_num", "wealth_num",
        "alcohol_use", "cannabis_use", "khat_use",
        "any_substance", "polysubstance",
        "age_of_initiation", "early_initiation",
        "years_since_initiation", "hiv_positive_num",
        "employed_num", "unemployed_num",
    ]


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05):
    """
    Fits Isolation Forest and assigns anomaly scores to all individuals.

    contamination: expected proportion of anomalies in the dataset
    (0.05 = 5% of individuals are expected to be anomalous)

    Returns the dataframe with anomaly scores and flags added.
    """
    print(f"Fitting Isolation Forest (contamination={contamination})...")

    feature_cols = get_feature_columns()
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)

    # Anomaly scores: lower = more anomalous
    # decision_function returns negative scores for anomalies
    anomaly_scores = iso_forest.decision_function(X_scaled)
    anomaly_labels = iso_forest.predict(X_scaled)  # -1 = anomaly, 1 = normal

    df["anomaly_score"] = anomaly_scores
    df["is_anomaly"] = (anomaly_labels == -1).astype(int)

    n_anomalies = df["is_anomaly"].sum()
    print(f"Anomaly detection complete!")
    print(f"  Total individuals: {len(df)}")
    print(f"  Flagged as anomalous: {n_anomalies} ({n_anomalies/len(df):.1%})")

    return df, iso_forest, scaler, X_scaled


def analyse_anomalies(df: pd.DataFrame):
    """
    Analyses the characteristics of flagged anomalies vs normal individuals
    to understand what makes them unusual.
    """
    print("\n--- Anomaly Profile Analysis ---")

    anomalies = df[df["is_anomaly"] == 1]
    normals = df[df["is_anomaly"] == 0]

    feature_cols = get_feature_columns()

    print(f"\n  {'Feature':<25} {'Normal Mean':>12} {'Anomaly Mean':>13} {'Difference':>12}")
    print(f"  {'-'*65}")

    differences = []
    for col in feature_cols:
        normal_mean = normals[col].mean()
        anomaly_mean = anomalies[col].mean()
        diff = anomaly_mean - normal_mean
        differences.append((col, normal_mean, anomaly_mean, diff))

    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x[3]), reverse=True)

    for col, normal_mean, anomaly_mean, diff in differences:
        marker = "***" if abs(diff) > 0.3 else "  "
        print(f"  {col:<25} {normal_mean:>12.3f} {anomaly_mean:>13.3f} "
              f"{diff:>+12.3f} {marker}")

    print(f"\n  Disorder progression rate:")
    print(f"    Normal individuals:    {normals['disorder_progression'].mean():.1%}")
    print(f"    Anomalous individuals: {anomalies['disorder_progression'].mean():.1%}")

    print(f"\n  County distribution of anomalies:")
    county_anomaly = df.groupby("county")["is_anomaly"].agg(["sum", "mean"]).round(3)
    county_anomaly.columns = ["n_anomalies", "anomaly_rate"]
    county_anomaly = county_anomaly.sort_values("anomaly_rate", ascending=False)
    print(county_anomaly.to_string())
    print("--------------------------------\n")

    return anomalies


def save_anomaly_plots(df: pd.DataFrame, X_scaled: np.ndarray):
    """
    Saves three plots:
    1. PCA scatter of anomalies vs normals
    2. Anomaly score distribution
    3. County-level anomaly rates
    """
    ensure_directories()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "ZeraMatumizi - Isolation Forest Anomaly Detection",
        fontsize=13, fontweight="bold"
    )

    # --- Plot 1: PCA scatter ---
    ax1 = axes[0]
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    normal_mask = df["is_anomaly"] == 0
    anomaly_mask = df["is_anomaly"] == 1

    ax1.scatter(
        X_pca[normal_mask, 0], X_pca[normal_mask, 1],
        c="#3498DB", alpha=0.4, s=8, label="Normal"
    )
    ax1.scatter(
        X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
        c="#E74C3C", alpha=0.8, s=25, label="Anomaly", zorder=5
    )
    ax1.set_title("PCA Projection\n(Anomalies in Red)", fontweight="bold")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Anomaly score distribution ---
    ax2 = axes[1]
    ax2.hist(
        df[df["is_anomaly"] == 0]["anomaly_score"],
        bins=40, color="#3498DB", alpha=0.7, label="Normal", density=True
    )
    ax2.hist(
        df[df["is_anomaly"] == 1]["anomaly_score"],
        bins=20, color="#E74C3C", alpha=0.8, label="Anomaly", density=True
    )
    threshold = df[df["is_anomaly"] == 1]["anomaly_score"].max()
    ax2.axvline(x=threshold, color="black", linestyle="--",
                linewidth=1.5, label=f"Threshold: {threshold:.3f}")
    ax2.set_title("Anomaly Score Distribution", fontweight="bold")
    ax2.set_xlabel("Anomaly Score (lower = more anomalous)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: County-level anomaly rates ---
    ax3 = axes[2]
    county_rates = df.groupby("county")["is_anomaly"].mean().sort_values(ascending=True)
    colours = ["#E74C3C" if r > county_rates.mean() else "#3498DB"
               for r in county_rates.values]
    ax3.barh(county_rates.index, county_rates.values, color=colours, edgecolor="white")
    ax3.axvline(x=county_rates.mean(), color="black", linestyle="--",
                linewidth=1.5, label=f"Mean: {county_rates.mean():.1%}")
    ax3.set_title("Anomaly Rate by County\n(Red = above average)", fontweight="bold")
    ax3.set_xlabel("Proportion Flagged as Anomalous")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    output_path = os.path.join(REPORTS_PATH, "isolation_forest.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Anomaly detection plots saved: {output_path}")


def flag_priority_cases(df: pd.DataFrame, n: int = 10):
    """
    Identifies and prints the most anomalous individuals -
    those who most urgently need manual review by a NACADA officer.
    """
    print(f"\n--- Top {n} Most Anomalous Individuals (Priority Review List) ---")

    priority = df.nsmallest(n, "anomaly_score")[[
        "respondent_id", "county", "age", "gender",
        "disorder_progression", "any_substance", "polysubstance",
        "wealth_index", "employment_status", "anomaly_score"
    ]]

    print(priority.to_string(index=False))
    print("----------------------------------------------------------------\n")

    return priority


def run_county_level_anomaly_detection(df: pd.DataFrame):
    """
    Runs county-level anomaly detection on aggregated monthly indicators
    to flag counties with unusual patterns - mimicking DHIS2 surveillance.
    """
    print("\n--- County-Level Anomaly Detection (DHIS2-style) ---")

    county_stats = df.groupby("county").agg(
        n=("respondent_id", "count"),
        disorder_rate=("disorder_progression", "mean"),
        alcohol_rate=("alcohol_use", "mean"),
        cannabis_rate=("cannabis_use", "mean"),
        khat_rate=("khat_use", "mean"),
        polysubstance_rate=("polysubstance", "mean"),
        unemployment_rate=("employment_status", lambda x: (x == "unemployed").mean()),
        hiv_rate=("hiv_positive_num", "mean"),
    ).round(3)

    scaler = StandardScaler()
    X_county = scaler.fit_transform(county_stats)

    iso = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    iso.fit(X_county)
    county_scores = iso.decision_function(X_county)
    county_labels = iso.predict(X_county)

    county_stats["anomaly_score"] = county_scores
    county_stats["is_anomaly"] = (county_labels == -1).astype(int)
    county_stats = county_stats.sort_values("anomaly_score")

    print(f"\n  County anomaly rankings (most unusual first):")
    print(f"  {'County':<15} {'Disorder Rate':>14} {'Unemployment':>13} "
          f"{'Anomaly Score':>14} {'Flag':>6}")
    print(f"  {'-'*65}")

    for county, row in county_stats.iterrows():
        flag = "ALERT" if row["is_anomaly"] == 1 else "OK"
        print(f"  {county:<15} {row['disorder_rate']:>14.1%} "
              f"{row['unemployment_rate']:>13.1%} "
              f"{row['anomaly_score']:>14.3f} {flag:>6}")

    print("------------------------------------------------------\n")
    return county_stats


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - Isolation Forest Anomaly Detection")
    print("=" * 60)

    ensure_directories()

    df = load_and_engineer_features()
    print(f"\nLoaded {len(df)} individuals")

    # Individual-level anomaly detection
    df, iso_forest, scaler, X_scaled = run_isolation_forest(df, contamination=0.05)
    anomalies = analyse_anomalies(df)
    save_anomaly_plots(df, X_scaled)
    priority_cases = flag_priority_cases(df, n=10)

    # County-level anomaly detection
    county_anomalies = run_county_level_anomaly_detection(df)

    print("Isolation Forest anomaly detection complete!")