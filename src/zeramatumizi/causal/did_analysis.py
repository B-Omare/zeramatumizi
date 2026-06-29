"""
did_analysis.py
Difference-in-Differences analysis for ZeraMatumizi.

Natural experiment: NACADA county enforcement campaigns 2019-2022.
Question: Do NACADA campaigns causally reduce disorder progression,
or do they merely displace use geographically?

Treatment group: Counties receiving intensive campaigns
(Kisumu, Kakamega, Siaya, Homa Bay, Migori)
Control group: Matched comparable counties without campaigns
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Output folder
REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(REPORTS_PATH, exist_ok=True)


def create_sample_panel_data() -> pd.DataFrame:
    """
    Creates a simulated panel dataset representing monthly
    substance use disorder registrations across Kenyan counties
    from 2019 to 2022.

    In production this will use real DHIS2 and NACADA data.

    Panel structure:
    - Unit: county
    - Time: month (2019-01 to 2022-12)
    - Treatment: NACADA intensive campaign (1=yes, 0=no)
    - Outcome: monthly SUD treatment registrations
    """
    np.random.seed(42)

    # Treatment counties (received NACADA campaigns)
    treated_counties = [
        "Kisumu", "Kakamega", "Siaya", "Homa Bay", "Migori"
    ]

    # Control counties (no intensive campaigns)
    control_counties = [
        "Nakuru", "Kisii", "Nyamira", "Vihiga", "Bungoma"
    ]

    all_counties = treated_counties + control_counties

    # Generate monthly time periods 2019-2022
    periods = pd.date_range(start="2019-01", end="2022-12", freq="MS")

    rows = []
    for county in all_counties:
        is_treated = county in treated_counties

        # Campaign started mid-2021 in treated counties
        for period in periods:
            post_treatment = (period >= pd.Timestamp("2021-07-01"))
            treatment_active = is_treated and post_treatment

            # Baseline SUD registrations (higher in treated counties
            # due to higher baseline prevalence)
            baseline = 45 if is_treated else 30

            # Pre-treatment trend (slight increase over time)
            time_index = (period.year - 2019) * 12 + period.month
            trend = time_index * 0.3

            # Treatment effect: campaigns reduce registrations by ~25%
            treatment_effect = -12 if treatment_active else 0

            # Random noise
            noise = np.random.normal(0, 4)

            # Final outcome
            sud_registrations = max(0, baseline + trend + treatment_effect + noise)

            rows.append({
                "county": county,
                "period": period,
                "year": period.year,
                "month": period.month,
                "is_treated": int(is_treated),
                "post_treatment": int(post_treatment),
                "treatment_active": int(treatment_active),
                "sud_registrations": round(sud_registrations, 1),
                "time_index": time_index,
            })

    df = pd.DataFrame(rows)
    return df


def run_did_estimation(df: pd.DataFrame) -> dict:
    """
    Runs the Difference-in-Differences estimation.

    DiD Formula:
    ATT = (Treated_Post - Treated_Pre) - (Control_Post - Control_Pre)

    Where ATT = Average Treatment Effect on the Treated
    """
    print("Running Difference-in-Differences estimation...")

    # Calculate group means
    treated_pre = df[
        (df["is_treated"] == 1) & (df["post_treatment"] == 0)
    ]["sud_registrations"].mean()

    treated_post = df[
        (df["is_treated"] == 1) & (df["post_treatment"] == 1)
    ]["sud_registrations"].mean()

    control_pre = df[
        (df["is_treated"] == 0) & (df["post_treatment"] == 0)
    ]["sud_registrations"].mean()

    control_post = df[
        (df["is_treated"] == 0) & (df["post_treatment"] == 1)
    ]["sud_registrations"].mean()

    # DiD estimate
    did_estimate = (treated_post - treated_pre) - (control_post - control_pre)

    # Counterfactual: what would treated counties look like without campaign?
    counterfactual = treated_pre + (control_post - control_pre)

    results = {
        "treated_pre":      round(treated_pre, 2),
        "treated_post":     round(treated_post, 2),
        "control_pre":      round(control_pre, 2),
        "control_post":     round(control_post, 2),
        "did_estimate":     round(did_estimate, 2),
        "counterfactual":   round(counterfactual, 2),
        "pct_reduction":    round((did_estimate / treated_pre) * 100, 1),
    }

    return results


def print_did_results(results: dict):
    """Prints a formatted DiD results table."""
    print("\n--- Difference-in-Differences Results ---")
    print(f"  Treated counties (pre-campaign):   {results['treated_pre']} registrations/month")
    print(f"  Treated counties (post-campaign):  {results['treated_post']} registrations/month")
    print(f"  Control counties (pre-campaign):   {results['control_pre']} registrations/month")
    print(f"  Control counties (post-campaign):  {results['control_post']} registrations/month")
    print(f"\n  Counterfactual (no campaign):      {results['counterfactual']} registrations/month")
    print(f"\n  DiD Estimate (ATT):                {results['did_estimate']} registrations/month")
    print(f"  Reduction from campaign:           {results['pct_reduction']}%")
    print(f"\n  Interpretation:")
    if results['did_estimate'] < 0:
        print(f"  ✓ NACADA campaigns causally REDUCED SUD registrations")
        print(f"    by {abs(results['pct_reduction'])}% in treated counties.")
    else:
        print(f"  ✗ No significant causal reduction detected.")
    print("-----------------------------------------\n")


def check_parallel_trends(df: pd.DataFrame):
    """
    Checks the parallel trends assumption — the key assumption
    of DiD analysis. Before the campaign, treated and control
    counties should have similar trends.
    """
    print("Checking parallel trends assumption...")

    pre_data = df[df["post_treatment"] == 0].copy()

    treated_trend = pre_data[pre_data["is_treated"] == 1].groupby(
        "time_index"
    )["sud_registrations"].mean()

    control_trend = pre_data[pre_data["is_treated"] == 0].groupby(
        "time_index"
    )["sud_registrations"].mean()

    # Calculate correlation of pre-treatment trends
    correlation = treated_trend.corr(control_trend)

    print(f"  Pre-treatment trend correlation: {correlation:.3f}")
    if correlation > 0.90:
        print(f"  ✓ Parallel trends assumption likely satisfied (correlation > 0.90)")
    else:
        print(f"  ⚠ Parallel trends assumption may be violated (correlation < 0.90)")

    return correlation


def save_did_plot(df: pd.DataFrame, results: dict):
    """
    Saves a DiD visualisation showing treated vs control trends
    with the counterfactual line.
    """
    ensure_directories()

    # Monthly averages by group
    monthly = df.groupby(
        ["period", "is_treated"]
    )["sud_registrations"].mean().reset_index()

    treated = monthly[monthly["is_treated"] == 1]
    control = monthly[monthly["is_treated"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "ZeraMatumizi — Difference-in-Differences Analysis\n"
        "Effect of NACADA Campaigns on SUD Registrations",
        fontsize=13, fontweight="bold"
    )

    # --- Plot 1: Trends over time ---
    ax1 = axes[0]
    ax1.plot(
        treated["period"], treated["sud_registrations"],
        color="#E74C3C", linewidth=2, marker="o", markersize=3,
        label="Treated counties (Kisumu, Kakamega, Siaya, Homa Bay, Migori)"
    )
    ax1.plot(
        control["period"], control["sud_registrations"],
        color="#3498DB", linewidth=2, marker="s", markersize=3,
        label="Control counties"
    )
    ax1.axvline(
        x=pd.Timestamp("2021-07-01"),
        color="green", linestyle="--", linewidth=2,
        label="Campaign start (July 2021)"
    )
    ax1.set_title("SUD Registrations Over Time", fontweight="bold")
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Monthly SUD Registrations")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: DiD Summary bar chart ---
    ax2 = axes[1]
    categories = ["Treated\n(Pre)", "Treated\n(Post)",
                  "Control\n(Pre)", "Control\n(Post)", "Counterfactual"]
    values = [
        results["treated_pre"],
        results["treated_post"],
        results["control_pre"],
        results["control_post"],
        results["counterfactual"],
    ]
    colours = ["#E74C3C", "#C0392B", "#3498DB", "#2980B9", "#95A5A6"]
    bars = ax2.bar(categories, values, color=colours, edgecolor="white", linewidth=0.5)
    ax2.set_title("DiD Summary", fontweight="bold")
    ax2.set_ylabel("Monthly SUD Registrations")

    # Annotate bars
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Annotate DiD estimate
    ax2.annotate(
        f"DiD Estimate: {results['did_estimate']} ({results['pct_reduction']}%)",
        xy=(0.5, 0.05), xycoords="axes fraction",
        ha="center", fontsize=10, color="#27AE60", fontweight="bold"
    )

    plt.tight_layout()
    output_path = os.path.join(REPORTS_PATH, "did_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ DiD plot saved: {output_path}")


if __name__ == "__main__":
    print("Running ZeraMatumizi Difference-in-Differences Analysis...")
    ensure_directories()

    # Generate panel data
    df = create_sample_panel_data()
    print(f"✓ Panel data created: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Counties: {df['county'].nunique()}")
    print(f"  Time periods: {df['period'].nunique()} months")

    # Check parallel trends
    check_parallel_trends(df)

    # Run DiD estimation
    results = run_did_estimation(df)

    # Print results
    print_did_results(results)

    # Save plot
    save_did_plot(df, results)

    print("✓ DiD analysis complete!")