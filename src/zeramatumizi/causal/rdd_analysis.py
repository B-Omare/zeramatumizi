"""
rdd_analysis.py
Regression Discontinuity Design analysis for ZeraMatumizi.

Running variable: exact age of respondent
Cutoff: 18 years (Kenya's legal drinking age)
Question: Does legal access to alcohol at age 18 causally
increase alcohol use disorder — and by how much?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Output folder
REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(REPORTS_PATH, exist_ok=True)


def create_rdd_sample_data() -> pd.DataFrame:
    """
    Creates a simulated dataset for RDD analysis.
    Each row is one individual with:
    - age_months: exact age in months (running variable)
    - age_years: age in years
    - above_cutoff: 1 if aged 18+, 0 if under 18
    - alcohol_use_score: AUDIT-C style score (0-12)
    - disorder_flag: 1 if AUDIT-C >= 4 (disorder threshold)

    A sharp discontinuity is introduced at age 18 to simulate
    the causal effect of legal alcohol access.
    """
    np.random.seed(42)
    n = 2000

    # Age in months centred around 18 years (216 months)
    # Range: 15 to 21 years old
    age_months = np.random.uniform(180, 252, n)  # 15-21 years in months
    age_years = age_months / 12
    above_cutoff = (age_years >= 18).astype(int)

    # Centred running variable (0 = exactly age 18)
    age_centred = age_years - 18

    # Baseline alcohol use score increases smoothly with age
    baseline_score = 2.0 + (age_centred * 0.4)

    # Sharp jump at age 18 due to legal access (treatment effect)
    treatment_effect = 1.8 * above_cutoff

    # Random noise
    noise = np.random.normal(0, 1.2, n)

    # Final AUDIT-C style score (clipped to 0-12)
    alcohol_use_score = np.clip(baseline_score + treatment_effect + noise, 0, 12)

    # Disorder flag: AUDIT-C >= 4
    disorder_flag = (alcohol_use_score >= 4).astype(int)

    df = pd.DataFrame({
        "age_months":        age_months,
        "age_years":         age_years,
        "age_centred":       age_centred,
        "above_cutoff":      above_cutoff,
        "alcohol_use_score": alcohol_use_score,
        "disorder_flag":     disorder_flag,
    })

    return df


def estimate_rdd_effect(df: pd.DataFrame, bandwidth: float = 2.0) -> dict:
    """
    Estimates the RDD treatment effect at the age-18 cutoff.

    Uses a local linear regression on both sides of the cutoff
    within the specified bandwidth (default: ±2 years).

    Returns a dictionary of results.
    """
    print(f"Running RDD estimation with bandwidth ±{bandwidth} years...")

    # Restrict to bandwidth
    within_bw = df[df["age_centred"].abs() <= bandwidth].copy()

    # Split into below and above cutoff
    below = within_bw[within_bw["above_cutoff"] == 0]
    above = within_bw[within_bw["above_cutoff"] == 1]

    # Fit local linear regression on each side
    slope_below, intercept_below, _, _, _ = stats.linregress(
        below["age_centred"], below["alcohol_use_score"]
    )
    slope_above, intercept_above, _, _, _ = stats.linregress(
        above["age_centred"], above["alcohol_use_score"]
    )

    # Predicted values at cutoff (age_centred = 0)
    y_hat_below = intercept_below  # at age_centred = 0
    y_hat_above = intercept_above  # at age_centred = 0

    # RDD estimate = jump at cutoff
    rdd_estimate = y_hat_above - y_hat_below

    # Disorder rate on each side
    disorder_rate_below = below["disorder_flag"].mean()
    disorder_rate_above = above["disorder_flag"].mean()
    disorder_jump = disorder_rate_above - disorder_rate_below

    # Simple t-test for significance
    t_stat, p_value = stats.ttest_ind(
        above["alcohol_use_score"],
        below["alcohol_use_score"]
    )

    results = {
        "bandwidth":            bandwidth,
        "n_below":              len(below),
        "n_above":              len(above),
        "y_hat_below":          round(y_hat_below, 3),
        "y_hat_above":          round(y_hat_above, 3),
        "rdd_estimate":         round(rdd_estimate, 3),
        "disorder_rate_below":  round(disorder_rate_below, 3),
        "disorder_rate_above":  round(disorder_rate_above, 3),
        "disorder_jump":        round(disorder_jump, 3),
        "t_statistic":          round(t_stat, 3),
        "p_value":              round(p_value, 4),
        "slope_below":          round(slope_below, 3),
        "slope_above":          round(slope_above, 3),
        "intercept_below":      round(intercept_below, 3),
        "intercept_above":      round(intercept_above, 3),
    }

    return results


def print_rdd_results(results: dict):
    """Prints a formatted RDD results summary."""
    print("\n--- Regression Discontinuity Results ---")
    print(f"  Bandwidth:                    ±{results['bandwidth']} years around age 18")
    print(f"  Sample below cutoff (< 18):   {results['n_below']} individuals")
    print(f"  Sample above cutoff (≥ 18):   {results['n_above']} individuals")
    print(f"\n  Predicted AUDIT-C at cutoff:")
    print(f"    Just below 18:              {results['y_hat_below']}")
    print(f"    Just above 18:              {results['y_hat_above']}")
    print(f"\n  RDD Estimate (causal jump):   {results['rdd_estimate']} AUDIT-C points")
    print(f"\n  Disorder rate below 18:       {results['disorder_rate_below']:.1%}")
    print(f"  Disorder rate above 18:       {results['disorder_rate_above']:.1%}")
    print(f"  Jump in disorder rate:        {results['disorder_jump']:.1%}")
    print(f"\n  T-statistic:                  {results['t_statistic']}")
    print(f"  P-value:                      {results['p_value']}")

    if results["p_value"] < 0.05:
        print(f"\n  ✓ Statistically significant at p < 0.05")
        print(f"  ✓ Legal alcohol access at age 18 causally increases")
        print(f"    AUDIT-C score by {results['rdd_estimate']} points")
        print(f"    and disorder rate by {results['disorder_jump']:.1%}")
    else:
        print(f"\n  ✗ Not statistically significant at p < 0.05")
    print("-----------------------------------------\n")


def run_bandwidth_sensitivity(df: pd.DataFrame):
    """
    Tests RDD estimate stability across different bandwidths.
    A robust RDD estimate should be stable across bandwidth choices.
    """
    print("Running bandwidth sensitivity analysis...")
    bandwidths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    sensitivity = []

    for bw in bandwidths:
        results = estimate_rdd_effect(df, bandwidth=bw)
        sensitivity.append({
            "bandwidth": bw,
            "rdd_estimate": results["rdd_estimate"],
            "p_value": results["p_value"],
            "n": results["n_below"] + results["n_above"],
        })

    sens_df = pd.DataFrame(sensitivity)
    print("\n  Bandwidth Sensitivity:")
    print(f"  {'Bandwidth':>10} {'RDD Estimate':>14} {'P-value':>10} {'N':>6}")
    print(f"  {'-'*44}")
    for _, row in sens_df.iterrows():
        sig = "✓" if row["p_value"] < 0.05 else "✗"
        print(f"  {row['bandwidth']:>10} {row['rdd_estimate']:>14} "
              f"{row['p_value']:>10} {int(row['n']):>6} {sig}")

    return sens_df


def save_rdd_plot(df: pd.DataFrame, results: dict):
    """
    Saves a publication-quality RDD plot showing:
    1. Scatter of individual observations with fitted lines
    2. Binned means plot
    3. Disorder rate jump
    4. Bandwidth sensitivity
    """
    ensure_directories()

    bw = results["bandwidth"]
    plot_data = df[df["age_centred"].abs() <= bw].copy()

    # Bin the data for cleaner visualisation
    plot_data["age_bin"] = pd.cut(
        plot_data["age_centred"],
        bins=np.arange(-bw, bw + 0.25, 0.25)
    )
    binned = plot_data.groupby("age_bin", observed=True).agg(
        mean_score=("alcohol_use_score", "mean"),
        mean_disorder=("disorder_flag", "mean"),
        age_mid=("age_centred", "mean"),
    ).dropna()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "ZeraMatumizi — Regression Discontinuity Analysis\n"
        "Effect of Legal Alcohol Access at Age 18 on Disorder Risk",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Plot 1: Binned means with RDD lines ---
    ax1 = fig.add_subplot(gs[0, :])
    below_bins = binned[binned["age_mid"] < 0]
    above_bins = binned[binned["age_mid"] >= 0]

    ax1.scatter(
        below_bins["age_mid"], below_bins["mean_score"],
        color="#3498DB", s=40, zorder=3, label="Below 18 (binned means)"
    )
    ax1.scatter(
        above_bins["age_mid"], above_bins["mean_score"],
        color="#E74C3C", s=40, zorder=3, label="Above 18 (binned means)"
    )

    # Fitted lines
    x_below = np.linspace(-bw, 0, 100)
    x_above = np.linspace(0, bw, 100)
    ax1.plot(
        x_below,
        results["intercept_below"] + results["slope_below"] * x_below,
        color="#3498DB", linewidth=2.5
    )
    ax1.plot(
        x_above,
        results["intercept_above"] + results["slope_above"] * x_above,
        color="#E74C3C", linewidth=2.5
    )

    # Cutoff line
    ax1.axvline(x=0, color="green", linestyle="--", linewidth=2,
                label="Age 18 cutoff (legal drinking age)")

    # Annotate the jump
    ax1.annotate(
        f"Causal jump: +{results['rdd_estimate']} AUDIT-C points\n(p = {results['p_value']})",
        xy=(0.05, results["y_hat_above"]),
        xytext=(0.5, results["y_hat_above"] + 0.5),
        fontsize=9, color="#27AE60", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27AE60")
    )

    ax1.set_title("AUDIT-C Score at Age 18 Cutoff", fontweight="bold")
    ax1.set_xlabel("Age (centred at 18 years)")
    ax1.set_ylabel("Mean AUDIT-C Score")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Disorder rate jump ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(
        below_bins["age_mid"], below_bins["mean_disorder"],
        color="#3498DB", s=40, zorder=3
    )
    ax2.scatter(
        above_bins["age_mid"], above_bins["mean_disorder"],
        color="#E74C3C", s=40, zorder=3
    )
    ax2.axvline(x=0, color="green", linestyle="--", linewidth=2)
    ax2.set_title("Disorder Rate at Age 18 Cutoff", fontweight="bold")
    ax2.set_xlabel("Age (centred at 18 years)")
    ax2.set_ylabel("Proportion with Disorder")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Summary statistics ---
    ax3 = fig.add_subplot(gs[1, 1])
    labels = ["Below 18\n(AUDIT-C)", "Above 18\n(AUDIT-C)",
              "Below 18\n(Disorder %)", "Above 18\n(Disorder %)"]
    values = [
        results["y_hat_below"],
        results["y_hat_above"],
        results["disorder_rate_below"] * 10,
        results["disorder_rate_above"] * 10,
    ]
    colours = ["#3498DB", "#E74C3C", "#85C1E9", "#F1948A"]
    bars = ax3.bar(labels, values, color=colours, edgecolor="white")
    for bar, val, orig in zip(
        bars, values,
        [results["y_hat_below"], results["y_hat_above"],
         results["disorder_rate_below"], results["disorder_rate_above"]]
    ):
        label = f"{orig:.1%}" if orig < 1 else f"{orig:.2f}"
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax3.set_title("Summary at Cutoff", fontweight="bold")
    ax3.set_ylabel("Score / Scaled Disorder Rate")
    ax3.grid(True, alpha=0.3, axis="y")

    output_path = os.path.join(REPORTS_PATH, "rdd_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ RDD plot saved: {output_path}")


if __name__ == "__main__":
    print("Running ZeraMatumizi Regression Discontinuity Analysis...")
    ensure_directories()

    df = create_rdd_sample_data()
    print(f"✓ RDD dataset created: {df.shape[0]} individuals")
    print(f"  Age range: {df['age_years'].min():.1f} – {df['age_years'].max():.1f} years")
    print(f"  Below 18: {(df['above_cutoff']==0).sum()} | Above 18: {(df['above_cutoff']==1).sum()}")

    results = estimate_rdd_effect(df, bandwidth=2.0)
    print_rdd_results(results)
    run_bandwidth_sensitivity(df)
    save_rdd_plot(df, results)

    print("\n✓ RDD analysis complete!")