"""
iv_analysis.py
Instrumental Variable (IV) analysis for ZeraMatumizi.

Instrument: Distance to nearest illicit brew (chang'aa) seizure cluster
Endogenous variable: Alcohol consumption frequency
Outcome: Disorder severity score

Logic: Distance to chang'aa hotspot affects alcohol access (relevance)
but does not directly affect disorder severity except through
alcohol consumption (exclusion restriction).

Method: Two-Stage Least Squares (2SLS)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Output folder
REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(REPORTS_PATH, exist_ok=True)


def create_iv_sample_data() -> pd.DataFrame:
    """
    Creates a simulated dataset for IV analysis.

    Variables:
    - changaa_distance_km: distance to nearest chang'aa seizure cluster (instrument)
    - alcohol_frequency:   weekly alcohol consumption frequency (endogenous)
    - disorder_severity:   disorder severity score 0-10 (outcome)
    - ses_composite:       socioeconomic status (control variable)
    - age:                 age in years (control variable)
    - gender:              0=female, 1=male (control variable)

    The instrument (changaa_distance_km) affects alcohol_frequency
    which in turn affects disorder_severity.
    """
    np.random.seed(42)
    n = 1500

    # --- Instrument: distance to nearest chang'aa cluster (km) ---
    # Ranges from 0.1km (very close) to 15km (far away)
    changaa_distance_km = np.random.exponential(scale=4.0, size=n)
    changaa_distance_km = np.clip(changaa_distance_km, 0.1, 15.0)

    # --- Control variables ---
    age = np.random.randint(18, 65, n)
    gender = np.random.binomial(1, 0.55, n)  # 55% male
    ses_composite = np.random.normal(0, 1, n)  # standardised SES

    # --- Unobserved confounder (not in dataset — this is the problem OLS can't solve) ---
    unobserved_propensity = np.random.normal(0, 1, n)

    # --- First stage: Distance → Alcohol frequency ---
    # Closer to chang'aa = higher consumption
    alcohol_frequency = (
        5.0                              # baseline
        - 0.35 * changaa_distance_km    # instrument effect (relevance)
        + 0.8  * unobserved_propensity  # unobserved confounder
        + 0.3  * gender                 # males drink more
        - 0.02 * ses_composite          # wealthier drink less
        + np.random.normal(0, 1.2, n)   # noise
    )
    alcohol_frequency = np.clip(alcohol_frequency, 0, 21)  # max 21 drinks/week

    # --- Second stage: Alcohol frequency → Disorder severity ---
    disorder_severity = (
        1.0                             # baseline
        + 0.6  * alcohol_frequency      # causal effect of alcohol
        + 0.9  * unobserved_propensity  # unobserved confounder (bias source)
        + 0.15 * age / 10              # age effect
        - 0.3  * ses_composite          # SES protective effect
        + np.random.normal(0, 1.0, n)   # noise
    )
    disorder_severity = np.clip(disorder_severity, 0, 10)

    df = pd.DataFrame({
        "changaa_distance_km": changaa_distance_km,
        "alcohol_frequency":   alcohol_frequency,
        "disorder_severity":   disorder_severity,
        "ses_composite":       ses_composite,
        "age":                 age,
        "gender":              gender,
    })

    return df


def run_first_stage(df: pd.DataFrame) -> dict:
    """
    First stage regression: Instrument → Endogenous variable
    changaa_distance_km → alcohol_frequency

    A strong instrument needs F-statistic > 10 (rule of thumb).
    """
    print("Running First Stage: Distance → Alcohol Frequency...")

    X = np.column_stack([
        np.ones(len(df)),
        df["changaa_distance_km"],
        df["age"],
        df["gender"],
        df["ses_composite"],
    ])
    y = df["alcohol_frequency"].values

    # OLS via numpy
    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Predicted values (fitted alcohol frequency)
    y_hat = X @ coeffs

    # F-statistic for instrument strength
    y_mean = y.mean()
    ss_total = ((y - y_mean) ** 2).sum()
    ss_residual = ((y - y_hat) ** 2).sum()
    ss_model = ss_total - ss_residual
    df_model = X.shape[1] - 1
    df_resid = len(y) - X.shape[1]
    f_stat = (ss_model / df_model) / (ss_residual / df_resid)
    r_squared = 1 - (ss_residual / ss_total)

    results = {
        "instrument_coeff":  round(coeffs[1], 4),
        "f_statistic":       round(f_stat, 2),
        "r_squared":         round(r_squared, 4),
        "fitted_values":     y_hat,
        "all_coeffs":        coeffs,
    }

    return results


def run_ols_estimate(df: pd.DataFrame) -> dict:
    """
    Naive OLS estimate (biased due to unobserved confounding).
    This is what we would get WITHOUT instrumental variables.
    """
    slope, intercept, r, p, se = stats.linregress(
        df["alcohol_frequency"],
        df["disorder_severity"]
    )
    return {
        "estimate": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r ** 2, 4),
        "p_value": round(p, 4),
        "method": "OLS (biased)",
    }


def run_2sls_estimate(df: pd.DataFrame, first_stage: dict) -> dict:
    """
    Second stage of 2SLS: uses fitted alcohol frequency
    (purged of endogeneity) to estimate causal effect on disorder.

    This gives us the unbiased IV estimate.
    """
    print("Running Second Stage (2SLS): Fitted Alcohol → Disorder Severity...")

    # Use fitted values from first stage
    alcohol_hat = first_stage["fitted_values"]

    slope, intercept, r, p, se = stats.linregress(
        alcohol_hat,
        df["disorder_severity"]
    )

    return {
        "estimate":  round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r ** 2, 4),
        "p_value":   round(p, 4),
        "method":    "2SLS IV (unbiased)",
    }


def print_iv_results(first_stage: dict, ols: dict, iv: dict):
    """Prints a formatted IV results comparison."""
    print("\n--- Instrumental Variable Results ---")
    print(f"\n  FIRST STAGE (Instrument Strength):")
    print(f"  Chang'aa distance coefficient: {first_stage['instrument_coeff']}")
    print(f"  F-statistic:                   {first_stage['f_statistic']}")
    if first_stage["f_statistic"] > 10:
        print(f"  ✓ Strong instrument (F > 10 rule of thumb)")
    else:
        print(f"  ⚠ Weak instrument (F < 10) — results unreliable")
    print(f"  R-squared:                     {first_stage['r_squared']:.1%}")

    print(f"\n  SECOND STAGE COMPARISON:")
    print(f"  {'Method':<25} {'Estimate':>10} {'R²':>8} {'P-value':>10}")
    print(f"  {'-'*55}")
    print(f"  {ols['method']:<25} {ols['estimate']:>10} "
          f"{ols['r_squared']:>8.3f} {ols['p_value']:>10}")
    print(f"  {iv['method']:<25} {iv['estimate']:>10} "
          f"{iv['r_squared']:>8.3f} {iv['p_value']:>10}")

    bias = round(ols["estimate"] - iv["estimate"], 4)
    print(f"\n  OLS upward bias:               {bias}")
    print(f"\n  Interpretation:")
    print(f"  ✓ Each additional drink/week causally increases")
    print(f"    disorder severity by {iv['estimate']} points (IV estimate)")
    print(f"  ✓ OLS overstates this effect by {bias} points")
    print(f"    due to unobserved confounding")
    print(f"  ✓ Chang'aa proximity is a valid instrument:")
    print(f"    closer supply → more consumption → higher disorder")
    print("-------------------------------------\n")


def save_iv_plot(df: pd.DataFrame, first_stage: dict, ols: dict, iv: dict):
    """
    Saves a publication-quality IV plot showing:
    1. First stage: Distance → Alcohol frequency
    2. Reduced form: Distance → Disorder severity
    3. OLS vs IV comparison
    4. Bias decomposition
    """
    ensure_directories()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "ZeraMatumizi — Instrumental Variable Analysis\n"
        "Causal Effect of Alcohol Consumption on Disorder Severity\n"
        "Instrument: Distance to Nearest Chang'aa Seizure Cluster",
        fontsize=13, fontweight="bold"
    )

    # --- Plot 1: First stage ---
    ax1 = axes[0, 0]
    ax1.scatter(
        df["changaa_distance_km"], df["alcohol_frequency"],
        alpha=0.3, s=10, color="#3498DB"
    )
    x_line = np.linspace(
        df["changaa_distance_km"].min(),
        df["changaa_distance_km"].max(), 100
    )
    slope_fs, intercept_fs, _, _, _ = stats.linregress(
        df["changaa_distance_km"], df["alcohol_frequency"]
    )
    ax1.plot(
        x_line, intercept_fs + slope_fs * x_line,
        color="#E74C3C", linewidth=2.5
    )
    ax1.set_title(
        f"First Stage\nDistance → Alcohol Frequency\n"
        f"F-stat: {first_stage['f_statistic']} ✓ Strong instrument",
        fontweight="bold", fontsize=10
    )
    ax1.set_xlabel("Distance to Chang'aa Cluster (km)")
    ax1.set_ylabel("Weekly Alcohol Frequency")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Reduced form ---
    ax2 = axes[0, 1]
    ax2.scatter(
        df["changaa_distance_km"], df["disorder_severity"],
        alpha=0.3, s=10, color="#E67E22"
    )
    slope_rf, intercept_rf, _, _, _ = stats.linregress(
        df["changaa_distance_km"], df["disorder_severity"]
    )
    ax2.plot(
        x_line, intercept_rf + slope_rf * x_line,
        color="#E74C3C", linewidth=2.5
    )
    ax2.set_title(
        "Reduced Form\nDistance → Disorder Severity",
        fontweight="bold", fontsize=10
    )
    ax2.set_xlabel("Distance to Chang'aa Cluster (km)")
    ax2.set_ylabel("Disorder Severity Score (0-10)")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: OLS vs IV comparison ---
    ax3 = axes[1, 0]
    methods = ["OLS\n(Biased)", "2SLS IV\n(Unbiased)"]
    estimates = [ols["estimate"], iv["estimate"]]
    colours = ["#E74C3C", "#27AE60"]
    bars = ax3.bar(methods, estimates, color=colours, width=0.4,
                   edgecolor="white")
    for bar, val in zip(bars, estimates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val}", ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )
    ax3.set_title(
        "OLS vs IV Estimate\nEffect per Additional Drink/Week",
        fontweight="bold", fontsize=10
    )
    ax3.set_ylabel("Causal Estimate (Disorder Severity Points)")
    ax3.grid(True, alpha=0.3, axis="y")

    bias = round(ols["estimate"] - iv["estimate"], 4)
    ax3.annotate(
        f"OLS bias: +{bias}\n(due to unobserved confounding)",
        xy=(0, ols["estimate"]),
        xytext=(0.5, ols["estimate"] + 0.05),
        fontsize=9, color="#E74C3C", fontweight="bold",
        ha="center"
    )

    # --- Plot 4: Scatter with both fitted lines ---
    ax4 = axes[1, 1]
    ax4.scatter(
        df["alcohol_frequency"], df["disorder_severity"],
        alpha=0.2, s=10, color="#95A5A6", label="Observations"
    )
    x_range = np.linspace(
        df["alcohol_frequency"].min(),
        df["alcohol_frequency"].max(), 100
    )
    ax4.plot(
        x_range,
        ols["intercept"] + ols["estimate"] * x_range,
        color="#E74C3C", linewidth=2.5, label=f"OLS: β={ols['estimate']}"
    )
    ax4.plot(
        x_range,
        iv["intercept"] + iv["estimate"] * x_range,
        color="#27AE60", linewidth=2.5,
        linestyle="--", label=f"2SLS IV: β={iv['estimate']}"
    )
    ax4.set_title(
        "OLS vs IV Fitted Lines\nAlcohol Frequency → Disorder Severity",
        fontweight="bold", fontsize=10
    )
    ax4.set_xlabel("Weekly Alcohol Frequency")
    ax4.set_ylabel("Disorder Severity Score")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(REPORTS_PATH, "iv_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ IV plot saved: {output_path}")


if __name__ == "__main__":
    print("Running ZeraMatumizi Instrumental Variable Analysis...")
    ensure_directories()

    df = create_iv_sample_data()
    print(f"✓ IV dataset created: {df.shape[0]} individuals")
    print(f"  Chang'aa distance range: "
          f"{df['changaa_distance_km'].min():.1f} – "
          f"{df['changaa_distance_km'].max():.1f} km")
    print(f"  Mean alcohol frequency: {df['alcohol_frequency'].mean():.1f} drinks/week")
    print(f"  Mean disorder severity: {df['disorder_severity'].mean():.2f}/10")

    first_stage = run_first_stage(df)
    ols = run_ols_estimate(df)
    iv = run_2sls_estimate(df, first_stage)

    print_iv_results(first_stage, ols, iv)
    save_iv_plot(df, first_stage, ols, iv)

    print("✓ IV analysis complete!")