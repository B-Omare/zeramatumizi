"""
hierarchical_model.py
Bayesian hierarchical logistic model for ZeraMatumizi.

Models substance use disorder progression across Kenyan counties,
with county-level random effects capturing geographic clustering.

Structure:
- Level 1: Individual respondents
- Level 2: Counties (random intercepts)

Output:
- Posterior disorder probability per county
- Credible intervals for all parameters
- County risk ranking with uncertainty
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_and_prepare_data():
    """
    Loads the sample KDHS data and prepares it for
    Bayesian hierarchical modelling.
    """
    print("Loading and preparing data...")

    df = pd.read_parquet("data/raw/kdhs_sample.parquet")

    # Encode county as integer index
    counties = df["county"].unique().tolist()
    county_to_idx = {c: i for i, c in enumerate(counties)}
    df["county_idx"] = df["county"].map(county_to_idx)

    # Encode categorical variables as numeric
    df["gender_num"] = (df["gender"] == "male").astype(int)
    df["employed_num"] = (df["employment_status"] == "employed").astype(int)
    df["hiv_pos_num"] = (df["hiv_status"] == "positive").astype(int)

    # Standardise continuous variables
    df["age_std"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["initiation_std"] = (
        df["age_of_initiation"] - df["age_of_initiation"].mean()
    ) / df["age_of_initiation"].std()

    # Any substance use flag
    df["any_substance"] = (
        (df["alcohol_use"] + df["cannabis_use"] + df["khat_use"]) > 0
    ).astype(int)

    print(f"✓ Data prepared: {len(df)} individuals across {len(counties)} counties")
    print(f"  Disorder progression rate: {df['disorder_progression'].mean():.1%}")
    print(f"  Counties: {counties}")

    return df, counties, county_to_idx


def build_and_sample_model(df, counties):
    """
    Builds and samples the Bayesian hierarchical logistic model.

    Model structure:
    disorder_progression ~ Bernoulli(p)
    logit(p) = alpha_county[county] + beta * X

    Priors:
    alpha_county ~ Normal(mu_alpha, sigma_alpha)  # county random effects
    mu_alpha ~ Normal(0, 1)                        # hyperprior mean
    sigma_alpha ~ HalfNormal(0.5)                  # hyperprior SD
    betas ~ Normal(0, 0.5)                         # feature coefficients
    """
    print("\nBuilding Bayesian hierarchical model...")
    n_counties = len(counties)

    # Feature matrix
    X = df[[
        "age_std",
        "gender_num",
        "employed_num",
        "hiv_pos_num",
        "any_substance",
        "initiation_std",
    ]].values

    y = df["disorder_progression"].values
    county_idx = df["county_idx"].values

    with pm.Model() as hierarchical_model:

        # --- Hyperpriors (county-level) ---
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)

        # --- County random intercepts ---
        alpha_county = pm.Normal(
            "alpha_county",
            mu=mu_alpha,
            sigma=sigma_alpha,
            shape=n_counties
        )

        # --- Individual-level fixed effects ---
        betas = pm.Normal("betas", mu=0, sigma=0.5, shape=X.shape[1])

        # --- Linear predictor ---
        logit_p = (
            alpha_county[county_idx] +
            pm.math.dot(X, betas)
        )

        # --- Likelihood ---
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        outcome = pm.Bernoulli("outcome", p=p, observed=y)

        # --- Sample ---
        print("Sampling posterior (this takes 2-4 minutes)...")
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            progressbar=True,
            return_inferencedata=True,
            nuts_sampler="numpyro" if False else "pymc",
        )

    print("✓ Sampling complete!")
    return hierarchical_model, trace


def print_convergence_diagnostics(trace, counties):
    """
    Prints convergence diagnostics.
    R-hat < 1.01 means the chains have converged.
    ESS > 400 means enough effective samples.
    """
    print("\n--- Convergence Diagnostics ---")

    summary = az.summary(
        trace,
        var_names=["mu_alpha", "sigma_alpha", "betas"],
        round_to=3
    )
    print(summary.to_string())

    # Check R-hat
    rhat = az.rhat(trace, var_names=["alpha_county"])
    max_rhat = float(rhat["alpha_county"].max())
    print(f"\n  Max R-hat (alpha_county): {max_rhat:.4f}")
    if max_rhat < 1.01:
        print(f"  ✓ All chains converged (R-hat < 1.01)")
    else:
        print(f"  ⚠ Convergence issues detected (R-hat > 1.01)")
    print("-------------------------------\n")


def extract_county_risk(trace, counties) -> pd.DataFrame:
    """
    Extracts posterior county risk estimates.
    Returns a DataFrame with mean risk and credible intervals per county.
    """
    # Extract county intercepts
    alpha_samples = trace.posterior["alpha_county"].values
    # Shape: (chains, draws, counties)
    alpha_flat = alpha_samples.reshape(-1, len(counties))

    # Convert log-odds to probability
    prob_samples = 1 / (1 + np.exp(-alpha_flat))

    county_risk = pd.DataFrame({
        "county": counties,
        "mean_risk": prob_samples.mean(axis=0),
        "lower_ci": np.percentile(prob_samples, 2.5, axis=0),
        "upper_ci": np.percentile(prob_samples, 97.5, axis=0),
        "sd": prob_samples.std(axis=0),
    }).sort_values("mean_risk", ascending=False)

    return county_risk


def print_county_risk_table(county_risk: pd.DataFrame):
    """Prints a formatted county risk ranking table."""
    print("\n--- County Disorder Risk Ranking ---")
    print(f"  {'Rank':<6} {'County':<15} {'Mean Risk':>10} "
          f"{'95% CI Lower':>13} {'95% CI Upper':>13} {'Uncertainty':>12}")
    print(f"  {'-'*72}")

    for rank, (_, row) in enumerate(county_risk.iterrows(), 1):
        uncertainty = "HIGH ⚠" if row["sd"] > 0.05 else "low ✓"
        print(
            f"  {rank:<6} {row['county']:<15} "
            f"{row['mean_risk']:>10.1%} "
            f"{row['lower_ci']:>13.1%} "
            f"{row['upper_ci']:>13.1%} "
            f"{uncertainty:>12}"
        )
    print("------------------------------------\n")


def save_county_risk_plot(county_risk: pd.DataFrame, trace):
    """
    Saves two plots:
    1. County risk ranking with credible intervals
    2. Trace plot for convergence visualisation
    """
    ensure_directories()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "ZeraMatumizi — Bayesian Hierarchical Model\n"
        "Posterior Disorder Risk by County with 95% Credible Intervals",
        fontsize=13, fontweight="bold"
    )

    # --- Plot 1: County risk with credible intervals ---
    ax1 = axes[0]
    counties_sorted = county_risk["county"].tolist()
    mean_risk = county_risk["mean_risk"].tolist()
    lower = county_risk["lower_ci"].tolist()
    upper = county_risk["upper_ci"].tolist()

    # Colour by risk level
    colours = [
        "#E74C3C" if r > 0.20 else
        "#E67E22" if r > 0.15 else
        "#F1C40F" if r > 0.10 else
        "#27AE60"
        for r in mean_risk
    ]

    y_pos = range(len(counties_sorted))
    ax1.barh(y_pos, mean_risk, color=colours, alpha=0.8, edgecolor="white")

    # Credible interval lines
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        ax1.plot([lo, hi], [i, i], color="#2C3E50", linewidth=2.5, zorder=5)
        ax1.plot(lo, i, "|", color="#2C3E50", markersize=8, zorder=6)
        ax1.plot(hi, i, "|", color="#2C3E50", markersize=8, zorder=6)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(counties_sorted, fontsize=10)
    ax1.set_xlabel("Posterior Disorder Probability")
    ax1.set_title("County Risk Ranking\n(bars = posterior mean, lines = 95% CI)",
                  fontweight="bold")
    ax1.axvline(x=0.15, color="red", linestyle="--",
                alpha=0.5, label="High risk threshold (15%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    # Add risk labels
    for i, (r, lo, hi) in enumerate(zip(mean_risk, lower, upper)):
        ax1.text(
            hi + 0.002, i, f"{r:.1%}",
            va="center", fontsize=8, fontweight="bold"
        )

    # --- Plot 2: Posterior distributions ---
    ax2 = axes[1]
    alpha_samples = trace.posterior["alpha_county"].values
    alpha_flat = alpha_samples.reshape(-1, len(counties_sorted))
    prob_samples = 1 / (1 + np.exp(-alpha_flat))

    cmap = plt.cm.RdYlGn_r
    for i, county in enumerate(counties_sorted):
        colour = cmap(mean_risk[i] / max(mean_risk))
        ax2.hist(
            prob_samples[:, county_risk.index[
                county_risk["county"] == county
            ][0] if False else
            list(county_risk["county"]).index(county)],
            bins=30, alpha=0.6, color=colour,
            label=county, density=True
        )

    ax2.set_xlabel("Posterior Disorder Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Posterior Distributions by County",
                  fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    plt.tight_layout()
    output_path = os.path.join(REPORTS_PATH, "bayesian_county_risk.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ County risk plot saved: {output_path}")


def save_trace_plot(trace):
    """Saves ArviZ trace plot for convergence diagnostics."""
    ensure_directories()
    axes = az.plot_trace(
        trace,
        var_names=["mu_alpha", "sigma_alpha"],
        figsize=(12, 6)
    )
    plt.suptitle(
        "ZeraMatumizi — MCMC Trace Plots\nConvergence Diagnostics",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    output_path = os.path.join(REPORTS_PATH, "bayesian_trace.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Trace plot saved: {output_path}")


if __name__ == "__main__":
    print("Running ZeraMatumizi Bayesian Hierarchical Model...")
    ensure_directories()

    # Load data
    df, counties, county_to_idx = load_and_prepare_data()

    # Build and sample model
    model, trace = build_and_sample_model(df, counties)

    # Diagnostics
    print_convergence_diagnostics(trace, counties)

    # County risk estimates
    county_risk = extract_county_risk(trace, counties)
    print_county_risk_table(county_risk)

    # Save plots
    save_county_risk_plot(county_risk, trace)
    save_trace_plot(trace)

    print("\n✓ Bayesian hierarchical model complete!")