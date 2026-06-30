"""
random_survival_forest.py
Random Survival Forest model for ZeraMatumizi.

Unlike the XGBoost classifier (which predicts yes/no disorder
progression), this model predicts TIME-TO-EVENT - estimating how
many months until disorder onset for each individual, while properly
handling censored data (individuals who haven't progressed yet by
the end of the observation window).

This is critical for resource planning: NACADA needs to know not just
WHO is at risk, but WHEN intervention is most urgent.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_and_engineer_survival_data() -> pd.DataFrame:
    """
    Loads the KDHS sample data and engineers survival analysis fields:
    - time_to_event: months until disorder onset (or censoring)
    - event_observed: whether disorder progression was actually observed
      (1) or the individual was censored (0) - i.e. still disorder-free
      at the end of the observation window.

    In production, time_to_event comes from longitudinal DHIS2 treatment
    registration dates. Here we simulate a realistic survival time based
    on the same risk factors used in the XGBoost model, so high-risk
    individuals have shorter simulated time-to-event.
    """
    print("Loading and engineering survival data...")
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

    # --- Simulate survival time based on risk score ---
    # Higher risk -> shorter time to event (disorder onset)
    np.random.seed(42)
    wealth_penalty = {0: 0.5, 1: 0.3, 2: 0.0, 3: -0.3, 4: -0.5}
    wealth_score = df["wealth_num"].map(wealth_penalty)

    risk_score = (
        0.9 * df["alcohol_use"]
        + 1.1 * df["cannabis_use"]
        + 0.8 * df["khat_use"]
        + 1.0 * df["unemployed_num"]
        + 0.7 * df["hiv_positive_num"]
        + 0.6 * df["early_initiation"]
        + wealth_score
        - 0.4 * (df["education_num"] == 3).astype(int)
    )

    # Baseline median survival time of 60 months (5 years), scaled by risk
    # Higher risk_score -> shorter time-to-event via exponential model
    baseline_months = 60
    hazard_multiplier = np.exp(risk_score * 0.3)
    simulated_time = baseline_months / hazard_multiplier
    simulated_time = simulated_time * np.random.uniform(0.7, 1.3, len(df))  # noise
    simulated_time = np.clip(simulated_time, 1, 120)  # 1 to 120 months

    # Observation window: 36 months. Anyone whose simulated time exceeds
    # this is censored (we never observed their disorder onset).
    observation_window = 36
    df["time_to_event"] = np.minimum(simulated_time, observation_window)
    df["event_observed"] = (simulated_time <= observation_window).astype(int)

    print(f"Survival data prepared: {len(df)} individuals")
    print(f"  Events observed (disorder onset within 36 months): {df['event_observed'].sum()} "
          f"({df['event_observed'].mean():.1%})")
    print(f"  Censored (still disorder-free at 36 months): {(1-df['event_observed']).sum()} "
          f"({1-df['event_observed'].mean():.1%})")

    return df


def get_survival_feature_columns():
    return [
        "age", "gender_num", "education_num", "wealth_num",
        "alcohol_use", "cannabis_use", "khat_use", "any_substance",
        "polysubstance", "age_of_initiation", "early_initiation",
        "hiv_positive_num", "employed_num", "unemployed_num",
    ]


def prepare_survival_arrays(df: pd.DataFrame):
    """
    Prepares the structured array format scikit-survival requires:
    a structured array of (event_observed: bool, time_to_event: float)
    """
    feature_cols = get_survival_feature_columns()
    X = df[feature_cols]

    y = Surv.from_arrays(
        event=df["event_observed"].astype(bool),
        time=df["time_to_event"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} | Test set: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_rsf_model(X_train, y_train):
    """Trains the Random Survival Forest model."""
    print("\nTraining Random Survival Forest...")
    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X_train, y_train)
    print("Model trained!")
    return rsf


def evaluate_rsf_model(rsf, X_test, y_test):
    """
    Evaluates using Harrell's concordance index (C-index) - the
    survival analysis equivalent of AUROC. Measures how well the
    model ranks individuals by risk relative to their actual
    time-to-event ordering.
    """
    risk_scores = rsf.predict(X_test)
    c_index = concordance_index_censored(
        y_test["event"], y_test["time"], risk_scores
    )[0]

    print(f"\n--- Random Survival Forest Evaluation ---")
    print(f"  Concordance Index (C-index): {c_index:.4f}")

    if c_index >= 0.70:
        print(f"  Model meets C-index >= 0.70 - good discriminative ability")
    elif c_index >= 0.60:
        print(f"  C-index {c_index:.2f} - moderate discriminative ability")
    else:
        print(f"  C-index {c_index:.2f} - weak discriminative ability, "
              f"more data/features needed")
    print("------------------------------------------\n")

    return c_index, risk_scores


def plot_survival_curves(rsf, X_test, df_test_full):
    """
    Plots predicted survival curves for a sample of high-risk vs
    low-risk individuals - showing the probability of remaining
    disorder-free over time for each group.
    """
    ensure_directories()

    risk_scores = rsf.predict(X_test)
    sorted_idx = np.argsort(risk_scores)

    # Lowest risk (longest survival) and highest risk (shortest survival)
    low_risk_idx = sorted_idx[:5]
    high_risk_idx = sorted_idx[-5:]

    surv_funcs = rsf.predict_survival_function(X_test)
    time_points = rsf.unique_times_

    plt.figure(figsize=(11, 7))

    for i in low_risk_idx:
        plt.step(
            time_points, surv_funcs[i](time_points),
            where="post", color="#27AE60", alpha=0.6, linewidth=1.5
        )
    for i in high_risk_idx:
        plt.step(
            time_points, surv_funcs[i](time_points),
            where="post", color="#E74C3C", alpha=0.6, linewidth=1.5
        )

    plt.plot([], [], color="#27AE60", linewidth=2, label="Low risk individuals (top 5)")
    plt.plot([], [], color="#E74C3C", linewidth=2, label="High risk individuals (top 5)")

    plt.xlabel("Months")
    plt.ylabel("Probability of Remaining Disorder-Free")
    plt.title(
        "ZeraMatumizi - Random Survival Forest\n"
        "Predicted Disorder-Free Survival Curves: High-Risk vs Low-Risk Individuals",
        fontsize=12, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    output_path = os.path.join(REPORTS_PATH, "survival_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Survival curves plot saved: {output_path}")


def plot_feature_importance(rsf, X_train, y_train):
    """
    Plots permutation-based feature importance for the RSF model
    using sklearn's permutation_importance, scored by concordance index.
    """
    ensure_directories()

    print("Computing permutation feature importance (this takes a moment)...")
    from sklearn.inspection import permutation_importance

    def c_index_scorer(estimator, X, y):
        risk_scores = estimator.predict(X)
        return concordance_index_censored(y["event"], y["time"], risk_scores)[0]

    result = permutation_importance(
        rsf, X_train, y_train,
        scoring=c_index_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 7))
    plt.barh(importance_df["feature"], importance_df["importance"],
             color="#3498DB", edgecolor="white")
    plt.xlabel("Permutation Importance (drop in C-index)")
    plt.title(
        "ZeraMatumizi - Random Survival Forest\nFeature Importance for Time-to-Disorder-Onset",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    output_path = os.path.join(REPORTS_PATH, "survival_feature_importance.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved: {output_path}")

    print("\nTop 5 features by importance:")
    for _, row in importance_df.sort_values("importance", ascending=False).head(5).iterrows():
        print(f"  {row['feature']:<25}: {row['importance']:.4f}")


def identify_urgent_cases(rsf, X_test, df, n=5):
    """
    Identifies the individuals with the shortest predicted time to
    disorder onset - these are the cases NACADA should prioritise
    for immediate intervention.
    """
    risk_scores = rsf.predict(X_test)
    median_survival_times = []

    surv_funcs = rsf.predict_survival_function(X_test)
    time_points = rsf.unique_times_

    for sf in surv_funcs:
        probs = sf(time_points)
        below_50 = time_points[probs <= 0.5]
        median_time = below_50[0] if len(below_50) > 0 else time_points[-1]
        median_survival_times.append(median_time)

    results_df = pd.DataFrame({
        "predicted_median_survival_months": median_survival_times,
        "risk_score": risk_scores,
    }, index=X_test.index)

    urgent = results_df.sort_values("predicted_median_survival_months").head(n)

    print(f"\n--- Top {n} Most Urgent Cases (Shortest Predicted Time to Onset) ---")
    for idx, row in urgent.iterrows():
        print(f"  Individual {idx}: predicted median time to onset = "
              f"{row['predicted_median_survival_months']:.1f} months")
    print("----------------------------------------------------------------\n")

    return urgent


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - Random Survival Forest (Time-to-Onset Model)")
    print("=" * 60)

    ensure_directories()

    df = load_and_engineer_survival_data()
    X_train, X_test, y_train, y_test = prepare_survival_arrays(df)

    rsf = train_rsf_model(X_train, y_train)
    c_index, risk_scores = evaluate_rsf_model(rsf, X_test, y_test)

    plot_survival_curves(rsf, X_test, df)
    plot_feature_importance(rsf, X_train, y_train)
    identify_urgent_cases(rsf, X_test, df)

    print("Random Survival Forest model complete!")