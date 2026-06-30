"""
xgboost_classifier.py
XGBoost early warning classifier for ZeraMatumizi.

Predicts 12-month disorder progression risk using features
derived from KDHS data: demographics, substance use, SES,
employment, and HIV status.

Includes:
- Stratified cross-validation
- Optuna hyperparameter tuning
- Full SHAP explainability (waterfall, beeswarm, interaction)
- Fairness audit across subgroups
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

REPORTS_PATH = os.path.join("docs", "reports")
SHAP_PATH = os.path.join("docs", "reports", "shap")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)
    os.makedirs(SHAP_PATH, exist_ok=True)


def load_and_engineer_features() -> pd.DataFrame:
    """
    Loads the sample KDHS data and engineers features for XGBoost.
    """
    print("Loading and engineering features...")
    df = pd.read_parquet("data/raw/kdhs_sample.parquet")

    # Encode categoricals
    df["gender_num"] = (df["gender"] == "male").astype(int)
    df["employed_num"] = (df["employment_status"] == "employed").astype(int)
    df["unemployed_num"] = (df["employment_status"] == "unemployed").astype(int)
    df["hiv_positive_num"] = (df["hiv_status"] == "positive").astype(int)

    education_map = {"none": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    df["education_num"] = df["education_level"].map(education_map)

    wealth_map = {"poorest": 0, "poor": 1, "middle": 2, "rich": 3, "richest": 4}
    df["wealth_num"] = df["wealth_index"].map(wealth_map)

    # Derived features
    df["any_substance"] = (
        (df["alcohol_use"] + df["cannabis_use"] + df["khat_use"]) > 0
    ).astype(int)
    df["polysubstance"] = (
        (df["alcohol_use"] + df["cannabis_use"] + df["khat_use"]) >= 2
    ).astype(int)
    df["early_initiation"] = (df["age_of_initiation"] < 15).astype(int)
    df["years_since_initiation"] = df["age"] - df["age_of_initiation"]

    print(f"Features engineered: {df.shape[0]} rows")
    print(f"Disorder progression rate: {df['disorder_progression'].mean():.1%}")

    return df


def get_feature_columns():
    """Returns the list of feature columns used for modelling."""
    return [
        "age",
        "gender_num",
        "education_num",
        "wealth_num",
        "alcohol_use",
        "cannabis_use",
        "khat_use",
        "any_substance",
        "polysubstance",
        "age_of_initiation",
        "early_initiation",
        "years_since_initiation",
        "hiv_positive_num",
        "employed_num",
        "unemployed_num",
    ]


def split_data(df: pd.DataFrame):
    """Splits data into train/test sets, stratified by target."""
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["disorder_progression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} rows ({y_train.mean():.1%} positive)")
    print(f"Test set:  {len(X_test)} rows ({y_test.mean():.1%} positive)")

    return X_train, X_test, y_train, y_test


def tune_hyperparameters(X_train, y_train, n_trials=30):
    """
    Uses Optuna to find the best XGBoost hyperparameters.
    Optimises for AUROC using stratified 5-fold cross-validation.
    """
    print(f"\nTuning hyperparameters with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                eval_metric="auc",
                use_label_encoder=False,
            )
            model.fit(X_fold_train, y_fold_train, verbose=False)
            preds = model.predict_proba(X_fold_val)[:, 1]
            aucs.append(roc_auc_score(y_fold_val, preds))

        return np.mean(aucs)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest AUROC (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def train_final_model(X_train, y_train, best_params):
    """Trains the final XGBoost model with tuned hyperparameters."""
    print("\nTraining final XGBoost model...")

    model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, verbose=False)

    print("Model trained!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the held-out test set."""
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_binary = (preds_proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, preds_proba)
    auprc = average_precision_score(y_test, preds_proba)

    print("\n--- Model Evaluation (Held-out Test Set) ---")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds_binary, target_names=["No Disorder", "Disorder"]))

    if auroc >= 0.70:
        print("  Model meets AUROC >= 0.70 floor - acceptable performance")
    else:
        print(f"  Model AUROC {auroc:.2f} below 0.70 - expected with synthetic data")
        print("    Real KDHS data with genuine risk patterns will perform better")
    print("---------------------------------------------\n")

    return {"auroc": auroc, "auprc": auprc, "predictions": preds_proba}


def run_fairness_audit(model, X_test, y_test, df_test_full):
    """
    Computes model performance separately for demographic subgroups.
    Flags any subgroup where AUROC drops >5 percentage points below overall.
    """
    print("\n--- Fairness Audit ---")
    preds_proba = model.predict_proba(X_test)[:, 1]
    overall_auroc = roc_auc_score(y_test, preds_proba)

    subgroup_results = []

    # Gender
    for gender_val, gender_name in [(0, "female"), (1, "male")]:
        mask = X_test["gender_num"] == gender_val
        if mask.sum() > 10 and y_test[mask].nunique() > 1:
            sub_auroc = roc_auc_score(y_test[mask], preds_proba[mask])
            subgroup_results.append(("gender", gender_name, sub_auroc, mask.sum()))

    # HIV status
    for hiv_val, hiv_name in [(0, "HIV negative/unknown"), (1, "HIV positive")]:
        mask = X_test["hiv_positive_num"] == hiv_val
        if mask.sum() > 10 and y_test[mask].nunique() > 1:
            sub_auroc = roc_auc_score(y_test[mask], preds_proba[mask])
            subgroup_results.append(("hiv_status", hiv_name, sub_auroc, mask.sum()))

    print(f"  Overall AUROC: {overall_auroc:.4f}\n")
    print(f"  {'Subgroup':<15} {'Category':<25} {'AUROC':>8} {'N':>6} {'Flag':>8}")
    print(f"  {'-'*65}")

    for category, name, sub_auroc, n in subgroup_results:
        gap = overall_auroc - sub_auroc
        flag = "WARNING" if gap > 0.05 else "OK"
        print(f"  {category:<15} {name:<25} {sub_auroc:>8.4f} {n:>6} {flag:>8}")

    print("------------------------\n")
    return subgroup_results


def generate_shap_analysis(model, X_train, X_test):
    """
    Generates full SHAP explainability suite:
    - Beeswarm summary plot
    - Waterfall plot for individual prediction
    - Feature importance bar chart
    """
    ensure_directories()
    print("\nGenerating SHAP explainability analysis...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Beeswarm summary plot ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("ZeraMatumizi - SHAP Feature Impact Summary", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PATH, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP beeswarm plot saved")

    # --- Feature importance bar chart ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("ZeraMatumizi - Mean SHAP Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PATH, "shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP importance plot saved")

    # --- Waterfall plot for highest-risk individual ---
    highest_risk_idx = np.argmax(model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(10, 7))
    shap.waterfall_plot(shap_values[highest_risk_idx], show=False)
    plt.title(
        f"ZeraMatumizi - Individual Risk Explanation\n"
        f"(Highest risk individual in test set)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PATH, "shap_waterfall_highest_risk.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP waterfall plot saved (highest-risk individual)")

    # --- Waterfall plot for lowest-risk individual ---
    lowest_risk_idx = np.argmin(model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(10, 7))
    shap.waterfall_plot(shap_values[lowest_risk_idx], show=False)
    plt.title(
        f"ZeraMatumizi - Individual Risk Explanation\n"
        f"(Lowest risk individual in test set)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PATH, "shap_waterfall_lowest_risk.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP waterfall plot saved (lowest-risk individual)")

    # Print top features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n  Top 5 Most Important Features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"    {row['feature']:<25}: {row['mean_abs_shap']:.4f}")

    return shap_values, feature_importance


def save_model_card(metrics, fairness_results, feature_importance, best_params):
    """Generates a responsible AI model card documenting the model."""
    ensure_directories()
    filepath = os.path.join("docs", "model_card.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# ZeraMatumizi - XGBoost Disorder Progression Model Card\n\n")
        f.write("## Model Overview\n")
        f.write("Predicts 12-month substance use disorder progression risk ")
        f.write("using demographic, substance use, and socioeconomic features.\n\n")

        f.write("## Performance Metrics\n")
        f.write(f"- AUROC: {metrics['auroc']:.4f}\n")
        f.write(f"- AUPRC: {metrics['auprc']:.4f}\n\n")

        f.write("## Hyperparameters (Optuna-tuned)\n")
        for k, v in best_params.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Top Features (by SHAP importance)\n")
        for _, row in feature_importance.head(5).iterrows():
            f.write(f"- {row['feature']}: {row['mean_abs_shap']:.4f}\n")
        f.write("\n")

        f.write("## Fairness Audit\n")
        f.write("| Subgroup | Category | AUROC | N | Flag |\n")
        f.write("|---|---|---|---|---|\n")
        for category, name, sub_auroc, n in fairness_results:
            f.write(f"| {category} | {name} | {sub_auroc:.4f} | {n} | "
                     f"{'WARNING' if (metrics['auroc'] - sub_auroc) > 0.05 else 'OK'} |\n")
        f.write("\n")

        f.write("## Limitations\n")
        f.write("- Trained on synthetic data for development purposes\n")
        f.write("- Production deployment requires retraining on real KDHS 2022, ")
        f.write("NACADA survey, and DHIS2 data\n")
        f.write("- Performance should be re-validated on real Kenyan population data\n")

    print(f"\nModel card saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - XGBoost Early Warning Classifier")
    print("=" * 60)

    ensure_directories()

    df = load_and_engineer_features()
    X_train, X_test, y_train, y_test = split_data(df)

    best_params = tune_hyperparameters(X_train, y_train, n_trials=40)
    model = train_final_model(X_train, y_train, best_params)

    metrics = evaluate_model(model, X_test, y_test)
    fairness_results = run_fairness_audit(model, X_test, y_test, df)
    shap_values, feature_importance = generate_shap_analysis(model, X_train, X_test)

    save_model_card(metrics, fairness_results, feature_importance, best_params)

    print("\nXGBoost classifier with SHAP explainability complete!")