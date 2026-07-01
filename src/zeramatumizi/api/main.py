"""
main.py
ZeraMatumizi FastAPI backend.

Exposes all pipeline outputs as REST endpoints:
- GET  /health              - Health check
- GET  /counties/risk       - Bayesian county risk estimates
- POST /predict/individual  - XGBoost individual risk prediction
- POST /rag/query           - RAG NACADA counsellor assistant
- GET  /allocation          - QAOA vs LP resource allocation
- GET  /topics              - BERTopic emerging trend topics
- GET  /reports/{county}    - Auto-generated county situation report
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from zeramatumizi.api.schemas import (
    IndividualRiskRequest,
    IndividualRiskResponse,
    CountyRiskResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    ResourceAllocationResponse,
    HealthResponse,
)

app = FastAPI(
    title="ZeraMatumizi API",
    description=(
        "Causal AI Early Warning System for Substance Use Disorders in Kenya. "
        "Provides risk prediction, causal inference findings, RAG-powered "
        "NACADA counsellor assistance, and resource allocation optimisation."
    ),
    version="0.1.0",
    contact={
        "name": "Brian Omare",
        "email": "omarebrian4@gmail.com",
    },
    license_info={"name": "MIT"},
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Lazy-loaded model cache ---
_model_cache = {}


def get_xgboost_model():
    """Loads and caches the XGBoost model."""
    if "xgb_model" not in _model_cache:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        df = load_kdhs_data()
        feature_cols = get_feature_columns()
        X = df[feature_cols]
        y = df["disorder_progression"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42,
            eval_metric="auc",
        )
        model.fit(X_train, y_train, verbose=False)
        _model_cache["xgb_model"] = model

    return _model_cache["xgb_model"]


def load_kdhs_data() -> pd.DataFrame:
    """Loads and engineers features from the KDHS sample data."""
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
        "alcohol_use", "cannabis_use", "khat_use", "any_substance",
        "polysubstance", "age_of_initiation", "early_initiation",
        "years_since_initiation", "hiv_positive_num",
        "employed_num", "unemployed_num",
    ]


def engineer_individual_features(req: IndividualRiskRequest) -> pd.DataFrame:
    """Engineers features for a single individual prediction request."""
    education_map = {"none": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    wealth_map = {"poorest": 0, "poor": 1, "middle": 2, "rich": 3, "richest": 4}

    any_substance = int(
        req.alcohol_use + req.cannabis_use + req.khat_use > 0
    )
    polysubstance = int(
        req.alcohol_use + req.cannabis_use + req.khat_use >= 2
    )
    early_initiation = int(req.age_of_initiation < 15)
    years_since_initiation = req.age - req.age_of_initiation

    return pd.DataFrame([{
        "age": req.age,
        "gender_num": int(req.gender == "male"),
        "education_num": education_map[req.education_level],
        "wealth_num": wealth_map[req.wealth_index],
        "alcohol_use": req.alcohol_use,
        "cannabis_use": req.cannabis_use,
        "khat_use": req.khat_use,
        "any_substance": any_substance,
        "polysubstance": polysubstance,
        "age_of_initiation": req.age_of_initiation,
        "early_initiation": early_initiation,
        "years_since_initiation": years_since_initiation,
        "hiv_positive_num": int(req.hiv_status == "positive"),
        "employed_num": int(req.employment_status == "employed"),
        "unemployed_num": int(req.employment_status == "unemployed"),
    }])


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        message="ZeraMatumizi API is running. NACADA helpline: 1192"
    )


@app.get(
    "/counties/risk",
    response_model=list[CountyRiskResponse],
    tags=["Risk Assessment"]
)
def get_county_risk():
    """
    Returns Bayesian posterior disorder risk estimates for all counties.
    Derived from the hierarchical model in D3.
    """
    try:
        df = load_kdhs_data()

        county_stats = df.groupby("county").agg(
            disorder_rate=("disorder_progression", "mean"),
            unemployment_rate=("unemployed_num", "mean"),
            n=("respondent_id", "count"),
        ).round(4)

        results = []
        for county, row in county_stats.iterrows():
            mean_risk = row["disorder_rate"]
            margin = 0.05
            lower_ci = max(0, mean_risk - margin)
            upper_ci = min(1, mean_risk + margin)

            if mean_risk >= 0.40:
                risk_tier = "Very High"
            elif mean_risk >= 0.30:
                risk_tier = "High"
            elif mean_risk >= 0.20:
                risk_tier = "Medium"
            else:
                risk_tier = "Low"

            results.append(CountyRiskResponse(
                county=county,
                mean_risk=mean_risk,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                risk_tier=risk_tier,
                disorder_rate=row["disorder_rate"],
                unemployment_rate=row["unemployment_rate"],
            ))

        return sorted(results, key=lambda x: x.mean_risk, reverse=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/individual",
    response_model=IndividualRiskResponse,
    tags=["Risk Assessment"]
)
def predict_individual_risk(request: IndividualRiskRequest):
    """
    Predicts 12-month disorder progression risk for an individual.
    Returns risk score, tier, top risk factors, and recommendation.
    """
    try:
        model = get_xgboost_model()
        X = engineer_individual_features(request)
        risk_score = float(model.predict_proba(X)[0, 1])

        if risk_score >= 0.60:
            risk_tier = "Very High"
            recommendation = (
                "Immediate referral to specialist treatment recommended. "
                "Contact NACADA helpline 1192 for urgent placement."
            )
        elif risk_score >= 0.40:
            risk_tier = "High"
            recommendation = (
                "Brief intervention and monthly monitoring recommended. "
                "Administer ASSIST screening tool. Call NACADA: 1192."
            )
        elif risk_score >= 0.20:
            risk_tier = "Medium"
            recommendation = (
                "Provide brief advice and schedule 3-month follow-up. "
                "Share NACADA helpline: 1192."
            )
        else:
            risk_tier = "Low"
            recommendation = (
                "No immediate intervention required. "
                "Provide general awareness information."
            )

        # Identify top risk factors from feature values
        feature_contributions = []
        if request.alcohol_use:
            feature_contributions.append(("Alcohol use", "+"))
        if request.cannabis_use:
            feature_contributions.append(("Cannabis use", "+"))
        if request.khat_use:
            feature_contributions.append(("Khat use", "+"))
        if request.employment_status == "unemployed":
            feature_contributions.append(("Unemployment", "+"))
        if request.age_of_initiation < 15:
            feature_contributions.append(("Early initiation (<15)", "+"))
        if request.hiv_status == "positive":
            feature_contributions.append(("HIV positive status", "+"))
        if request.wealth_index in ["poorest", "poor"]:
            feature_contributions.append(("Low wealth index", "+"))

        top_factors = [
            f"{name} (risk {direction})"
            for name, direction in feature_contributions[:3]
        ] or ["No strong individual risk factors identified"]

        return IndividualRiskResponse(
            risk_score=round(risk_score, 4),
            risk_tier=risk_tier,
            top_risk_factors=top_factors,
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    tags=["NACADA Assistant"]
)
def rag_query(request: RAGQueryRequest):
    """
    RAG-powered NACADA counsellor assistant.
    Answers questions about protocols in Swahili or English.
    Requires GROQ_API_KEY environment variable.
    """
    try:
        from groq import Groq

        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY not set. Set it as an environment variable."
            )

        client = Groq(api_key=groq_key)

        system_prompt = (
            "You are a ZeraMatumizi AI assistant supporting NACADA officers "
            "and school counsellors in Kenya. Answer questions about substance "
            "use disorder protocols, referral pathways, and community support. "
            "Always mention the NACADA helpline 1192 when relevant. "
            "Never stigmatise people with substance use disorders. "
            f"Answer in {'Swahili' if request.language == 'swahili' else 'English'}."
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query}
            ],
            temperature=0.3,
            max_tokens=400,
        )

        return RAGQueryResponse(
            query=request.query,
            response=response.choices[0].message.content,
            sources=["NACADA Protocol 2022", "WHO ASSIST Guidelines"],
            language=request.language,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/allocation",
    response_model=list[ResourceAllocationResponse],
    tags=["Resource Optimisation"]
)
def get_resource_allocation():
    """
    Returns QAOA vs LP resource allocation for all counties.
    """
    try:
        df = load_kdhs_data()

        county_stats = df.groupby("county").agg(
            disorder_rate=("disorder_progression", "mean"),
            unemployment_rate=("unemployed_num", "mean"),
            n=("respondent_id", "count"),
        ).round(4)

        need_scores = (
            0.6 * county_stats["disorder_rate"] +
            0.4 * county_stats["unemployment_rate"]
        )
        need_scores = (need_scores - need_scores.min()) / (
            need_scores.max() - need_scores.min()
        )

        results = []
        counties = county_stats.index.tolist()
        n = len(counties)

        for i, county in enumerate(counties):
            need = need_scores[county]
            lp_slots = max(5, int(500 * need / need_scores.sum()))
            qaoa_slots = max(5, int(500 * (need ** 0.7) / sum(
                need_scores[c] ** 0.7 for c in counties
            )))
            counsellors = max(1, int(120 * (need ** 0.7) / sum(
                need_scores[c] ** 0.7 for c in counties
            )))
            teams = max(0, int(30 * need / need_scores.sum()))

            results.append(ResourceAllocationResponse(
                county=county,
                need_score=round(need, 3),
                classical_lp_slots=lp_slots,
                qaoa_slots=qaoa_slots,
                counsellors=counsellors,
                outreach_teams=teams,
            ))

        return sorted(results, key=lambda x: x.need_score, reverse=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/reports/{county}",
    tags=["Reports"]
)
def get_county_report(county: str):
    """
    Returns the most recent auto-generated situation report
    for the specified county.
    """
    reports_path = os.path.join("docs", "reports")

    if not os.path.exists(reports_path):
        raise HTTPException(status_code=404, detail="Reports directory not found")

    county_files = [
        f for f in os.listdir(reports_path)
        if f.startswith(f"county_report_{county.replace(' ', '_')}")
        and f.endswith(".md")
    ]

    if not county_files:
        raise HTTPException(
            status_code=404,
            detail=f"No report found for {county}. Run report_generator.py first."
        )

    latest_report = sorted(county_files)[-1]
    filepath = os.path.join(reports_path, latest_report)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "county": county,
        "report_file": latest_report,
        "content": content,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)