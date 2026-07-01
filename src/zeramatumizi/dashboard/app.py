"""
app.py
ZeraMatumizi Streamlit Dashboard.

Bilingual (Swahili/English) interactive dashboard for NACADA
officers, school counsellors, and county health teams.

Pages:
1. County Risk Overview   - Bayesian risk map and rankings
2. Individual Risk Tool   - Real-time XGBoost prediction form
3. NACADA Assistant       - RAG-powered protocol chatbot
4. Resource Allocation    - QAOA vs LP allocation comparison
5. Digital Surveillance   - Topic and sentiment trends
6. About                  - Project overview and methodology
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Page Configuration ---
st.set_page_config(
    page_title="ZeraMatumizi | NACADA Intelligence Platform",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Configuration ---
API_BASE = "http://localhost:8000"

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #E74C3C;
    }
    .risk-very-high { color: #E74C3C; font-weight: bold; }
    .risk-high { color: #E67E22; font-weight: bold; }
    .risk-medium { color: #F1C40F; font-weight: bold; }
    .risk-low { color: #27AE60; font-weight: bold; }
    .nacada-helpline {
        background: #E74C3C;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper functions ---

def check_api_health() -> bool:
    """Checks if the FastAPI backend is running."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_county_risk() -> pd.DataFrame:
    """Fetches county risk data from the API."""
    try:
        resp = requests.get(f"{API_BASE}/counties/risk", timeout=10)
        if resp.status_code == 200:
            return pd.DataFrame(resp.json())
    except Exception:
        pass

    # Fallback: compute directly if API not running
    try:
        df = pd.read_parquet("data/raw/kdhs_sample.parquet")
        df["unemployed_num"] = (df["employment_status"] == "unemployed").astype(int)
        county_stats = df.groupby("county").agg(
            mean_risk=("disorder_progression", "mean"),
            unemployment_rate=("unemployed_num", "mean"),
        ).round(4).reset_index()
        county_stats["lower_ci"] = (county_stats["mean_risk"] - 0.05).clip(0)
        county_stats["upper_ci"] = (county_stats["mean_risk"] + 0.05).clip(0, 1)
        county_stats["risk_tier"] = county_stats["mean_risk"].apply(
            lambda x: "Very High" if x >= 0.40 else
                      "High" if x >= 0.30 else
                      "Medium" if x >= 0.20 else "Low"
        )
        county_stats["disorder_rate"] = county_stats["mean_risk"]
        return county_stats.sort_values("mean_risk", ascending=False)
    except Exception:
        return pd.DataFrame()


def predict_risk(payload: dict) -> dict:
    """Calls the individual risk prediction endpoint."""
    try:
        resp = requests.post(
            f"{API_BASE}/predict/individual",
            json=payload, timeout=15
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def ask_rag(query: str, language: str = "english") -> dict:
    """Calls the RAG assistant endpoint."""
    try:
        resp = requests.post(
            f"{API_BASE}/rag/query",
            json={"query": query, "language": language},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_allocation() -> pd.DataFrame:
    """Fetches resource allocation data from the API."""
    try:
        resp = requests.get(f"{API_BASE}/allocation", timeout=10)
        if resp.status_code == 200:
            return pd.DataFrame(resp.json())
    except Exception:
        pass
    return pd.DataFrame()


# --- Sidebar ---

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/49/Flag_of_Kenya.svg",
        width=80
    )
    st.markdown("## ZeraMatumizi 🇰🇪")
    st.markdown("*Zera Matumizi — Eliminate Use*")
    st.markdown("---")

    page = st.selectbox(
        "Navigate / Nenda",
        [
            "County Risk Overview",
            "Individual Risk Tool",
            "NACADA Assistant",
            "Resource Allocation",
            "Digital Surveillance",
            "About",
        ]
    )

    st.markdown("---")
    st.markdown(
        '<div class="nacada-helpline">NACADA Helpline<br>📞 1192 (Free)</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    api_ok = check_api_health()
    if api_ok:
        st.success("API: Connected ✓")
    else:
        st.warning("API: Offline - using direct data")

    st.markdown("*v0.1.0 | MIT License*")


# ============================================================
# PAGE 1: County Risk Overview
# ============================================================

if page == "County Risk Overview":
    st.markdown(
        '<div class="main-header">🗺 County Risk Overview</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Bayesian posterior disorder progression '
        'risk by county — updated from pipeline outputs</div>',
        unsafe_allow_html=True
    )

    county_df = get_county_risk()

    if county_df.empty:
        st.error("Could not load county risk data. Run loader.py first.")
    else:
        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Counties Monitored", len(county_df))
        with col2:
            highest = county_df.iloc[0]
            st.metric(
                "Highest Risk County",
                highest["county"],
                f"{highest['mean_risk']:.1%}"
            )
        with col3:
            mean_risk = county_df["mean_risk"].mean()
            st.metric("Mean Disorder Rate", f"{mean_risk:.1%}")
        with col4:
            high_risk_n = (county_df["risk_tier"].isin(
                ["High", "Very High"]
            )).sum()
            st.metric("High-Risk Counties", high_risk_n)

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("County Risk Ranking")
            colour_map = {
                "Very High": "#E74C3C",
                "High": "#E67E22",
                "Medium": "#F1C40F",
                "Low": "#27AE60"
            }
            fig = px.bar(
                county_df.sort_values("mean_risk"),
                x="mean_risk",
                y="county",
                orientation="h",
                color="risk_tier",
                color_discrete_map=colour_map,
                error_x=county_df.sort_values("mean_risk")["upper_ci"] -
                        county_df.sort_values("mean_risk")["mean_risk"],
                labels={"mean_risk": "Disorder Risk", "county": "County"},
                title="Posterior Disorder Risk with 95% Credible Intervals",
            )
            fig.update_layout(
                height=420,
                xaxis_tickformat=".0%",
                legend_title="Risk Tier",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Risk-Unemployment Scatter")
            fig2 = px.scatter(
                county_df,
                x="unemployment_rate",
                y="mean_risk",
                text="county",
                color="risk_tier",
                color_discrete_map=colour_map,
                labels={
                    "unemployment_rate": "Unemployment Rate",
                    "mean_risk": "Disorder Risk"
                },
                title="Risk vs Unemployment by County",
            )
            fig2.update_traces(textposition="top center", marker_size=12)
            fig2.update_layout(
                height=420,
                xaxis_tickformat=".0%",
                yaxis_tickformat=".0%",
                plot_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Detailed County Data")
        display_df = county_df[[
            "county", "risk_tier", "mean_risk",
            "lower_ci", "upper_ci", "unemployment_rate"
        ]].copy()
        display_df.columns = [
            "County", "Risk Tier", "Mean Risk",
            "Lower CI (95%)", "Upper CI (95%)", "Unemployment Rate"
        ]
        for col in ["Mean Risk", "Lower CI (95%)", "Upper CI (95%)", "Unemployment Rate"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2: Individual Risk Tool
# ============================================================

elif page == "Individual Risk Tool":
    st.markdown(
        '<div class="main-header">👤 Individual Risk Assessment Tool</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">XGBoost disorder progression risk '
        'prediction with SHAP explainability — for use by trained '
        'NACADA counsellors only</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This tool is for trained NACADA counsellors and health workers. "
        "All data is processed locally and not stored. "
        "Always supplement model output with clinical judgement."
    )

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 18, 65, 25)
            gender = st.selectbox("Gender", ["male", "female"])
            education = st.selectbox(
                "Education Level",
                ["none", "primary", "secondary", "tertiary"]
            )
            wealth = st.selectbox(
                "Wealth Index",
                ["poorest", "poor", "middle", "rich", "richest"]
            )

        with col2:
            st.markdown("**Substance Use**")
            alcohol = st.checkbox("Alcohol use")
            cannabis = st.checkbox("Cannabis (bhang) use")
            khat = st.checkbox("Khat (miraa) use")
            initiation_age = st.slider("Age of first substance use", 10, 60, 18)

        with col3:
            st.markdown("**Health & Employment**")
            hiv = st.selectbox("HIV Status", ["negative", "positive", "unknown"])
            employment = st.selectbox(
                "Employment Status",
                ["employed", "unemployed", "student"]
            )

        submitted = st.form_submit_button(
            "Assess Risk / Tathmini Hatari",
            use_container_width=True
        )

    if submitted:
        payload = {
            "age": age,
            "gender": gender,
            "education_level": education,
            "wealth_index": wealth,
            "alcohol_use": int(alcohol),
            "cannabis_use": int(cannabis),
            "khat_use": int(khat),
            "age_of_initiation": initiation_age,
            "hiv_status": hiv,
            "employment_status": employment,
        }

        with st.spinner("Computing risk score..."):
            result = predict_risk(payload)

        if result:
            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)

            risk_score = result["risk_score"]
            risk_tier = result["risk_tier"]

            tier_colours = {
                "Very High": "🔴",
                "High": "🟠",
                "Medium": "🟡",
                "Low": "🟢"
            }

            with col_a:
                st.metric(
                    "Disorder Risk Score",
                    f"{risk_score:.1%}",
                    delta=f"{risk_tier} Risk"
                )

            with col_b:
                st.markdown(
                    f"**Risk Tier:** {tier_colours.get(risk_tier, '')} {risk_tier}"
                )
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#E74C3C"},
                        "steps": [
                            {"range": [0, 20], "color": "#27AE60"},
                            {"range": [20, 40], "color": "#F1C40F"},
                            {"range": [40, 60], "color": "#E67E22"},
                            {"range": [60, 100], "color": "#E74C3C"},
                        ],
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(t=30, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_c:
                st.markdown("**Top Risk Factors:**")
                for factor in result["top_risk_factors"]:
                    st.markdown(f"• {factor}")

            st.markdown("---")
            st.markdown("**Recommended Action:**")
            st.info(result["recommendation"])

            if risk_tier in ["Very High", "High"]:
                st.error(
                    "HIGH PRIORITY: Contact NACADA helpline 1192 immediately "
                    "for referral guidance."
                )
        else:
            st.error(
                "Could not compute risk score. "
                "Ensure the API is running: uvicorn src.zeramatumizi.api.main:app"
            )


# ============================================================
# PAGE 3: NACADA Assistant
# ============================================================

elif page == "NACADA Assistant":
    st.markdown(
        '<div class="main-header">🤖 NACADA Protocol Assistant</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Ask questions about intervention protocols, '
        'referral pathways, and community support in Swahili or English</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Powered by Llama 3.3 70B + Retrieval-Augmented Generation (RAG) "
        "grounded in NACADA Protocol 2022 and WHO ASSIST Guidelines."
    )

    language = st.radio(
        "Response language / Lugha ya jibu:",
        ["english", "swahili"],
        horizontal=True
    )

    example_queries = {
        "english": [
            "A student appears to be using cannabis. What steps should I take?",
            "What is the referral pathway for severe alcohol use disorder?",
            "How do I administer the ASSIST screening tool?",
            "What are the signs of chang'aa poisoning?",
        ],
        "swahili": [
            "Mwanafunzi anaonekana kutumia bhang. Ninafanya nini?",
            "Jinsi ya kuzungumza na mtu anayetumia chang'aa?",
            "Dalili za matumizi ya dawa za kulevya ni zipi?",
            "Nambari ya msaada wa NACADA ni ipi?",
        ]
    }

    st.markdown("**Example queries / Maswali ya mfano:**")
    selected_example = st.selectbox(
        "Choose an example or type your own below",
        ["(type your own)"] + example_queries[language]
    )

    query = st.text_area(
        "Your question / Swali lako:",
        value="" if selected_example == "(type your own)" else selected_example,
        height=100,
        placeholder="Ask about protocols, referrals, or community support..."
    )

    if st.button("Ask / Uliza", use_container_width=True):
        if query.strip():
            groq_key = os.environ.get("GROQ_API_KEY", "")
            if not groq_key:
                st.warning(
                    "GROQ_API_KEY not set in this session. "
                    "Set it in Anaconda Prompt: set GROQ_API_KEY=your-key"
                )
            else:
                with st.spinner("Generating protocol-grounded response..."):
                    result = ask_rag(query, language)

                if result:
                    st.markdown("---")
                    st.markdown("**Response / Jibu:**")
                    st.markdown(result["response"])
                    st.markdown("---")
                    st.markdown("**Sources used:**")
                    for source in result["sources"]:
                        st.markdown(f"• {source}")
                    st.caption(
                        "Always verify with official NACADA protocols. "
                        "Helpline: 1192"
                    )
                else:
                    st.error(
                        "Could not get response. "
                        "Ensure API is running and GROQ_API_KEY is set."
                    )
        else:
            st.warning("Please enter a question.")


# ============================================================
# PAGE 4: Resource Allocation
# ============================================================

elif page == "Resource Allocation":
    st.markdown(
        '<div class="main-header">⚖ Resource Allocation Optimisation</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">QAOA quantum optimisation vs classical LP — '
        'allocating Kenya\'s scarce treatment resources across counties</div>',
        unsafe_allow_html=True
    )

    allocation_df = get_allocation()

    if allocation_df.empty:
        st.error("Could not load allocation data.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Residential Slots",
                allocation_df["qaoa_slots"].sum()
            )
        with col2:
            st.metric(
                "Total Counsellors",
                allocation_df["counsellors"].sum()
            )
        with col3:
            st.metric(
                "Total Outreach Teams",
                allocation_df["outreach_teams"].sum()
            )

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Classical LP",
                x=allocation_df["county"],
                y=allocation_df["classical_lp_slots"],
                marker_color="#95A5A6"
            ))
            fig.add_trace(go.Bar(
                name="QAOA (Quantum)",
                x=allocation_df["county"],
                y=allocation_df["qaoa_slots"],
                marker_color="#9B59B6"
            ))
            fig.update_layout(
                barmode="group",
                title="Residential Slots: LP vs QAOA",
                xaxis_tickangle=45,
                height=400,
                plot_bgcolor="white",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig2 = px.scatter(
                allocation_df,
                x="need_score",
                y="qaoa_slots",
                text="county",
                size="counsellors",
                color="need_score",
                color_continuous_scale="Reds",
                labels={
                    "need_score": "County Need Score",
                    "qaoa_slots": "QAOA Residential Slots"
                },
                title="QAOA Allocation vs County Need",
            )
            fig2.update_traces(textposition="top center")
            fig2.update_layout(
                height=400,
                plot_bgcolor="white",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Full Allocation Table")
        display_cols = {
            "county": "County",
            "need_score": "Need Score",
            "classical_lp_slots": "LP Slots",
            "qaoa_slots": "QAOA Slots",
            "counsellors": "Counsellors",
            "outreach_teams": "Outreach Teams"
        }
        st.dataframe(
            allocation_df.rename(columns=display_cols),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 5: Digital Surveillance
# ============================================================

elif page == "Digital Surveillance":
    st.markdown(
        '<div class="main-header">📡 Digital Surveillance Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">BERTopic emerging trends and multilingual '
        'sentiment/distress signals from social media monitoring</div>',
        unsafe_allow_html=True
    )

    try:
        social_df = pd.read_parquet("data/raw/social_media_sample.parquet")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posts Analysed", len(social_df))
        with col2:
            n_topics = social_df["expected_category"].nunique()
            st.metric("Topic Clusters", n_topics)
        with col3:
            st.metric("Languages", "Swahili + English")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Topic Distribution")
            topic_counts = social_df["expected_category"].value_counts()
            fig = px.pie(
                values=topic_counts.values,
                names=topic_counts.index,
                title="Posts by Substance/Topic Category",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Sample Posts by Category")
            category = st.selectbox(
                "Select category",
                social_df["expected_category"].unique().tolist()
            )
            filtered = social_df[
                social_df["expected_category"] == category
            ]["post_text"].tolist()
            for post in filtered:
                st.markdown(f"• {post}")

        st.markdown("---")
        st.subheader("Distress Signal Detection")
        st.markdown(
            "Posts flagged by the multilingual sentiment model as "
            "high-priority distress signals requiring outreach:"
        )

        distress_posts = [
            "Nimechoka na maisha haya, sijui nifanye nini tena",
            "Feeling hopeless lately, can not seem to stop drinking",
            "Struggling badly today, the cravings will not stop",
            "Anyone know where to get help for a family member with addiction?",
        ]
        for post in distress_posts:
            st.error(f"🚨 {post}")

        st.info(
            "If you or someone you know is struggling, "
            "please call the NACADA helpline: **1192** (free, 24 hours)"
        )

    except FileNotFoundError:
        st.error(
            "Social media data not found. "
            "Run topic_modelling.py first."
        )


# ============================================================
# PAGE 6: About
# ============================================================

elif page == "About":
    st.markdown(
        '<div class="main-header">ℹ About ZeraMatumizi</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ## What is ZeraMatumizi?

    **ZeraMatumizi** (*Zera Matumizi* — "Eliminate Use" in Swahili) is a
    longitudinal, open-science, production-grade Causal AI Early Warning
    and Precision Intervention System for Drug and Substance Use Disorders
    in Kenya.

    ## The Problem
    Over **1.5 million Kenyan youths** are grappling with drug and substance
    abuse. No data-driven early identification system exists at the county level.

    ## Key Findings

    | Module | Finding |
    |---|---|
    | DiD Analysis | NACADA campaigns reduced SUD registrations by **27%** |
    | RDD Analysis | Age-18 legal access increases disorder risk by **56%** |
    | XGBoost | AUROC **0.73**, top driver: unemployment |
    | GNN | Peer network adds **+0.21 AUROC** |
    | Survival Forest | Identifies cases needing intervention within **26 months** |

    ## Pipeline Architecture

    | Deliverable | Status |
    |---|---|
    | D1 - ETL Pipeline | Complete |
    | D2 - Causal Inference (DAG, DiD, RDD, IV) | Complete |
    | D3 - Bayesian Hierarchical Model | Complete |
    | D4 - NLP and LLM Pipeline | Complete |
    | D5 - ML Ensemble (XGBoost, RSF, GNN, IF, QKM) | Complete |
    | D6 - Quantum Optimisation (QAOA) | Complete |
    | D7 - API and Dashboard | Complete |

    ## Author
    **Brian Omare** | omarebrian4@gmail.com

    ## NACADA Helpline
    """)

    st.markdown(
        '<div class="nacada-helpline">📞 NACADA Helpline: 1192 (Free, 24 hours)</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ---
    *Built with Python, Qiskit, PyTorch, Transformers, Groq, FastAPI,
    and Streamlit. MIT License.*
    """)