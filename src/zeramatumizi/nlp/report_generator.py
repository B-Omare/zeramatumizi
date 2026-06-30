"""
report_generator.py
LLM-powered automated monthly county situation report generator
for ZeraMatumizi.

Combines outputs from across the pipeline - Bayesian county risk
estimates, topic modelling trends, sentiment/distress signals, and
causal inference findings - into a structured, readable monthly
report that NACADA county officers can use directly.

Uses Groq's free Llama API (same as rag_pipeline.py).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from groq import Groq

REPORTS_PATH = os.path.join("docs", "reports")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_pipeline_outputs() -> dict:
    """
    Loads outputs from earlier pipeline stages to feed into the report.
    Falls back gracefully with placeholder context if a file is missing -
    each module can be run independently.
    """
    context = {}

    # County risk from Bayesian model (re-derive simply from sample data
    # since hierarchical_model.py doesn't persist a CSV currently)
    try:
        df = pd.read_parquet("data/raw/kdhs_sample.parquet")
        county_summary = df.groupby("county").agg(
            n=("respondent_id", "count"),
            disorder_rate=("disorder_progression", "mean"),
            alcohol_rate=("alcohol_use", "mean"),
            cannabis_rate=("cannabis_use", "mean"),
            khat_rate=("khat_use", "mean"),
            unemployment_rate=("employment_status", lambda x: (x == "unemployed").mean()),
        ).round(3)
        context["county_summary"] = county_summary
        print(f"Loaded county summary for {len(county_summary)} counties")
    except FileNotFoundError:
        context["county_summary"] = None
        print("KDHS sample data not found - run loader.py first")

    # Social media topics and sentiment
    try:
        social_df = pd.read_parquet("data/raw/social_media_sample.parquet")
        context["n_posts"] = len(social_df)
        context["topic_categories"] = social_df["expected_category"].value_counts().to_dict()
        print(f"Loaded {len(social_df)} social media posts for context")
    except FileNotFoundError:
        context["n_posts"] = 0
        context["topic_categories"] = {}
        print("Social media sample not found - run topic_modelling.py first")

    return context


def select_focus_county(context: dict) -> str:
    """
    Selects the highest-risk county to generate a focused report on,
    based on disorder progression rate.
    """
    if context["county_summary"] is not None:
        focus_county = context["county_summary"]["disorder_rate"].idxmax()
        return focus_county
    return "Nyamira"  # fallback default


def build_report_context(context: dict, focus_county: str) -> str:
    """
    Builds a structured text context summarising all pipeline findings
    for the focus county, to feed into the LLM.
    """
    lines = []

    if context["county_summary"] is not None:
        row = context["county_summary"].loc[focus_county]
        lines.append(f"COUNTY: {focus_county}")
        lines.append(f"Sample size: {int(row['n'])} respondents")
        lines.append(f"Disorder progression rate: {row['disorder_rate']:.1%}")
        lines.append(f"Alcohol use rate: {row['alcohol_rate']:.1%}")
        lines.append(f"Cannabis use rate: {row['cannabis_rate']:.1%}")
        lines.append(f"Khat use rate: {row['khat_rate']:.1%}")
        lines.append(f"Unemployment rate: {row['unemployment_rate']:.1%}")

        national_avg = context["county_summary"]["disorder_rate"].mean()
        lines.append(f"National average disorder rate (sample counties): {national_avg:.1%}")

    if context["topic_categories"]:
        lines.append(f"\nDIGITAL SURVEILLANCE SIGNALS ({context['n_posts']} posts analysed):")
        for cat, count in sorted(context["topic_categories"].items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count} posts")

    lines.append(f"\nCAUSAL EVIDENCE BASE (from prior analysis):")
    lines.append("  - NACADA campaigns reduced disorder registrations by ~27% in treated counties (DiD analysis)")
    lines.append("  - Legal alcohol access at age 18 causally increases disorder risk by ~56% (RDD analysis)")
    lines.append("  - Each additional weekly drink causally increases disorder severity (IV analysis, chang'aa proximity instrument)")

    return "\n".join(lines)


def generate_report(focus_county: str, report_context: str) -> str:
    """
    Sends the structured context to Groq's Llama model to generate
    a polished, NACADA-officer-ready monthly situation report.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set - run 'set GROQ_API_KEY=your-key' first")

    client = Groq(api_key=GROQ_API_KEY)

    month_year = datetime.now().strftime("%B %Y")

    prompt = (
        f"You are a report-writing assistant for NACADA (Kenya's National "
        f"Authority for the Campaign Against Alcohol and Drug Abuse).\n\n"
        f"Write a professional Monthly County Situation Report for {focus_county} "
        f"County for {month_year}, based strictly on the data below. The report "
        f"will be read by a county NACADA officer making resource allocation decisions.\n\n"
        f"Structure the report with these sections:\n"
        f"1. Executive Summary (2-3 sentences)\n"
        f"2. Key Risk Indicators (bullet points with the actual numbers)\n"
        f"3. Digital Surveillance Findings (what online signals suggest)\n"
        f"4. Evidence-Based Recommendations (2-3 specific, actionable steps "
        f"grounded in the causal evidence provided)\n"
        f"5. Resources (mention NACADA helpline 1192)\n\n"
        f"DATA:\n{report_context}\n\n"
        f"Write the report now. Be concise, professional, and avoid stigmatising "
        f"language. Use actual numbers from the data, not vague language."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=900,
    )

    return response.choices[0].message.content


def save_report(report_text: str, focus_county: str):
    """Saves the generated report to docs/reports."""
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m")
    filename = f"county_report_{focus_county.replace(' ', '_')}_{timestamp}.md"
    filepath = os.path.join(REPORTS_PATH, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# ZeraMatumizi Monthly Situation Report\n\n")
        f.write(report_text)

    print(f"\nReport saved: {filepath}")
    return filepath


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - Automated County Report Generator")
    print("=" * 60)

    ensure_directories()

    context = load_pipeline_outputs()
    focus_county = select_focus_county(context)
    print(f"\nGenerating report for highest-risk county: {focus_county}")

    report_context = build_report_context(context, focus_county)
    print(f"\n--- Report Context Sent to LLM ---")
    print(report_context)
    print("-----------------------------------\n")

    print("Generating report via Groq Llama...")
    report_text = generate_report(focus_county, report_context)

    print(f"\n--- Generated Report ---\n")
    print(report_text)
    print(f"\n------------------------\n")

    save_report(report_text, focus_county)

    print("Report generation complete!")