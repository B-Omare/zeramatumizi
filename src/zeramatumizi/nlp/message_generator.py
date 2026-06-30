"""
message_generator.py
LLM-powered de-stigmatisation message generator for ZeraMatumizi.

Generates short, culturally appropriate, non-stigmatising public
health messages in Swahili and English for radio spots, SMS campaigns,
and community posters - tailored per county based on the dominant
substance and risk profile identified by the pipeline.

Uses Groq's free Llama API (same as rag_pipeline.py and report_generator.py).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from datetime import datetime
from groq import Groq

REPORTS_PATH = os.path.join("docs", "reports")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Stigmatising language patterns to check against -
# generated messages must NOT contain these
STIGMA_RED_FLAGS_EN = [
    "addict", "junkie", "drunkard", "useless", "shameful",
    "disgrace", "weak-willed", "criminal"
]
STIGMA_RED_FLAGS_SW = [
    "mlevi", "mzembe", "fedhuli", "aibu", "mhalifu"
]


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def get_county_risk_profile(county: str) -> dict:
    """
    Derives a simple risk profile for a county from the sample KDHS
    data, to tailor message content to the dominant substance.
    """
    try:
        df = pd.read_parquet("data/raw/kdhs_sample.parquet")
        county_df = df[df["county"] == county]

        if len(county_df) == 0:
            return {"county": county, "dominant_substance": "alcohol", "n": 0}

        rates = {
            "alcohol": county_df["alcohol_use"].mean(),
            "cannabis": county_df["cannabis_use"].mean(),
            "khat": county_df["khat_use"].mean(),
        }
        dominant = max(rates, key=rates.get)

        return {
            "county": county,
            "dominant_substance": dominant,
            "dominant_rate": rates[dominant],
            "n": len(county_df),
        }
    except FileNotFoundError:
        return {"county": county, "dominant_substance": "alcohol", "n": 0}


def build_message_prompt(profile: dict, channel: str, language: str) -> str:
    """
    Builds the LLM prompt for generating a single de-stigmatisation
    message, tailored by channel (radio/SMS/poster) and language.
    """
    substance_label = {
        "alcohol": "alcohol and illicit brew (chang'aa)",
        "cannabis": "cannabis (bhang)",
        "khat": "khat (miraa)",
    }.get(profile["dominant_substance"], "substance use")

    channel_specs = {
        "radio": "a 30-second radio spot script (around 80-100 words), warm and conversational tone, suitable for a community radio station",
        "sms": "a single SMS message under 160 characters, direct and actionable",
        "poster": "short poster text (a headline and 2-3 supporting lines), suitable for a community noticeboard",
    }

    language_instruction = (
        "Write entirely in Swahili." if language == "swahili"
        else "Write entirely in English."
    )

    prompt = (
        f"You are a public health communications expert working with NACADA "
        f"(Kenya's National Authority for the Campaign Against Alcohol and "
        f"Drug Abuse) to reduce stigma around substance use disorders.\n\n"
        f"Write {channel_specs[channel]} for {profile['county']} County, "
        f"focused on {substance_label}, which is the most prevalent substance "
        f"of concern in this county based on recent survey data.\n\n"
        f"{language_instruction}\n\n"
        f"STRICT RULES:\n"
        f"- NEVER use stigmatising words like 'addict', 'junkie', 'drunkard', "
        f"'criminal', 'mlevi', 'fedhuli', or similar shaming language\n"
        f"- Frame substance use disorder as a treatable health condition, "
        f"not a moral failing\n"
        f"- Always include a clear, hopeful call to action\n"
        f"- Always mention the NACADA helpline: 1192\n"
        f"- Use warm, respectful, community-oriented language\n"
        f"- Do not be preachy or judgmental in tone\n\n"
        f"Write only the message text, nothing else."
    )

    return prompt


def check_for_stigma(text: str) -> list:
    """
    Checks generated text against known stigmatising language patterns.
    Returns a list of any red flags found - acts as a safety net on
    top of the prompt-level instructions.
    """
    text_lower = text.lower()
    flags = []

    for word in STIGMA_RED_FLAGS_EN + STIGMA_RED_FLAGS_SW:
        if word in text_lower:
            flags.append(word)

    return flags


def generate_message(profile: dict, channel: str, language: str) -> dict:
    """
    Generates a single message and runs it through the stigma checker.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set - run 'set GROQ_API_KEY=your-key' first")

    client = Groq(api_key=GROQ_API_KEY)
    prompt = build_message_prompt(profile, channel, language)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300,
    )

    message_text = response.choices[0].message.content.strip()
    stigma_flags = check_for_stigma(message_text)

    return {
        "county": profile["county"],
        "channel": channel,
        "language": language,
        "dominant_substance": profile["dominant_substance"],
        "message": message_text,
        "stigma_flags": stigma_flags,
        "passed_stigma_check": len(stigma_flags) == 0,
    }


def print_message(result: dict):
    """Prints a formatted message result."""
    print(f"\n{'='*60}")
    print(f"COUNTY: {result['county']} | CHANNEL: {result['channel'].upper()} "
          f"| LANGUAGE: {result['language'].upper()}")
    print(f"Focus substance: {result['dominant_substance']}")
    print(f"{'='*60}")
    print(f"\n{result['message']}\n")

    if result["passed_stigma_check"]:
        print("STIGMA CHECK: PASSED - no flagged language detected")
    else:
        print(f"STIGMA CHECK: FAILED - flagged words: {result['stigma_flags']}")
    print(f"{'='*60}\n")


def save_messages(results: list):
    """Saves all generated messages to a markdown file for review."""
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(REPORTS_PATH, f"destigma_messages_{timestamp}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# ZeraMatumizi - Generated De-Stigmatisation Messages\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for r in results:
            f.write(f"## {r['county']} - {r['channel'].upper()} ({r['language']})\n\n")
            f.write(f"**Focus substance:** {r['dominant_substance']}\n\n")
            f.write(f"> {r['message']}\n\n")
            status = "PASSED" if r["passed_stigma_check"] else f"FAILED - {r['stigma_flags']}"
            f.write(f"**Stigma check:** {status}\n\n")
            f.write("---\n\n")

    print(f"\nAll messages saved: {filepath}")
    return filepath


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - De-Stigmatisation Message Generator")
    print("=" * 60)

    ensure_directories()

    # Generate a varied set of messages across counties, channels, languages
    jobs = [
        ("Nyamira", "radio", "swahili"),
        ("Kisumu", "sms", "english"),
        ("Kakamega", "poster", "swahili"),
        ("Homa Bay", "radio", "english"),
        ("Migori", "sms", "swahili"),
    ]

    results = []
    for county, channel, language in jobs:
        profile = get_county_risk_profile(county)
        result = generate_message(profile, channel, language)
        print_message(result)
        results.append(result)

    n_passed = sum(r["passed_stigma_check"] for r in results)
    print(f"\nStigma check summary: {n_passed}/{len(results)} messages passed")

    save_messages(results)

    print("Message generation complete!")