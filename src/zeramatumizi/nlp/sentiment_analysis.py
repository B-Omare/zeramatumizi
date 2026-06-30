"""
sentiment_analysis.py
Multilingual sentiment and distress signal detection for ZeraMatumizi.

Applies multilingual sentiment analysis to social media-style posts
and clinical/community notes to flag individuals or communities showing
signs of distress related to substance use - enabling early outreach
before crisis points.

Model: multilingual BERT-based sentiment classifier (XLM-RoBERTa)
which supports Swahili-English code-switched text common in Kenyan
digital discourse.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

REPORTS_PATH = os.path.join("docs", "reports")
RAW_DATA_PATH = os.path.join("data", "raw")

# Distress keywords used as an additional rule-based signal layer
# (combined with model sentiment for higher reliability on short,
# code-switched text where model confidence alone can be noisy)
DISTRESS_KEYWORDS_SW = [
    "nimechoka", "sijui nifanye", "hakuna msaada", "naumia",
    "sina tumaini", "nataka kufa", "naumwa", "nimekata tamaa"
]
DISTRESS_KEYWORDS_EN = [
    "hopeless", "can't stop", "struggling", "cravings won't stop",
    "need help", "give up", "no one understands", "exhausted"
]


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)


def load_sample_posts() -> pd.DataFrame:
    """
    Loads the sample social posts created in topic_modelling.py.
    Falls back to creating them if the file doesn't exist yet.
    """
    filepath = os.path.join(RAW_DATA_PATH, "social_media_sample.parquet")

    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        print(f"Loaded {len(df)} sample posts from topic_modelling output")
        return df

    print("Sample posts file not found - run topic_modelling.py first")
    raise FileNotFoundError(filepath)


def load_sentiment_model():
    """
    Loads a multilingual sentiment classification model.
    Uses cardiffnlp's XLM-RoBERTa model which supports 8+ languages
    and handles code-switched text reasonably well.
    """
    print("Loading multilingual sentiment model (first run downloads ~1GB)...")
    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        top_k=None,
    )
    print("Model loaded!")
    return classifier


def classify_sentiment(text: str, classifier) -> dict:
    """
    Classifies sentiment of a single text.
    Returns the dominant label and confidence score.
    """
    results = classifier(text[:512])[0]  # truncate to model max length
    best = max(results, key=lambda x: x["score"])
    return {
        "sentiment": best["label"],
        "confidence": round(best["score"], 3),
    }


def detect_distress_keywords(text: str) -> dict:
    """
    Rule-based distress keyword detection as a complementary signal
    to model-based sentiment, since short code-switched social posts
    can be noisy for pure model-based classification.
    """
    text_lower = text.lower()
    matched_sw = [kw for kw in DISTRESS_KEYWORDS_SW if kw in text_lower]
    matched_en = [kw for kw in DISTRESS_KEYWORDS_EN if kw in text_lower]
    all_matches = matched_sw + matched_en

    return {
        "distress_keyword_flag": len(all_matches) > 0,
        "matched_keywords": all_matches,
    }


def run_sentiment_pipeline(df: pd.DataFrame, classifier) -> pd.DataFrame:
    """
    Runs sentiment classification and distress keyword detection
    on every post in the dataset.
    """
    print(f"\nAnalysing sentiment for {len(df)} posts...")

    sentiments = []
    confidences = []
    distress_flags = []
    matched_kw_list = []

    for text in df["post_text"]:
        sent_result = classify_sentiment(text, classifier)
        distress_result = detect_distress_keywords(text)

        sentiments.append(sent_result["sentiment"])
        confidences.append(sent_result["confidence"])
        distress_flags.append(distress_result["distress_keyword_flag"])
        matched_kw_list.append(", ".join(distress_result["matched_keywords"]))

    df["sentiment"] = sentiments
    df["sentiment_confidence"] = confidences
    df["distress_keyword_flag"] = distress_flags
    df["matched_keywords"] = matched_kw_list

    # Combined distress signal: negative sentiment + high confidence,
    # OR explicit distress keywords matched
    df["high_priority_distress"] = (
        ((df["sentiment"] == "negative") & (df["sentiment_confidence"] > 0.6))
        | df["distress_keyword_flag"]
    )

    print("Sentiment analysis complete!")
    return df


def print_sentiment_summary(df: pd.DataFrame):
    """Prints a summary of sentiment distribution and flagged posts."""
    print("\n--- Sentiment Analysis Summary ---")
    print(f"\n  Sentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    n_distress = df["high_priority_distress"].sum()
    print(f"\n  High-priority distress signals flagged: {n_distress} / {len(df)}")

    if n_distress > 0:
        print(f"\n  Flagged posts requiring outreach:")
        flagged = df[df["high_priority_distress"]]
        for _, row in flagged.iterrows():
            kw_note = f" [keywords: {row['matched_keywords']}]" if row["matched_keywords"] else ""
            print(f"    - \"{row['post_text'][:60]}...\" "
                  f"(sentiment: {row['sentiment']}, conf: {row['sentiment_confidence']}){kw_note}")

    print("\n-----------------------------------\n")


def save_sentiment_plot(df: pd.DataFrame):
    """Saves a sentiment distribution chart with distress flags highlighted."""
    ensure_directories()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Sentiment distribution ---
    ax1 = axes[0]
    sentiment_counts = df["sentiment"].value_counts()
    colours_map = {"positive": "#27AE60", "neutral": "#95A5A6", "negative": "#E74C3C"}
    colours = [colours_map.get(s, "#3498DB") for s in sentiment_counts.index]
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colours, edgecolor="white")
    ax1.set_title("Sentiment Distribution\nAcross Sample Social Posts", fontweight="bold")
    ax1.set_ylabel("Number of Posts")
    for i, v in enumerate(sentiment_counts.values):
        ax1.text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    # --- Plot 2: Distress flags by category ---
    ax2 = axes[1]
    if "expected_category" in df.columns:
        distress_by_cat = df.groupby("expected_category")["high_priority_distress"].sum().sort_values()
        ax2.barh(distress_by_cat.index, distress_by_cat.values, color="#E74C3C", edgecolor="white")
        ax2.set_title("Distress Signals by Topic Category", fontweight="bold")
        ax2.set_xlabel("Number of Flagged Posts")

    plt.suptitle(
        "ZeraMatumizi - Sentiment & Distress Signal Detection",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = os.path.join(REPORTS_PATH, "sentiment_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sentiment plot saved: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - Multilingual Sentiment & Distress Detection")
    print("=" * 60)

    ensure_directories()

    df = load_sample_posts()
    classifier = load_sentiment_model()
    df = run_sentiment_pipeline(df, classifier)

    print_sentiment_summary(df)
    save_sentiment_plot(df)

    print("Sentiment analysis complete!")