"""
topic_modelling.py
BERTopic-based topic modelling for ZeraMatumizi.

Discovers emerging substance use trends from social media-style
discourse (Twitter/X, Facebook community groups) and clinical notes -
identifying new psychoactive substances or shifting patterns months
before they appear in clinical data.

In production this connects to Twitter/X Academic API and Meta
CrowdTangle. For development we use representative sample posts
covering the substance landscape described in NACADA/UNODC reports.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

REPORTS_PATH = os.path.join("docs", "reports")
RAW_DATA_PATH = os.path.join("data", "raw")


def ensure_directories():
    os.makedirs(REPORTS_PATH, exist_ok=True)
    os.makedirs(RAW_DATA_PATH, exist_ok=True)


def create_sample_social_posts() -> pd.DataFrame:
    """
    Creates a representative sample of Kenyan social-media-style posts
    discussing substance use, mirroring the kind of discourse found on
    Twitter/X Kenya and Facebook community groups.

    In production, replace this with real scraped data via the
    Twitter/X Academic API.
    """
    posts = [
        # Chang'aa / illicit brew cluster
        "Watu wengi wamekufa Kakamega kwa kunywa changaa ya sumu",
        "Police wamekamata watu wakitengeneza changaa karibu na shule",
        "Chang'aa imekuwa shida kubwa Western Kenya, vijana wanaharibika",
        "Another illicit brew poisoning case reported in Kakamega county",
        "NACADA warns of toxic changaa batch circulating in Western region",
        "Watu wamelazwa hospitali baada ya kunywa changaa ya bandia",

        # Bhang / cannabis cluster
        "Vijana wengi Nairobi wanavuta bhang waziwazi sasa hivi",
        "Bhang use rising among university students in Nairobi",
        "Cannabis arrests up 40% in Nyanza region this year says police",
        "Watoto wa shule wanapatikana na bhang mfukoni",
        "New cannabis strain reportedly stronger circulating in Mombasa",
        "Parents worried about bhang smell from estate near school",

        # Miraa / khat cluster
        "Miraa trade booming in Meru but addiction concerns growing",
        "Wafanyabiashara wa miraa wanapinga vikwazo vipya vya serikali",
        "Khat chewing linked to sleep problems among truck drivers",
        "Vijana wa Mandera wanatumia miraa sana wakati wa kazi",

        # Emerging synthetic substances
        "New synthetic drug 'kuber' spreading fast among teenagers",
        "Kuber sachets being sold near schools in Mombasa",
        "Parents alarmed by new nicotine pouch trend among teens",
        "Authorities investigating new synthetic substance in Coast region",

        # Prescription drug misuse
        "Reports of codeine cough syrup abuse rising in Nairobi estates",
        "Pharmacy raided for selling controlled drugs without prescription",
        "Watu wanatumia dawa za maumivu vibaya bila agizo la daktari",

        # Treatment/recovery related
        "NACADA rehabilitation centre in Kisumu now accepting new patients",
        "Inspiring recovery story from Homa Bay rehab graduate shared online",
        "Community counsellor praised for helping youth quit chang'aa",
        "Free counselling sessions available this week NACADA Nyanza office",

        # Distress signals
        "Nimechoka na maisha haya, sijui nifanye nini tena",
        "Feeling hopeless lately, can't seem to stop drinking",
        "Anyone know where to get help for a family member with addiction?",
        "Struggling badly today, the cravings won't stop",

        # General awareness/policy
        "Government launches new anti-drug campaign in five counties",
        "NACADA officers trained on new community intervention approach",
        "County government allocates funds for new rehabilitation centre",
        "Schools to introduce mandatory substance abuse awareness classes",
    ]

    # Assign rough categories for ground-truth comparison
    categories = (
        ["illicit_brew"] * 6 +
        ["cannabis"] * 6 +
        ["khat"] * 4 +
        ["synthetic"] * 4 +
        ["prescription"] * 3 +
        ["treatment"] * 4 +
        ["distress"] * 4 +
        ["policy"] * 4
    )

    df = pd.DataFrame({
        "post_text": posts,
        "expected_category": categories,
    })

    return df


def run_topic_modelling(df: pd.DataFrame):
    """
    Runs BERTopic on the sample social posts to discover topic clusters.
    Uses a multilingual embedding model since posts mix Swahili and English.
    """
    print("Loading multilingual embedding model...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Small dataset needs relaxed clustering parameters
    vectorizer_model = CountVectorizer(stop_words=None, min_df=1, ngram_range=(1, 2))

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=2,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False,
    )

    print("Fitting BERTopic on sample social posts...")
    topics, probs = topic_model.fit_transform(df["post_text"].tolist())

    df["topic"] = topics

    return topic_model, df


def print_topic_summary(topic_model: BERTopic, df: pd.DataFrame):
    """Prints a summary of discovered topics with top keywords."""
    print("\n--- Discovered Topics ---")
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            label = "Outliers / unclustered"
        else:
            keywords = topic_model.get_topic(topic_id)
            top_words = ", ".join([w for w, _ in keywords[:5]])
            label = f"Topic {topic_id}: {top_words}"

        count = row["Count"]
        print(f"\n  {label} ({count} posts)")

        # Show example posts in this topic
        examples = df[df["topic"] == topic_id]["post_text"].head(2).tolist()
        for ex in examples:
            print(f"    - {ex[:70]}{'...' if len(ex) > 70 else ''}")

    print("\n-------------------------\n")


def save_topic_distribution_plot(df: pd.DataFrame, topic_model: BERTopic):
    """Saves a bar chart showing posts per discovered topic."""
    ensure_directories()

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].sort_values("Count", ascending=True)

    labels = []
    for _, row in topic_info.iterrows():
        keywords = topic_model.get_topic(row["Topic"])
        top_words = "/".join([w for w, _ in keywords[:3]])
        labels.append(f"T{row['Topic']}: {top_words}")

    plt.figure(figsize=(10, max(4, len(labels) * 0.5)))
    plt.barh(labels, topic_info["Count"], color="#3498DB", edgecolor="white")
    plt.xlabel("Number of Posts")
    plt.title(
        "ZeraMatumizi - Emerging Substance Use Topics\nDiscovered via BERTopic on Social Media-Style Posts",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    output_path = os.path.join(REPORTS_PATH, "topic_modelling.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Topic distribution plot saved: {output_path}")


def save_sample_posts(df: pd.DataFrame):
    """Saves the sample social posts dataset for reuse by other modules."""
    ensure_directories()
    filepath = os.path.join(RAW_DATA_PATH, "social_media_sample.parquet")
    df.to_parquet(filepath, index=False)
    print(f"Sample social posts saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("ZeraMatumizi - Topic Modelling (BERTopic)")
    print("=" * 60)

    ensure_directories()

    df = create_sample_social_posts()
    print(f"\nLoaded {len(df)} sample social media-style posts")

    topic_model, df = run_topic_modelling(df)
    print_topic_summary(topic_model, df)
    save_topic_distribution_plot(df, topic_model)
    save_sample_posts(df)

    print("Topic modelling complete!")