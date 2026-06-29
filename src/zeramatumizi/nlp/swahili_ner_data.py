"""
swahili_ner_data.py
Creates annotated training data for the Swahili SUD NER model.

Entity types:
- SUBSTANCE: chang'aa, busaa, bhang, miraa, kuber, uji wa pombe
- RISK_FACTOR: ukosefu wa kazi, shinikizo la marafiki
- SEVERITY: tegemezi, matumizi mabaya
- GEOGRAPHIC: county and location names
- TREATMENT: rehabilitation, ushauri nasaha, dawa
"""

import json
import os

REPORTS_PATH = os.path.join("data", "raw")


def create_training_data():
    """
    Creates annotated Swahili NER training sentences.
    Each entry has text and a list of entity spans.

    Format: {"text": "...", "entities": [(start, end, label)]}
    """
    training_data = [
        # SUBSTANCE entities
        {
            "text": "Kijana huyu anatumia chang'aa kila siku.",
            "entities": [(19, 27, "SUBSTANCE")],
            "translation": "This youth uses chang'aa every day."
        },
        {
            "text": "Wanaume wengi wanakunywa busaa karibu na soko.",
            "entities": [(24, 29, "SUBSTANCE")],
            "translation": "Many men drink busaa near the market."
        },
        {
            "text": "Watoto wa shule wanapatikana wakivuta bhang mtaani.",
            "entities": [(37, 42, "SUBSTANCE")],
            "translation": "School children are found smoking bhang in the streets."
        },
        {
            "text": "Matumizi ya miraa yameenea sana Meru na Nyanza.",
            "entities": [(11, 16, "SUBSTANCE")],
            "translation": "Use of miraa is very widespread in Meru and Nyanza."
        },
        {
            "text": "Wafanyabiashara wengine wanauza kuber kwa vijana.",
            "entities": [(31, 36, "SUBSTANCE")],
            "translation": "Some traders sell kuber to youth."
        },
        {
            "text": "Uji wa pombe unatengenezwa nyumbani kwa siri.",
            "entities": [(0, 11, "SUBSTANCE")],
            "translation": "Fermented porridge brew is made at home secretly."
        },
        {
            "text": "Alianza kutumia pombe akiwa na miaka kumi na tano.",
            "entities": [(16, 21, "SUBSTANCE")],
            "translation": "He started using alcohol at fifteen years old."
        },
        {
            "text": "Dawa za kulevya zinaathiri vijana wengi Kisumu.",
            "entities": [(0, 16, "SUBSTANCE")],
            "translation": "Drugs affect many youth in Kisumu."
        },

        # RISK_FACTOR entities
        {
            "text": "Ukosefu wa kazi umesababisha vijana wengi kuanza kunywa.",
            "entities": [(0, 12, "RISK_FACTOR")],
            "translation": "Unemployment has caused many youth to start drinking."
        },
        {
            "text": "Shinikizo la marafiki ndilo chanzo kikuu cha matumizi.",
            "entities": [(0, 20, "RISK_FACTOR")],
            "translation": "Peer pressure is the main cause of substance use."
        },
        {
            "text": "Umaskini na ukosefu wa kazi vinachangia matumizi ya pombe.",
            "entities": [
                (0, 8, "RISK_FACTOR"),
                (12, 24, "RISK_FACTOR")
            ],
            "translation": "Poverty and unemployment contribute to alcohol use."
        },
        {
            "text": "Familia iliyovunjika mara nyingi husababisha ulevi.",
            "entities": [(0, 20, "RISK_FACTOR")],
            "translation": "A broken family often leads to alcoholism."
        },
        {
            "text": "Kuacha shule mapema ni hatari kubwa ya matumizi ya dawa.",
            "entities": [(0, 13, "RISK_FACTOR")],
            "translation": "Early school dropout is a major risk for drug use."
        },

        # SEVERITY entities
        {
            "text": "Mtu huyu ni tegemezi wa pombe na anahitaji msaada.",
            "entities": [(12, 20, "SEVERITY")],
            "translation": "This person is alcohol dependent and needs help."
        },
        {
            "text": "Matumizi mabaya ya dawa yanaathiri afya yake.",
            "entities": [(0, 16, "SEVERITY")],
            "translation": "Substance abuse is affecting his health."
        },
        {
            "text": "Amekuwa mlevi wa kudumu kwa miaka mitano.",
            "entities": [(10, 15, "SEVERITY")],
            "translation": "He has been a chronic alcoholic for five years."
        },
        {
            "text": "Hali yake ni mbaya sana — anahitaji matibabu ya haraka.",
            "entities": [(10, 18, "SEVERITY")],
            "translation": "His condition is very serious — he needs urgent treatment."
        },

        # GEOGRAPHIC entities
        {
            "text": "Tatizo la dawa ni kubwa sana Kisumu na Kakamega.",
            "entities": [
                (29, 35, "GEOGRAPHIC"),
                (39, 47, "GEOGRAPHIC")
            ],
            "translation": "The drug problem is very serious in Kisumu and Kakamega."
        },
        {
            "text": "Wilaya ya Siaya ina idadi kubwa ya walevi.",
            "entities": [(10, 15, "GEOGRAPHIC")],
            "translation": "Siaya sub-county has a large number of alcoholics."
        },
        {
            "text": "Homa Bay na Migori zinakabiliwa na tatizo la miraa.",
            "entities": [
                (0, 8, "GEOGRAPHIC"),
                (12, 18, "GEOGRAPHIC")
            ],
            "translation": "Homa Bay and Migori face a miraa problem."
        },
        {
            "text": "Kaunti ya Nakuru ina vituo vichache vya ukarabati.",
            "entities": [(10, 16, "GEOGRAPHIC")],
            "translation": "Nakuru county has few rehabilitation centres."
        },

        # TREATMENT entities
        {
            "text": "Anahitaji kwenda katika kituo cha ukarabati haraka.",
            "entities": [(33, 42, "TREATMENT")],
            "translation": "He needs to go to a rehabilitation centre urgently."
        },
        {
            "text": "Ushauri nasaha unaweza kumsaidia kupona.",
            "entities": [(0, 14, "TREATMENT")],
            "translation": "Counselling can help him recover."
        },
        {
            "text": "Dawa za matibabu zinapatikana hospitalini kwa walevi.",
            "entities": [(0, 17, "TREATMENT")],
            "translation": "Medical treatment is available at hospitals for alcoholics."
        },
        {
            "text": "NACADA inatoa huduma za msaada kwa wanaotaka kupona.",
            "entities": [(0, 6, "TREATMENT")],
            "translation": "NACADA provides support services for those who want to recover."
        },
        {
            "text": "Programu ya ukarabati imesaidia familia nyingi Nyanza.",
            "entities": [(11, 20, "TREATMENT")],
            "translation": "The rehabilitation programme has helped many families in Nyanza."
        },

        # Mixed entities
        {
            "text": "Vijana wa Kisumu wanaotumia bhang wanahitaji ushauri nasaha.",
            "entities": [
                (10, 16, "GEOGRAPHIC"),
                (28, 33, "SUBSTANCE"),
                (44, 58, "TREATMENT")
            ],
            "translation": "Youth in Kisumu who use bhang need counselling."
        },
        {
            "text": "Ukosefu wa kazi Kakamega umesababisha matumizi ya chang'aa.",
            "entities": [
                (0, 12, "RISK_FACTOR"),
                (13, 21, "GEOGRAPHIC"),
                (49, 57, "SUBSTANCE")
            ],
            "translation": "Unemployment in Kakamega has caused chang'aa use."
        },
        {
            "text": "Tegemezi wa pombe Siaya anahitaji ukarabati wa dharura.",
            "entities": [
                (0, 8, "SEVERITY"),
                (18, 23, "GEOGRAPHIC"),
                (33, 42, "TREATMENT")
            ],
            "translation": "An alcohol dependent person in Siaya needs emergency rehabilitation."
        },
        {
            "text": "Shinikizo la marafiki limepelekea vijana Homa Bay kutumia busaa.",
            "entities": [
                (0, 20, "RISK_FACTOR"),
                (41, 49, "GEOGRAPHIC"),
                (57, 62, "SUBSTANCE")
            ],
            "translation": "Peer pressure has led youth in Homa Bay to use busaa."
        },
    ]

    return training_data


def save_training_data(training_data):
    """Saves training data to JSON file."""
    os.makedirs(REPORTS_PATH, exist_ok=True)
    filepath = os.path.join(REPORTS_PATH, "swahili_ner_training.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Training data saved: {filepath}")
    print(f"  Total sentences: {len(training_data)}")

    # Count entities by type
    entity_counts = {}
    for item in training_data:
        for _, _, label in item["entities"]:
            entity_counts[label] = entity_counts.get(label, 0) + 1

    print(f"\n  Entity distribution:")
    for label, count in sorted(entity_counts.items()):
        print(f"    {label:<15}: {count} entities")

    return filepath


def print_sample_annotations(training_data, n=3):
    """Prints sample annotated sentences."""
    print(f"\n--- Sample Annotated Sentences ---")
    for item in training_data[:n]:
        text = item["text"]
        print(f"\n  Text: {text}")
        print(f"  Translation: {item['translation']}")
        print(f"  Entities:")
        for start, end, label in item["entities"]:
            entity_text = text[start:end]
            print(f"    [{label}] '{entity_text}' (chars {start}-{end})")
    print("----------------------------------\n")


if __name__ == "__main__":
    print("Creating Swahili NER training data...")
    training_data = create_training_data()
    print_sample_annotations(training_data)
    save_training_data(training_data)
    print("\n✓ Training data creation complete!")