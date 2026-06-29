"""
swahili_ner_model.py
Fine-tunes multilingual BERT for Swahili SUD Named Entity Recognition.

Model: bert-base-multilingual-cased
Task: Token classification (NER)
Entities: SUBSTANCE, RISK_FACTOR, SEVERITY, GEOGRAPHIC, TREATMENT

This is the first open-source Swahili NER model for substance use
disorder surveillance in East Africa.
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import torch

# Paths
RAW_DATA_PATH = os.path.join("data", "raw")
MODEL_PATH = os.path.join("data", "processed", "swahili_ner_model")
REPORTS_PATH = os.path.join("docs", "reports")

# Label scheme (BIO tagging)
LABELS = [
    "O",            # Outside any entity
    "B-SUBSTANCE",  # Beginning of substance entity
    "I-SUBSTANCE",  # Inside substance entity
    "B-RISK_FACTOR",
    "I-RISK_FACTOR",
    "B-SEVERITY",
    "I-SEVERITY",
    "B-GEOGRAPHIC",
    "I-GEOGRAPHIC",
    "B-TREATMENT",
    "I-TREATMENT",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Model name
MODEL_NAME = "bert-base-multilingual-cased"


def load_training_data():
    """Loads the annotated Swahili NER training data."""
    filepath = os.path.join(RAW_DATA_PATH, "swahili_ner_training.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} annotated sentences")
    return data


def convert_to_bio_tags(text: str, entities: list) -> tuple:
    """
    Converts character-level entity spans to word-level BIO tags.

    BIO tagging:
    B- = Beginning of entity
    I- = Inside entity
    O  = Outside any entity
    """
    words = text.split()
    word_tags = []
    char_pos = 0

    for word in words:
        word_start = text.find(word, char_pos)
        word_end = word_start + len(word)

        # Find which entity (if any) this word belongs to
        tag = "O"
        for ent_start, ent_end, ent_label in entities:
            if word_start >= ent_start and word_end <= ent_end:
                if word_start == ent_start:
                    tag = f"B-{ent_label}"
                else:
                    tag = f"I-{ent_label}"
                break

        word_tags.append(tag)
        char_pos = word_end

    return words, word_tags


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes words and aligns BIO labels with subword tokens.
    BERT splits words into subwords — labels must align correctly.
    """
    tokenized = tokenizer(
        examples["words"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens [CLS], [SEP], [PAD]
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                # First subword of a word — use the word's label
                aligned_labels.append(labels[word_id])
            else:
                # Subsequent subwords — use I- version or -100
                label = labels[word_id]
                if label % 2 == 1:  # B- label
                    aligned_labels.append(label + 1)  # Convert to I-
                else:
                    aligned_labels.append(label)
            prev_word_id = word_id

        all_labels.append(aligned_labels)

    tokenized["labels"] = all_labels
    return tokenized


def prepare_dataset(training_data, tokenizer):
    """
    Prepares the HuggingFace Dataset from annotated training data.
    Splits into train (80%) and test (20%).
    """
    all_words = []
    all_tags = []

    for item in training_data:
        words, tags = convert_to_bio_tags(
            item["text"], item["entities"]
        )
        tag_ids = [LABEL2ID.get(t, 0) for t in tags]
        all_words.append(words)
        all_tags.append(tag_ids)

    dataset = Dataset.from_dict({
        "words": all_words,
        "ner_tags": all_tags,
    })

    # Split train/test
    split = dataset.train_test_split(test_size=0.2, seed=42)

    # Tokenize
    tokenized = split.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=["words", "ner_tags"]
    )

    print(f"✓ Dataset prepared:")
    print(f"  Train: {len(tokenized['train'])} sentences")
    print(f"  Test:  {len(tokenized['test'])} sentences")

    return tokenized


def compute_metrics(eval_pred):
    """
    Computes token-level precision, recall, and F1 score.
    Ignores special tokens (label = -100).
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_labels.append(label)
                true_predictions.append(pred)

    true_labels = np.array(true_labels)
    true_predictions = np.array(true_predictions)

    # Exclude O label (index 0) for entity-level metrics
    entity_mask = true_labels != 0
    if entity_mask.sum() == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    correct = (
        (true_predictions == true_labels) & entity_mask
    ).sum()
    predicted_entities = (true_predictions != 0).sum()
    true_entities = entity_mask.sum()

    precision = correct / predicted_entities if predicted_entities > 0 else 0
    recall = correct / true_entities if true_entities > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "f1":        round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
    }


def train_ner_model(tokenized_dataset, tokenizer):
    """
    Fine-tunes multilingual BERT for Swahili NER.
    Uses a small number of epochs suitable for our dataset size.
    """
    os.makedirs(MODEL_PATH, exist_ok=True)

    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning multilingual BERT on Swahili SUD data...")
    print("(This takes 3-5 minutes on CPU)")
    trainer.train()

    print(f"\n✓ Model trained and saved to: {MODEL_PATH}")
    return trainer, model


def evaluate_model(trainer):
    """Evaluates the model on the test set."""
    print("\nEvaluating on test set...")
    results = trainer.evaluate()

    print("\n--- NER Model Evaluation Results ---")
    print(f"  F1 Score:   {results.get('eval_f1', 0):.4f}")
    print(f"  Precision:  {results.get('eval_precision', 0):.4f}")
    print(f"  Recall:     {results.get('eval_recall', 0):.4f}")
    print(f"  Loss:       {results.get('eval_loss', 0):.4f}")

    f1 = results.get("eval_f1", 0)
    if f1 >= 0.70:
        print(f"\n  ✓ Model meets target F1 ≥ 0.70 — ready for deployment!")
    elif f1 >= 0.50:
        print(f"\n  ⚠ Model F1 = {f1:.2f} — acceptable for small dataset")
        print(f"    With 2,000+ annotated sentences F1 > 0.80 expected")
    else:
        print(f"\n  ℹ Model F1 = {f1:.2f} — more training data needed")
        print(f"    This is expected with 30 training sentences")
        print(f"    In production: 2,000 annotated sentences → F1 > 0.80")

    return results


def run_inference(model, tokenizer, text: str) -> list:
    """
    Runs NER inference on a new Swahili sentence.
    Returns a list of detected entities.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = [
        ID2LABEL[p.item()] for p in predictions[0]
    ]

    # Extract entities
    entities = []
    current_entity = None
    current_tokens = []

    for token, label in zip(tokens, predicted_labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if label.startswith("B-"):
            if current_entity:
                entities.append({
                    "text": tokenizer.convert_tokens_to_string(current_tokens),
                    "label": current_entity
                })
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append({
                    "text": tokenizer.convert_tokens_to_string(current_tokens),
                    "label": current_entity
                })
                current_entity = None
                current_tokens = []

    return entities


def demo_inference(model, tokenizer):
    """Demonstrates NER on sample Swahili sentences."""
    test_sentences = [
        "Kijana wa Kisumu anatumia chang'aa na anahitaji ukarabati.",
        "Ukosefu wa kazi Kakamega umesababisha matumizi ya busaa.",
        "Mwanafunzi huyu ni tegemezi wa bhang na anahitaji ushauri nasaha.",
        "Homa Bay ina tatizo kubwa la miraa kati ya vijana.",
    ]

    print("\n--- NER Inference Demo ---")
    for sentence in test_sentences:
        entities = run_inference(model, tokenizer, sentence)
        print(f"\n  Text: {sentence}")
        if entities:
            print(f"  Entities detected:")
            for ent in entities:
                print(f"    [{ent['label']}] → '{ent['text']}'")
        else:
            print(f"  No entities detected")
    print("--------------------------\n")


if __name__ == "__main__":
    print("=" * 55)
    print("ZeraMatumizi — Swahili SUD NER Model Training")
    print("=" * 55)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✓ Tokenizer loaded")

    # Load and prepare data
    training_data = load_training_data()
    tokenized_dataset = prepare_dataset(training_data, tokenizer)

    # Train model
    trainer, model = train_ner_model(tokenized_dataset, tokenizer)

    # Evaluate
    results = evaluate_model(trainer)

    # Demo inference
    demo_inference(model, tokenizer)

    print("✓ Swahili NER model training complete!")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Next step: Upload to HuggingFace Hub as")
    print(f"  'brian-omare/swahili-sud-ner'")