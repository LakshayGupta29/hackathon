"""Evaluate sentiment models on the untouched locked test set."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from src.config import output_path
from src.engine.sentiment import analyze, model_status

INT_TO_LABEL = {0: "negative", 1: "positive", 2: "neutral"}


def evaluate_model(mode: str, test_df: pd.DataFrame) -> dict:
    """Run one model over the locked test set."""
    print(f"\nEvaluating {mode}...")
    true_labels, pred_labels = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_label = INT_TO_LABEL[int(row["label"])]
        result = analyze(str(row["text"]), mode=mode)
        true_labels.append(true_label)
        pred_labels.append(result["label"])

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    print(f"\n{'=' * 40}")
    print(f"Model:     {mode.upper()}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Macro F1:  {macro_f1:.3f}  <- primary metric")
    print("\nPer-class report:")
    print(classification_report(
        true_labels,
        pred_labels,
        labels=["negative", "positive", "neutral"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(
        true_labels,
        pred_labels,
        labels=["negative", "positive", "neutral"],
    ))
    labels = ["negative", "positive", "neutral"]
    report = classification_report(
        true_labels, pred_labels, labels=labels, zero_division=0, output_dict=True
    )
    return {
        "mode": mode,
        "model": model_status()["model"] if mode == "finetuned" else mode,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            label: {k: round(float(v), 4) for k, v in report[label].items()}
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(true_labels, pred_labels, labels=labels).tolist(),
        },
    }


def print_summary(results: list[dict], test_df: pd.DataFrame) -> None:
    majority_accuracy = float(test_df["label"].value_counts(normalize=True).max())
    print(f"\n{'=' * 55}")
    print("SUMMARY — Macro F1 is primary metric")
    print(f"Majority-class accuracy baseline: {majority_accuracy:.3f}")
    print(f"{'=' * 55}")
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 35)
    for result in results:
        print(
            f"{result['mode']:<12} "
            f"{result['accuracy']:>10.3f} "
            f"{result['macro_f1']:>10.3f}"
        )
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="all",
        choices=["vader", "finbert", "finetuned", "all"],
    )
    parser.add_argument(
        "--test-path",
        default="data/test_set.csv",
        help="Locked test CSV; must not be used for model selection.",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_path)
    print(f"Test set: {len(test_df)} rows")

    modes = ["vader", "finbert", "finetuned"] if args.mode == "all" else [args.mode]
    results = [evaluate_model(mode, test_df) for mode in modes]
    print_summary(results, test_df)

    payload = {
        "test_path": args.test_path,
        "test_rows": len(test_df),
        "majority_class_accuracy": round(float(test_df["label"].value_counts(normalize=True).max()), 4),
        "results": {r["mode"]: r for r in results},
    }
    out = output_path("eval_results.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out}")
