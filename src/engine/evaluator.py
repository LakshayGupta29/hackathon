"""
Evaluator
Runs sentiment models on the locked test set and reports:
  - Accuracy
  - Macro F1 (primary metric due to class imbalance)
  - Per-class F1
  - Confusion matrix

Usage (always run from project root):
  python3 -m src.engine.evaluator --mode vader
  python3 -m src.engine.evaluator --mode finbert
  python3 -m src.engine.evaluator --mode all
"""

import sys
import os
from pathlib import Path

# Always resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
from src.engine.sentiment import analyze

INT_TO_LABEL = {0: "negative", 1: "positive", 2: "neutral"}


def evaluate_model(mode: str, test_df: pd.DataFrame) -> dict:
    """Run one model over the full test set, return metrics."""
    print(f"\nEvaluating {mode}...")
    true_labels, pred_labels = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_label = INT_TO_LABEL[row["label"]]
        result = analyze(str(row["text"]), mode=mode)
        true_labels.append(true_label)
        pred_labels.append(result["label"])

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    print(f"\n{'='*40}")
    print(f"Model:     {mode.upper()}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Macro F1:  {macro_f1:.3f}  <- primary metric")
    print(f"\nPer-class report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["negative", "positive", "neutral"]
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(
        true_labels, pred_labels,
        labels=["negative", "positive", "neutral"]
    ))
    return {
        "mode": mode,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
    }


def print_summary(results: list):
    print(f"\n{'='*40}")
    print("SUMMARY — Macro F1 is primary metric")
    print(f"{'='*40}")
    print(f"{'Model':<12} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 35)
    for r in results:
        print(f"{r['mode']:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")
    print(f"{'='*40}")
    print("Note: Macro F1 used due to class imbalance (65% neutral)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="all",
        choices=["vader", "finbert", "finetuned", "all"]
    )
    args = parser.parse_args()

    test_df = pd.read_csv("data/test_set.csv")
    print(f"Test set: {len(test_df)} rows")

    modes = ["vader", "finbert"] if args.mode == "all" else [args.mode]
    results = [evaluate_model(m, test_df) for m in modes]
    print_summary(results)
