#!/usr/bin/env python3
"""
Fresh, truly unseen event-classification test set.

The historical holdout headlines were visible while the keyword rules were
written, so this tool builds a second test set from TODAY's live news that
nobody has seen, labelled by a human BEFORE the model is run.

Step 1 - collect (needs internet):
    python scripts/fresh_test_set.py collect --n 40
  Writes data/fresh_test_set.csv with an empty `label` column. Model
  predictions are deliberately NOT shown, so labelling is blind.

Step 2 - label: open the CSV and fill `label` with exactly one of
    Geopolitical | Macroeconomic | Credit Event | Merger/Acquisition |
    Product Launch | Earnings/Corporate | Other
  Optional: `tickers` (e.g. "AAPL;MSFT") for the S&P 100 names the
  headline is about. Leave rows you cannot decide on blank: they are skipped.

  Labelling guide
    Geopolitical        war, sanctions, tariffs, export controls, OPEC/oil supply, elections abroad
    Macroeconomic       central banks, rates, inflation, jobs, GDP, PMI, trade balance
    Credit Event        default, downgrade, bankruptcy, bank failure, solvency stress
    Merger/Acquisition  deal announced/agreed/in talks, takeover, stake purchase
    Product Launch      a new product/feature/model is released or unveiled
    Earnings/Corporate  results, guidance, layoffs, recalls, strikes, analyst ratings, CEO changes
    Other               market wrap, personal finance, crypto price chatter, lifestyle, no event

Step 3 - score (needs the models):
    python scripts/fresh_test_set.py score
  Writes data/fresh_test_results.json (accuracy, macro-F1, 95% CI, majority
  baseline, confusion matrix, per-headline predictions, ticker recall).
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVENT_CLASSES, data_path, output_path  # noqa: E402

CSV_NAME = "fresh_test_set.csv"
RESULTS_NAME = "fresh_test_results.json"
FIELDS = ["id", "collected_at", "source", "headline", "label", "tickers"]


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def macro_f1(truth: list[str], pred: list[str]) -> float:
    classes = sorted(set(truth))
    scores = []
    for cls in classes:
        tp = sum(1 for t, p in zip(truth, pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truth, pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truth, pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def summarise(rows: list[dict]) -> dict:
    """Metrics from rows carrying `label` and `predicted` (pure, unit-tested)."""
    truth = [r["label"] for r in rows]
    pred = [r["predicted"] for r in rows]
    n = len(rows)
    correct = sum(t == p for t, p in zip(truth, pred))
    counts = Counter(truth)
    majority_class, majority_n = counts.most_common(1)[0] if counts else ("", 0)
    confusion = {
        t: {p: sum(1 for a, b in zip(truth, pred) if a == t and b == p) for p in EVENT_CLASSES}
        for t in EVENT_CLASSES if t in counts
    }
    actionable = [r for r in rows if r["label"] != "Other"]
    actionable_correct = sum(r["label"] == r["predicted"] for r in actionable)

    ticker_hits = ticker_total = 0
    for r in rows:
        expected = {t.strip().upper() for t in (r.get("tickers") or "").replace(",", ";").split(";") if t.strip()}
        if expected:
            ticker_total += len(expected)
            ticker_hits += len(expected & set(r.get("predicted_tickers", [])))

    return {
        "n_labelled": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "accuracy_95ci": wilson_interval(correct, n),
        "macro_f1": macro_f1(truth, pred),
        "majority_class": majority_class,
        "majority_baseline_accuracy": round(majority_n / n, 4) if n else 0.0,
        "accuracy_excluding_other": round(actionable_correct / len(actionable), 4) if actionable else None,
        "n_excluding_other": len(actionable),
        "label_distribution": dict(counts),
        "confusion_matrix": confusion,
        "ticker_recall": round(ticker_hits / ticker_total, 4) if ticker_total else None,
        "ticker_labels": ticker_total,
    }


def collect(n: int, seed: int) -> None:
    from src.ingestion.rss_loader import fetch_rss_articles

    path = data_path(CSV_NAME)
    if path.exists():
        print(f"{path} already exists. Delete it first if you really want a new sample.")
        return
    articles = fetch_rss_articles(max_per_feed=30)
    if not articles:
        print("No RSS feed reachable; cannot collect fresh headlines.")
        return
    # Balance across sources so one feed does not dominate.
    by_source: dict[str, list[dict]] = {}
    for a in articles:
        by_source.setdefault(a["source"], []).append(a)
    rng = random.Random(seed)
    for items in by_source.values():
        rng.shuffle(items)
    picked = []
    while len(picked) < n and any(by_source.values()):
        for source in list(by_source):
            if by_source[source] and len(picked) < n:
                picked.append(by_source[source].pop())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for i, a in enumerate(picked, 1):
            headline = a["text"].split(". ")[0][:300]  # title only: what a human labels at a glance
            writer.writerow({"id": f"fresh_{i:03d}", "collected_at": now, "source": a["source"],
                             "headline": headline, "label": "", "tickers": ""})
    print(f"Wrote {len(picked)} headlines to {path}")
    print("Now fill the `label` column (see the guide at the top of this script), then run: "
          "python scripts/fresh_test_set.py score")


def score() -> None:
    path = data_path(CSV_NAME)
    if not path.exists():
        print("Run `collect` first.")
        return
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    bad = [r for r in rows if r["label"].strip() and r["label"].strip() not in EVENT_CLASSES]
    if bad:
        for r in bad:
            print(f"Invalid label {r['label']!r} on {r['id']}. Use one of: {EVENT_CLASSES}")
        sys.exit(1)
    labelled = [r for r in rows if r["label"].strip()]
    if not labelled:
        print("No labelled rows yet.")
        return

    from src.engine.event_classifier import classify_event
    from src.engine.ner_linker import extract_tickers

    for r in labelled:
        r["label"] = r["label"].strip()
        result = classify_event(r["headline"])
        r["predicted"] = result["event_class"]
        r["method"] = result["method"]
        r["confidence"] = result["confidence"]
        r["predicted_tickers"] = extract_tickers(r["headline"])

    summary = summarise(labelled)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Headlines collected live, labelled by a human before any model output was "
                    "viewed; keyword rules and thresholds were frozen before collection.",
        "summary": summary,
        "rows": [{k: r.get(k) for k in ("id", "source", "headline", "label", "predicted", "method",
                                          "confidence", "tickers", "predicted_tickers")} for r in labelled],
    }
    output_path(RESULTS_NAME).write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nMisclassified:")
    for r in labelled:
        if r["label"] != r["predicted"]:
            print(f"  [{r['label']} -> {r['predicted']} via {r['method']}] {r['headline'][:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--n", type=int, default=40)
    c.add_argument("--seed", type=int, default=7)
    sub.add_parser("score")
    args = parser.parse_args()
    collect(args.n, args.seed) if args.command == "collect" else score()
