"""
Sentiment Analysis Module
Three modes, identical interface:
  - vader:     rule-based, no GPU, instant (baseline 1)
  - finbert:   ProsusAI/finbert pretrained (baseline 2)
  - finetuned: our fine-tuned model from HuggingFace Hub

All modes return:
  {
    "label":  "positive" | "negative" | "neutral",
    "score":  float in [-1.0, 1.0],
    "mode":   str,
    "raw":    dict  (model-specific raw output)
  }
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from functools import lru_cache
import os

# Your fine-tuned model on HuggingFace Hub (set after Day 3)
FINETUNED_MODEL = os.getenv("HF_FINETUNED_MODEL", "ProsusAI/finbert")


@lru_cache(maxsize=1)
def _load_vader():
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _load_finbert():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,        # return all class scores
        device=-1          # CPU; change to 0 for GPU
    )


@lru_cache(maxsize=1)
def _load_finetuned():
    return pipeline(
        "text-classification",
        model=FINETUNED_MODEL,
        tokenizer=FINETUNED_MODEL,
        top_k=None,
        device=-1
    )


def _vader_analyze(text: str) -> dict:
    sid = _load_vader()
    scores = sid.polarity_scores(text)
    compound = scores["compound"]   # -1.0 to +1.0

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": round(compound, 4), "raw": scores}


def _finbert_analyze(text: str, loader_fn) -> dict:
    """Shared logic for finbert and finetuned modes."""
    pipe = loader_fn()
    # Truncate to 512 tokens max (FinBERT limit)
    result = pipe(text[:512])[0]

    # result is list of {"label": ..., "score": ...}
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    best = max(result, key=lambda x: x["score"])
    label = best["label"].lower()

    # Build signed score: positive confidence positive, negative confidence negative
    score = best["score"] * label_map.get(label, 0.0)

    return {
        "label": label,
        "score": round(score, 4),
        "raw": {r["label"]: round(r["score"], 4) for r in result}
    }


def analyze(text: str, mode: str = "finbert") -> dict:
    """
    Main function.
    Args:
        text: input string
        mode: "vader" | "finbert" | "finetuned"
    Returns:
        dict with label, score, mode, raw
    """
    if mode == "vader":
        result = _vader_analyze(text)
    elif mode == "finbert":
        result = _finbert_analyze(text, _load_finbert)
    elif mode == "finetuned":
        result = _finbert_analyze(text, _load_finetuned)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use vader/finbert/finetuned")

    result["mode"] = mode
    return result


if __name__ == "__main__":
    tests = [
        "NVIDIA earnings crushed expectations, stock surges 12 percent",
        "Boeing faces fresh safety probe, shares fall sharply",
        "Fed holds rates steady, no change to guidance",
        "$TSLA deliveries miss estimates again this quarter",
        "JPMorgan warns of rising credit defaults in leveraged loans",
    ]

    for mode in ["vader", "finbert"]:
        print(f"\n--- {mode.upper()} ---")
        for text in tests:
            r = analyze(text, mode=mode)
            bar = "+" if r["label"] == "positive" else ("-" if r["label"] == "negative" else "=")
            print(f"  [{bar}] {r['score']:+.3f} {r['label']:<10} | {text[:55]}")
