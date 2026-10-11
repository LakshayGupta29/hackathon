"""
Sentiment Analysis Module.

Three modes share one interface:
  - vader:     rule-based baseline
  - finbert:   pretrained ProsusAI/FinBERT baseline
  - finetuned: project fine-tuned FinBERT model on Hugging Face

Every mode returns:
  {"label": positive|negative|neutral, "score": float in [-1, 1],
   "probabilities": {...}, "mode": str, "model": str}

For transformer modes the score is P(positive) - P(negative). This is
continuous and symmetric: a confident neutral prediction scores ~0, and a
weak positive/negative lean is preserved instead of being collapsed.
"""

from functools import lru_cache
import logging
import os
import threading

log = logging.getLogger(__name__)

BASE_FINBERT = "ProsusAI/finbert"
FINETUNED_MODEL = os.getenv(
    "HF_FINETUNED_MODEL",
    "LakshG299/finbert-financial-sentiment-v2",
)

VALID_LABELS = ("positive", "negative", "neutral")
LABEL_ALIASES = {
    "bullish": "positive",
    "bearish": "negative",
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
}
# Native ProsusAI/FinBERT ids, used only if a model ships without id2label.
NATIVE_ID_LABELS = {"0": "positive", "1": "negative", "2": "neutral"}

_finetuned_status = {"model": FINETUNED_MODEL, "fallback": False, "error": None}
_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:  # lexicon not downloaded yet
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _build_pipeline(model_name: str):
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
        device=-1,
    )


@lru_cache(maxsize=1)
def _load_finbert():
    return _build_pipeline(BASE_FINBERT)


@lru_cache(maxsize=1)
def _load_finetuned():
    """Load the fine-tuned model; fall back to base FinBERT so a demo never dies."""
    with _load_lock:
        try:
            return _build_pipeline(FINETUNED_MODEL)
        except Exception as exc:  # network, auth, missing repo...
            log.warning(
                "Could not load fine-tuned model %s (%s). Falling back to %s.",
                FINETUNED_MODEL, exc, BASE_FINBERT,
            )
            _finetuned_status.update(
                {"model": BASE_FINBERT, "fallback": True, "error": str(exc)}
            )
            return _load_finbert()


def model_status() -> dict:
    """Which model the 'finetuned' mode is actually serving."""
    return dict(_finetuned_status)


def _normalize_model_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized.startswith("label_"):
        normalized = NATIVE_ID_LABELS.get(normalized.removeprefix("label_"), normalized)
    normalized = LABEL_ALIASES.get(normalized, normalized)
    if normalized not in VALID_LABELS:
        raise ValueError(
            f"Unexpected sentiment label '{label}'. "
            "Model must expose positive/negative/neutral labels."
        )
    return normalized


def _as_score_list(output) -> list[dict]:
    """Normalize pipeline output for one input to a flat list of {label, score}."""
    if isinstance(output, dict):
        return [output]
    if output and isinstance(output[0], list):
        return output[0]
    return list(output)


def _vader_analyze(text: str) -> dict:
    scores = _load_vader().polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    probabilities = {
        "positive": round(scores.get("pos", 0.0), 4),
        "negative": round(scores.get("neg", 0.0), 4),
        "neutral": round(scores.get("neu", 0.0), 4),
    }
    return {
        "label": label,
        "score": round(compound, 4),
        "probabilities": probabilities,
        "model": "vader",
    }


def _transformer_analyze(text: str, loader_fn, model_name: str) -> dict:
    raw = loader_fn()([text], truncation=True, max_length=512)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        entries = raw  # flat list of {label, score} for the single input
    else:
        entries = _as_score_list(raw[0])

    probabilities = {label: 0.0 for label in VALID_LABELS}
    for entry in entries:
        probabilities[_normalize_model_label(entry["label"])] += float(entry["score"])

    label = max(probabilities, key=probabilities.get)
    score = probabilities["positive"] - probabilities["negative"]
    return {
        "label": label,
        "score": round(max(-1.0, min(1.0, score)), 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "model": model_name,
    }


def analyze_batch(texts: list[str], mode: str = "finetuned", batch_size: int = 32) -> list[dict]:
    """Score many texts efficiently (transformer modes run in mini-batches)."""
    texts = [str(t or "") for t in texts]
    if mode == "vader":
        return [analyze(t, mode="vader") for t in texts]
    if mode == "finetuned":
        pipe, model_name = _load_finetuned(), _finetuned_status["model"]
    elif mode == "finbert":
        pipe, model_name = _load_finbert(), BASE_FINBERT
    else:
        raise ValueError(f"Unknown mode: {mode}")
    results = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        outputs = pipe(chunk, truncation=True, max_length=128, batch_size=batch_size)
        for entries in outputs:
            entries = _as_score_list(entries)
            probabilities = {label: 0.0 for label in VALID_LABELS}
            for entry in entries:
                probabilities[_normalize_model_label(entry["label"])] += float(entry["score"])
            score = probabilities["positive"] - probabilities["negative"]
            results.append({
                "label": max(probabilities, key=probabilities.get),
                "score": round(max(-1.0, min(1.0, score)), 4),
                "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                "model": model_name,
                "mode": mode,
            })
    return results


def analyze(text: str, mode: str = "finbert") -> dict:
    text = str(text or "")
    if mode == "vader":
        result = _vader_analyze(text)
    elif mode == "finbert":
        result = _transformer_analyze(text, _load_finbert, BASE_FINBERT)
    elif mode == "finetuned":
        pipe = _load_finetuned()
        result = _transformer_analyze(text, lambda: pipe, _finetuned_status["model"])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use vader/finbert/finetuned")
    result["mode"] = mode
    # Backwards-compatible alias used by older callers.
    result["raw"] = result["probabilities"]
    return result
