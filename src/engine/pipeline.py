"""
Unified NLP risk pipeline.

text -> entity linking -> sentiment -> event classification -> embedding
     -> novelty / duplicate check -> risk-graph propagation -> impact score

Model inference is serialised with a lock: Hugging Face fast tokenizers are
not safe to call concurrently from several threads, and the API runs the
pipeline in worker threads so the event loop never blocks.
"""

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone

from src.config import PIPELINE_VERSION, output_path
from src.engine.embedder import DUPLICATE_THRESHOLD, embed
from src.engine.event_classifier import classify_event
from src.engine.event_fusion import fuse_events
from src.engine.impact_scorer import compute_impact
from src.engine.ner_linker import extract_tickers
from src.engine.novelty_scorer import NoveltyScorer
from src.engine.sentiment import analyze as analyze_sentiment
from src.risk.risk_graph import traverse

_novelty_scorer = NoveltyScorer()
_inference_lock = threading.Lock()

NEGATIVE_CUES = {
    "fall", "falls", "drop", "drops", "warn", "warns", "risk", "crisis",
    "default", "probe", "miss", "misses", "loss", "losses", "decline",
    "fear", "fears", "concern", "concerns", "cut", "cuts", "downgrade",
    "slump", "slumps", "plunge", "plunges", "tumble", "tumbles", "collapse",
}
POSITIVE_CUES = {
    "beat", "beats", "surge", "surges", "record", "growth", "strong",
    "exceed", "exceeds", "launch", "launches", "acquire", "profit", "rally",
    "gain", "gains", "rise", "rises", "upgrade", "soar", "soars", "jump",
}


def _get_triggering_sentence(text: str, sentiment_label: str) -> str:
    """Pick the sentence carrying the strongest sentiment cue (explanation)."""
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 20
    ]
    if not sentences:
        return text[:200]
    cues = NEGATIVE_CUES if sentiment_label == "negative" else POSITIVE_CUES
    best_sentence, best_score = sentences[0], -1
    for sentence in sentences:
        words = set(re.findall(r"[a-z]+", sentence.lower()))
        score = len(words & cues)
        if score > best_score:
            best_sentence, best_score = sentence, score
    return best_sentence[:300]


def process(text: str, source: str = "unknown", published_at: str | None = None) -> dict:
    """Process one headline/article into a machine-readable risk event."""
    start_time = time.time()
    text = str(text or "").strip()
    if not text:
        raise ValueError("text must be non-empty")

    with _inference_lock:
        tickers = extract_tickers(text)
        sentiment = analyze_sentiment(text, mode="finetuned")
        event = classify_event(text)
        embedding = embed(text)
        novelty = _novelty_scorer.score(text, embedding=embedding)

    graph = traverse(event["event_class"], tickers)
    impact = compute_impact(
        sentiment_score=sentiment["score"],
        event_class=event["event_class"],
        novelty_score=novelty["novelty_score"],
        affected_tickers=tickers,
        sources=[source],
    )

    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": now,
        "published_at": published_at or now,
        "headline": text[:500],
        "source": source,
        "sentiment_score": sentiment["score"],
        "sentiment_label": sentiment["label"],
        "sentiment_probabilities": sentiment.get("probabilities", {}),
        "sentiment_model": sentiment.get("model", ""),
        "event_class": event["event_class"],
        "event_confidence": event["confidence"],
        "event_method": event["method"],
        "event_low_confidence": bool(event.get("low_confidence", False)),
        "event_all_scores": event["all_scores"],
        "event_matched_rules": event.get("matched_rules", []),
        "novelty_score": novelty["novelty_score"],
        "novelty_label": novelty["novelty_label"],
        "max_similarity": novelty["max_similarity"],
        "most_similar_past": novelty["most_similar_past"],
        "is_duplicate": novelty["max_similarity"] >= DUPLICATE_THRESHOLD,
        "impact_score": impact["impact_score"],
        "impact_breakdown": impact["breakdown"],
        "impact_contributions": impact["contributions"],
        "impact_weights": impact["weights"],
        "affected_tickers": tickers,
        "affected_sectors": graph["affected_sectors"],
        "triggering_sentence": _get_triggering_sentence(text, sentiment["label"]),
        "pipeline_version": PIPELINE_VERSION,
        "latency_ms": round((time.time() - start_time) * 1000, 1),
    }


def process_batch(items: list, enable_fusion: bool = True) -> list:
    """Process a batch and optionally cluster semantically duplicate events."""
    events = [
        process(item["text"], item.get("source", "unknown"), item.get("published_at"))
        for item in items
    ]
    if enable_fusion and len(events) >= 2:
        return fuse_events(events)["events"]
    return events


def reset_novelty_memory() -> None:
    _novelty_scorer.reset()


def warmup() -> None:
    """Load every model once so the first live event is not slow."""
    sample = "Federal Reserve holds interest rates steady as inflation cools."
    with _inference_lock:
        extract_tickers(sample)
        analyze_sentiment(sample, mode="finetuned")
        classify_event("Company unveils quarterly update")  # forces zero-shot path
        embed(sample)


if __name__ == "__main__":
    test_items = [
        {
            "text": "Military exercises near Taiwan rattle semiconductor supply chains. TSMC and ASML shares fell sharply as geopolitical risk intensified.",
            "source": "news:reuters",
        },
        {
            "text": "Federal Reserve raises interest rates by 50 basis points amid persistent inflation concerns.",
            "source": "news:bloomberg",
        },
        {
            "text": "$NVDA absolutely crushing it this quarter, AI demand is insane",
            "source": "social:twitter",
        },
    ]
    reset_novelty_memory()
    results = process_batch(test_items)
    print(json.dumps(results, indent=2))
    output_path("sample_output.json").write_text(json.dumps(results, indent=2))
