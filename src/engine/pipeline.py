"""
NLP Pipeline
Orchestrates all engine components into a single function call.

Input:  raw text + source string
Output: structured event dict ready for risk engine

Processing order:
  1. NER          → extract affected tickers
  2. Sentiment    → score and label
  3. Event class  → category
  4. Embed        → vector for fusion/novelty
  5. Novelty      → how new is this
  6. Impact       → severity score 1-10
  7. Explainability → triggering sentence
"""

import time
import uuid
from datetime import datetime

from src.engine.ner_linker      import extract_tickers
from src.engine.sentiment       import analyze as analyze_sentiment
from src.engine.event_classifier import classify_event
from src.engine.embedder        import embed
from src.engine.novelty_scorer  import NoveltyScorer
from src.engine.impact_scorer   import compute_impact

import json
from pathlib import Path

# Sector lookup (loaded once)
import pandas as pd
_ticker_df = pd.read_csv("data/sp100_tickers.csv")
_ticker_to_sector = dict(zip(_ticker_df["ticker"], _ticker_df["sector"]))

# Shared novelty scorer instance (stateful across calls)
_novelty_scorer = NoveltyScorer()


def _get_triggering_sentence(text: str, sentiment_label: str) -> str:
    """
    Find the sentence most likely responsible for the sentiment.
    Simple heuristic: sentence with most sentiment-bearing words.
    """
    sentences = [s.strip() for s in text.replace("!", ".").split(".") if len(s.strip()) > 20]
    if not sentences:
        return text[:200]

    # Negative keywords for negative sentiment, positive for positive
    neg_words = {"fall", "drop", "warn", "risk", "crisis", "default",
                 "probe", "miss", "loss", "decline", "fear", "concern"}
    pos_words = {"beat", "surge", "record", "growth", "strong", "exceed",
                 "launch", "acquire", "profit", "rally", "gain", "rise"}

    target_words = neg_words if sentiment_label == "negative" else pos_words

    best_sentence = sentences[0]
    best_score = 0
    for sentence in sentences:
        score = sum(1 for w in target_words if w in sentence.lower())
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence


def process(text: str, source: str = "unknown") -> dict:
    """
    Main pipeline function. Process one raw text item.

    Args:
        text:   raw text (headline, tweet, article snippet)
        source: source identifier e.g. "news:reuters"

    Returns:
        Complete structured event dict.
    """
    start_time = time.time()

    # 1. NER — which tickers are mentioned
    tickers = extract_tickers(text)
    sectors = list({_ticker_to_sector.get(t, "Unknown") for t in tickers})

    # 2. Sentiment
    sentiment = analyze_sentiment(text, mode="finetuned")

    # 3. Event classification
    event_class = classify_event(text)

    # 4. Novelty
    novelty = _novelty_scorer.score(text)

    # 5. Impact
    impact = compute_impact(
        sentiment_score=sentiment["score"],
        event_class=event_class["event_class"],
        novelty_score=novelty["novelty_score"],
        affected_tickers=tickers,
        sources=[source],
    )

    # 6. Explainability
    triggering_sentence = _get_triggering_sentence(text, sentiment["label"])

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "event_id":            str(uuid.uuid4()),
        "timestamp":           datetime.utcnow().isoformat(),
        "headline":            text[:200],
        "source":              source,
        # Sentiment
        "sentiment_score":     sentiment["score"],
        "sentiment_label":     sentiment["label"],
        # Event
        "event_class":         event_class["event_class"],
        "event_confidence":    event_class["confidence"],
        # Novelty
        "novelty_score":       novelty["novelty_score"],
        "novelty_label":       novelty["novelty_label"],
        # Impact
        "impact_score":        impact["impact_score"],
        "impact_breakdown":    impact["breakdown"],
        # Entities
        "affected_tickers":    tickers,
        "affected_sectors":    sectors,
        # Explainability
        "triggering_sentence": triggering_sentence,
        # Meta
        "latency_ms":          latency_ms,
    }


def process_batch(items: list) -> list:
    """
    Process a list of {"text": str, "source": str} dicts.
    Returns list of structured events.
    """
    return [process(item["text"], item.get("source", "unknown"))
            for item in items]


def reset_novelty_memory():
    """Reset between replay sessions."""
    _novelty_scorer.reset()


if __name__ == "__main__":
    test_items = [
        {
            "text": "Military exercises near Taiwan rattle semiconductor supply chains. TSMC and ASML shares fell sharply as geopolitical risk intensified.",
            "source": "news:reuters"
        },
        {
            "text": "Federal Reserve raises interest rates by 50 basis points amid persistent inflation concerns.",
            "source": "news:bloomberg"
        },
        {
            "text": "$NVDA absolutely crushing it this quarter, AI demand is insane",
            "source": "social:twitter"
        },
    ]

    print("Running full NLP pipeline...\n")
    results = process_batch(test_items)

    for r in results:
        print(f"  Headline:   {r['headline'][:65]}")
        print(f"  Sentiment:  {r['sentiment_label']} ({r['sentiment_score']:+.3f})")
        print(f"  Event:      {r['event_class']} (conf={r['event_confidence']:.2f})")
        print(f"  Novelty:    {r['novelty_label']} ({r['novelty_score']:.3f})")
        print(f"  Impact:     {r['impact_score']}/10")
        print(f"  Tickers:    {r['affected_tickers']}")
        print(f"  Trigger:    {r['triggering_sentence'][:65]}")
        print(f"  Latency:    {r['latency_ms']}ms")
        print()

    # Save sample output to data/ for inspection
    Path("data/sample_output.json").write_text(
        json.dumps(results, indent=2)
    )
    print("Sample output saved to data/sample_output.json")
