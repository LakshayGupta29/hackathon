"""
Impact Scorer
Combines 5 signals into a single impact score (1-10).

Formula:
  impact = (
      sentiment_magnitude  * 0.25 +
      event_severity_prior * 0.30 +
      novelty_score        * 0.20 +
      entity_count_factor  * 0.15 +
      source_authority     * 0.10
  ) * 10

All factors are in range [0, 1] before scaling.

Why formula not model:
  - No ground truth for "market impact" exists
  - Formula is fully explainable to jury
  - Every coefficient can be defended and tuned
"""

# How severe each event type is historically
# Based on average S&P 500 drawdown during past events
# Geopolitical crises: ~8% drawdown
# Macro shocks: ~5% drawdown
# Credit events: ~6% drawdown
# M&A: slight positive, low severity
# Product launches: slight positive, low severity
EVENT_SEVERITY = {
    "Geopolitical":      0.85,
    "Macroeconomic":     0.65,
    "Credit Event":      0.75,
    "Merger/Acquisition": 0.35,
    "Product Launch":    0.25,
}

# Source authority weights
# Higher = more market-moving
SOURCE_AUTHORITY = {
    "news:reuters":    1.0,
    "news:bloomberg":  1.0,
    "news:ft":         0.90,
    "news:wsj":        0.90,
    "news:cnbc":       0.75,
    "news:unknown":    0.50,
    "social:twitter":  0.40,
}


def _sentiment_magnitude(sentiment_score: float) -> float:
    """Convert signed score (-1 to 1) to magnitude (0 to 1)."""
    return abs(sentiment_score)


def _entity_count_factor(entity_count: int) -> float:
    """
    More affected companies = broader market impact.
    Capped at 5 entities (diminishing returns after that).
    """
    return min(entity_count / 5.0, 1.0)


def _source_authority_factor(sources: list) -> float:
    """Average authority across all sources that reported this event."""
    if not sources:
        return 0.5
    scores = [SOURCE_AUTHORITY.get(s, 0.5) for s in sources]
    return sum(scores) / len(scores)


def compute_impact(
    sentiment_score: float,
    event_class: str,
    novelty_score: float,
    affected_tickers: list,
    sources: list,
) -> dict:
    """
    Compute impact score with full breakdown.

    Returns:
        {
            "impact_score": float (1-10),
            "breakdown": dict  (each factor's contribution)
        }
    """
    f1 = _sentiment_magnitude(sentiment_score)
    f2 = EVENT_SEVERITY.get(event_class, 0.50)
    f3 = novelty_score
    f4 = _entity_count_factor(len(affected_tickers))
    f5 = _source_authority_factor(sources)

    raw = (
        f1 * 0.25 +
        f2 * 0.30 +
        f3 * 0.20 +
        f4 * 0.15 +
        f5 * 0.10
    )

    # Scale to 1-10, clamp to valid range
    impact = round(max(1.0, min(10.0, raw * 10)), 2)

    return {
        "impact_score": impact,
        "breakdown": {
            "sentiment_magnitude":  round(f1, 3),
            "event_severity_prior": round(f2, 3),
            "novelty_score":        round(f3, 3),
            "entity_count_factor":  round(f4, 3),
            "source_authority":     round(f5, 3),
        }
    }


if __name__ == "__main__":
    tests = [
        {
            "label": "Taiwan crisis (should be HIGH ~8-9)",
            "sentiment_score": -0.85,
            "event_class": "Geopolitical",
            "novelty_score": 0.95,
            "affected_tickers": ["TSM", "NVDA", "ASML", "AAPL"],
            "sources": ["news:reuters", "news:bloomberg", "news:ft"],
        },
        {
            "label": "M&A deal (should be MEDIUM ~4-5)",
            "sentiment_score": 0.60,
            "event_class": "Merger/Acquisition",
            "novelty_score": 0.80,
            "affected_tickers": ["AAPL"],
            "sources": ["news:cnbc"],
        },
        {
            "label": "Repeated tweet (should be LOW ~2-3)",
            "sentiment_score": -0.30,
            "event_class": "Macroeconomic",
            "novelty_score": 0.10,
            "affected_tickers": [],
            "sources": ["social:twitter"],
        },
    ]

    print("Impact score breakdown:\n")
    for t in tests:
        result = compute_impact(
            t["sentiment_score"],
            t["event_class"],
            t["novelty_score"],
            t["affected_tickers"],
            t["sources"],
        )
        print(f"  {t['label']}")
        print(f"  Score: {result['impact_score']}/10")
        for k, v in result["breakdown"].items():
            print(f"    {k:<25} {v:.3f}")
        print()
