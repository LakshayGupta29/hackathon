"""
Explainable 1-10 event impact scoring.

impact = 10 * sum(weight_i * factor_i), every factor in [0, 1].

The weights and severity priors are transparent domain priors, not fitted
coefficients: there is no labelled "market impact" ground truth at this
scale, so the score is designed to be inspectable and tunable rather than
claimed as statistically estimated.
"""

IMPACT_WEIGHTS = {
    "sentiment_magnitude": 0.25,
    "event_severity_prior": 0.30,
    "novelty_score": 0.20,
    "entity_count_factor": 0.15,
    "source_authority": 0.10,
}

# Heuristic severity prior per event class (systemic > idiosyncratic).
EVENT_SEVERITY = {
    "Geopolitical": 0.85,
    "Credit Event": 0.75,
    "Macroeconomic": 0.65,
    "Earnings/Corporate": 0.45,
    "Merger/Acquisition": 0.35,
    "Product Launch": 0.25,
    "Other": 0.15,
}

# Source credibility / market-moving reach. Matched by substring so RSS feed
# ids such as "news:cnbc_markets" or NewsAPI names such as
# "news:Financial Times" resolve to the right outlet.
SOURCE_AUTHORITY = {
    "reuters": 1.00,
    "bloomberg": 1.00,
    "financial times": 0.90,
    "ft_": 0.90,
    "wsj": 0.90,
    "wall street journal": 0.90,
    "dow jones": 0.90,
    "cnbc": 0.75,
    "marketwatch": 0.70,
    "seeking_alpha": 0.60,
    "seeking alpha": 0.60,
    "investing": 0.60,
    "techcrunch": 0.60,
}
DEFAULT_NEWS_AUTHORITY = 0.50
SOCIAL_AUTHORITY = 0.40


def source_authority(source: str) -> float:
    """Authority in [0, 1] for one source identifier."""
    s = str(source or "").strip().lower()
    if s.startswith("social:"):
        return SOCIAL_AUTHORITY
    name = s.split(":", 1)[1] if ":" in s else s
    if name in {"ft", "financial_times"}:
        return 0.90
    for key, value in SOURCE_AUTHORITY.items():
        if key in name:
            return value
    return DEFAULT_NEWS_AUTHORITY


def _sentiment_magnitude(sentiment_score: float) -> float:
    return min(1.0, abs(float(sentiment_score)))


def _entity_count_factor(entity_count: int) -> float:
    """More affected companies -> broader impact; saturates at 5."""
    return min(max(int(entity_count), 0) / 5.0, 1.0)


def _source_authority_factor(sources: list[str]) -> float:
    if not sources:
        return DEFAULT_NEWS_AUTHORITY
    scores = [source_authority(s) for s in sources]
    return sum(scores) / len(scores)


def compute_impact(
    sentiment_score: float,
    event_class: str,
    novelty_score: float,
    affected_tickers: list,
    sources: list,
) -> dict:
    """Compute an explainable 1-10 impact score with factor breakdown."""
    factors = {
        "sentiment_magnitude": _sentiment_magnitude(sentiment_score),
        "event_severity_prior": EVENT_SEVERITY.get(event_class, 0.50),
        "novelty_score": min(1.0, max(0.0, float(novelty_score))),
        "entity_count_factor": _entity_count_factor(len(affected_tickers)),
        "source_authority": _source_authority_factor(sources),
    }

    raw = sum(factors[name] * weight for name, weight in IMPACT_WEIGHTS.items())
    impact = round(max(1.0, min(10.0, raw * 10.0)), 2)

    return {
        "impact_score": impact,
        "breakdown": {k: round(v, 3) for k, v in factors.items()},
        "contributions": {
            k: round(factors[k] * IMPACT_WEIGHTS[k] * 10.0, 3) for k in factors
        },
        "weights": IMPACT_WEIGHTS.copy(),
    }
