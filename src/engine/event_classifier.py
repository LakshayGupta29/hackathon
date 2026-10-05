"""
Event Classifier
Classifies financial text into one of 5 event categories.

Approach: keyword pre-screening for high-confidence cases,
zero-shot classification for everything else.
This hybrid approach is faster and more accurate than
pure zero-shot on short financial headlines.
"""

from transformers import pipeline
from functools import lru_cache
import re

# Keyword rules for high-confidence classifications
# Checked before the model runs — faster and more reliable
# for obvious cases
KEYWORD_RULES = {
    "Geopolitical": [
        "war", "military", "sanctions", "conflict", "invasion", "invades",
        "geopolit", "opec", "oil supply", "embargo", "tariff",
        "trade war", "nato", "missile", "troops", "taiwan strait",
        "attacks", "strikes", "airstrike", "ceasefire", "export restriction",
        "export control", "supply chain disruption", "lockdown"
    ],
    "Macroeconomic": [
        "federal reserve", "fed rate", "interest rate", "inflation",
        "gdp", "recession", "unemployment", "cpi", "basis point",
        "rate hike", "rate cut", "rate cuts", "central bank", "monetary policy",
        "layoffs", "job cuts", "margin pressure", "delivery decline",
        "sales decline", "year over year decline", "cost cuts", "buyback"
    ],
    "Credit Event": [
        "default", "downgrade", "credit rating", "junk", "moody",
        "s&p rating", "fitch", "bankruptcy", "insolvency",
        "credit spread", "leveraged loan", "bond yield", "restructur",
        "collapse", "collapses", "bank failure", "bank fails", "seized",
        "run on the bank", "emergency rescue", "rescue deal", "grounded"
    ],
    "Merger/Acquisition": [
        "acqui", "merger", "agrees to acquire", "takeover", "buyout",
        "deal worth", "billion deal", "purchase agreement", "bought by",
        "investment in", "stake in"
    ],
    "Product Launch": [
        "unveils", "announces new product", "introduces new",
        "next generation", "new product", "new model", "debut",
        "forecasts revenue", "revenue beat", "earnings beat",
        "quarterly revenue", "blowout earnings", "surge after earnings"
    ],
}

EVENT_CATEGORIES = [
    "Geopolitical risk or military conflict or trade sanctions or oil supply",
    "Macroeconomic policy or central bank interest rates or inflation data",
    "Credit risk or debt default or credit rating downgrade or bankruptcy",
    "Merger or acquisition or company buyout or takeover deal",
    "Product launch or new technology release or innovation announcement",
]

LABEL_MAP = {
    "Geopolitical risk or military conflict or trade sanctions or oil supply": "Geopolitical",
    "Macroeconomic policy or central bank interest rates or inflation data": "Macroeconomic",
    "Credit risk or debt default or credit rating downgrade or bankruptcy": "Credit Event",
    "Merger or acquisition or company buyout or takeover deal": "Merger/Acquisition",
    "Product launch or new technology release or innovation announcement": "Product Launch",
}


@lru_cache(maxsize=1)
def _load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=-1
    )


def _keyword_match(text: str):
    """
    Fast keyword pre-screen. Returns (category, confidence) if a
    confident match is found, None if ambiguous (falls through to
    zero-shot model).

    Confidence scales with:
      - number of distinct keyword hits in the winning category
      - length of the longest matched keyword (longer = more specific)
    Capped at 0.97 so it never claims total certainty.
    """
    text_lower = text.lower()
    scores = {}
    longest_match = {}
    for category, keywords in KEYWORD_RULES.items():
        hits = [kw for kw in keywords if kw in text_lower]
        if hits:
            scores[category] = len(hits)
            longest_match[category] = max(len(kw) for kw in hits)

    if not scores:
        return None

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_scores) == 1 or sorted_scores[0][1] > sorted_scores[1][1]:
        category = sorted_scores[0][0]
    else:
        # Tie-break: prefer the match with the longest keyword hit
        # (longer phrase = more specific = less likely false positive)
        tied_categories = [c for c, score in sorted_scores
                           if score == sorted_scores[0][1]]
        category = max(tied_categories, key=lambda c: longest_match[c])

    hit_count = scores[category]
    specificity = min(longest_match[category] / 20.0, 1.0)
    confidence = min(0.55 + (hit_count - 1) * 0.12 + specificity * 0.15, 0.97)

    return category, round(confidence, 3)


def classify_event(text: str) -> dict:
    """
    Hybrid classifier: keywords first, zero-shot as fallback.
    Args:
        text: headline or article snippet
    Returns:
        {
            "event_class": str,
            "confidence": float,
            "all_scores": dict,
            "method": "keyword" | "zero-shot"
        }
    """
    # Try keyword match first
    keyword_result = _keyword_match(text)
    if keyword_result:
        category, confidence = keyword_result
        return {
            "event_class": category,
            "confidence": confidence,
            "all_scores": {category: confidence},
            "method": "keyword"
        }

    # Fall back to zero-shot model
    classifier = _load_classifier()
    result = classifier(
        text[:512],
        candidate_labels=EVENT_CATEGORIES,
        multi_label=False
    )

    top_verbose = result["labels"][0]
    top_clean = LABEL_MAP[top_verbose]

    return {
        "event_class": top_clean,
        "confidence": round(result["scores"][0], 4),
        "all_scores": {
            LABEL_MAP[l]: round(s, 4)
            for l, s in zip(result["labels"], result["scores"])
        },
        "method": "zero-shot"
    }


if __name__ == "__main__":
    tests = [
        ("Military exercises near Taiwan rattle semiconductor supply chains", "Geopolitical"),
        ("Federal Reserve raises interest rates by 50 basis points", "Macroeconomic"),
        ("JPMorgan warns of rising corporate defaults in leveraged loan market", "Credit Event"),
        ("Apple acquires AI startup DarwinAI for 3.2 billion dollars", "Merger/Acquisition"),
        ("NVIDIA launches next generation Blackwell GPU architecture", "Product Launch"),
        ("Oil prices surge as OPEC extends production cuts", "Geopolitical"),
        ("Moody's downgrades Boeing credit rating to junk", "Credit Event"),
        ("Microsoft and Activision merger approved by regulators", "Merger/Acquisition"),
        ("Fed signals rate cuts as inflation cools toward 2 percent target", "Macroeconomic"),
        ("Goldman Sachs files for bankruptcy protection amid credit losses", "Credit Event"),
    ]

    print("Running hybrid event classifier...\n")
    passed_count = 0

    for text, expected in tests:
        result = classify_event(text)
        got = result["event_class"]
        passed = got == expected
        if passed:
            passed_count += 1
        status = "✅" if passed else "❌"
        method = result["method"][0].upper()  # K or Z
        print(f"  {status} [{method}] [{got:<20}] conf={result['confidence']:.2f} | {text[:50]}")

    print(f"\n{passed_count}/{len(tests)} correct")
    print("[K]=keyword match, [Z]=zero-shot model")
