"""
Event Classifier.

Two-stage hybrid:
  1. High-precision keyword/regex rules for obvious cases (fast, explainable).
  2. Zero-shot NLI (cross-encoder/nli-deberta-v3-small) for everything else.
     When keyword rules tie between categories, the NLI model decides
     between the tied categories only ("keywords narrow, NLI decides").
  3. Abstention: if the NLI model is not confident (top probability below
     ZERO_SHOT_MIN_CONFIDENCE) or picks the explicit "Other" description, the
     text is labelled "Other" (market commentary / non-event). "Other" never
     moves the index or triggers a stress test - precision over recall.

Keyword confidence is a heuristic ranking score, not a calibrated
probability. Zero-shot scores are model probabilities over the supplied
candidate labels.
"""

from functools import lru_cache
import re

from src.config import EVENT_CLASSES

# Each entry is either a plain phrase (matched case-insensitively on word
# boundaries) or a raw regex prefixed with "re:".
KEYWORD_RULES = {
    "Geopolitical": [
        r"re:(?<!price )\bwars?\b", "military", "sanctions", "invasion",
        "invades", "invade", r"re:\bgeopolit\w*", "embargo", "tariff", "tariffs",
        "trade war", "nato", "missile", "missiles", "troops", "taiwan strait",
        "airstrike", "airstrikes", "ceasefire", "export restriction",
        "export restrictions", "export control", "export controls",
        r"re:\bsupply[- ]chain disruptions?\b", "oil supply", "oil embargo",
        "russia", "russian", "ukraine", "middle east", "opec", "blockade",
        "armed conflict", "terror attack", "terrorist attack", "coup",
        "military drills",
    ],
    "Macroeconomic": [
        "federal reserve", r"re:\bthe fed\b", r"re:\bfed (rate|chair|officials?|signals?|minutes|raises|cuts|holds)\b",
        "fomc", "interest rate", "interest rates", "inflation", "gdp",
        "recession", "unemployment", "jobless claims", "cpi", "ppi",
        "basis point", "basis points", "rate hike", "rate hikes", "rate cut",
        "rate cuts", "central bank", "monetary policy", "consumer prices",
        "payrolls", "jobs report", "yield curve", "treasury yields",
        "bond yields", "ecb", "bank of england", "bank of japan",
        "fiscal stimulus", "stimulus package", "pmi", "purchasing managers",
        "manufacturing index", "services index", "trade deficit",
        "trade surplus", "retail sales", "housing starts", "consumer confidence",
        "imf", "world bank", "gdp growth",
    ],
    "Credit Event": [
        "debt default", "sovereign default", "default risk", "corporate defaults",
        "loan defaults", "default rates", "defaulted", r"re:\bdefaults? on\b",
        r"re:\bmiss(es|ed)? (a |an |its )?(coupon|interest|debt|bond) payments?\b",
        r"re:\bdowngrad\w*.{0,40}\b(credit|rating|junk|outlook)\b",
        r"re:\b(credit|debt|sovereign) (rating )?downgrades?\b",
        "credit rating", "junk bond", "junk bonds", "junk status", "bankruptcy",
        r"re:\b(moody'?s|fitch|s&p global ratings)\b.{0,60}\b(downgrad\w*|cuts?|rating|outlook)\b",
        "chapter 11", "files for insolvency", "credit spread",
        "credit spreads", "leveraged loan", "leveraged loans",
        "debt restructuring", "bank failure", "bank fails", "failed bank",
        "bank run", "run on the bank", "bailout", "emergency rescue",
        "solvency", "seized by regulators", "liquidity crisis", "contagion",
        "distressed debt", "margin call",
    ],
    "Merger/Acquisition": [
        "acquires", "acquire", "acquired", "acquisition", "acquisitions",
        "merger", "mergers", r"re:\bmerges? with\b", "takeover", "buyout",
        "purchase agreement", "bought by", "agrees to buy", "agrees to acquire",
        "to be acquired", "majority stake", "tender offer", "hostile bid",
        "deal to buy",
    ],
    "Product Launch": [
        "unveils", "unveiled", "unveil", "announces new", "introduces",
        "new product", "new model", "launches", "launch of", "product launch",
        "product release", "rolls out", r"re:\breleases? (a |its )?new\b",
        "new iphone", "new chip", "new platform", "new feature", "new features",
        r"re:\bnext[- ]generation\b",
    ],
    "Earnings/Corporate": [
        "earnings", "quarterly", r"re:\brevenues?\b", "profit", "profits",
        "eps", r"re:\b(beat|beats|top|tops|miss|misses|missed|above|below)\s+(analyst\s+|wall street\s+)?(estimates|expectations|forecasts)\b",
        "guidance", "layoffs", "job cuts", "cuts jobs", "lays off",
        r"re:\brecall(s|ed|ing)?\b",
        r"re:(?<!missile )(?<!drone )(?<!military )\bstrikes?\b",
        "price target", "buyback", "share repurchase", "dividend",
        "deliveries", "same-store sales", r"re:\bmargins?\b",
        r"re:\bq[1-4]\b", "ceo resigns", "ceo steps down", "steps down as ceo",
        r"re:\b(buy|sell|hold|overweight|underweight|outperform) rating\b",
        "price cuts", r"re:\bcuts prices\b",
    ],
    "Other": [],  # assigned by abstention, never by keywords
}

# Zero-shot candidate descriptions. Verbose labels work better with NLI.
EVENT_DESCRIPTIONS = {
    "Geopolitical": "Geopolitical risk or military conflict or trade sanctions or oil supply",
    "Macroeconomic": "Macroeconomic policy or central bank interest rates or inflation data",
    "Credit Event": "Credit risk or debt default or credit rating downgrade or bankruptcy",
    "Merger/Acquisition": "Merger or acquisition or company buyout or takeover deal",
    "Product Launch": "Product launch or new technology release or innovation announcement",
    "Earnings/Corporate": "Company earnings results or guidance or layoffs or recalls or other company-specific news",
    "Other": "General market commentary, personal finance, crypto prices or lifestyle story with no specific event",
}
assert set(EVENT_DESCRIPTIONS) == set(EVENT_CLASSES)
assert set(KEYWORD_RULES) == set(EVENT_CLASSES)

EVENT_CATEGORIES = [EVENT_DESCRIPTIONS[c] for c in EVENT_CLASSES]
LABEL_MAP = {EVENT_DESCRIPTIONS[c]: c for c in EVENT_CLASSES}

# Below this zero-shot probability the classifier abstains and returns "Other".
ZERO_SHOT_MIN_CONFIDENCE = 0.45
LOW_CONFIDENCE_THRESHOLD = ZERO_SHOT_MIN_CONFIDENCE  # backwards-compatible name
HYPOTHESIS_TEMPLATE = "This financial news headline is about {}."


def _compile(rule: str) -> re.Pattern:
    if rule.startswith("re:"):
        return re.compile(rule[3:], re.IGNORECASE)
    return re.compile(rf"(?<!\w){re.escape(rule.lower())}(?!\w)", re.IGNORECASE)


_COMPILED_RULES = {
    category: [(rule, _compile(rule)) for rule in rules]
    for category, rules in KEYWORD_RULES.items()
}


def _rule_length(rule: str) -> int:
    """Specificity proxy: number of literal characters in the rule."""
    if rule.startswith("re:"):
        return len(re.sub(r"[\\()?:<!|\[\]{}.*+^$]|\\[bwsd]", "", rule[3:]))
    return len(rule)


@lru_cache(maxsize=1)
def _load_classifier():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=-1,
    )


def _keyword_scores(text: str) -> dict:
    """Return {category: (hit_count, longest_rule_length, [matched rules])}."""
    scores = {}
    for category, rules in _COMPILED_RULES.items():
        hits = [rule for rule, pattern in rules if pattern.search(text)]
        if hits:
            scores[category] = (len(hits), max(_rule_length(h) for h in hits), hits)
    return scores


def _keyword_match(text: str):
    """
    Fast keyword pre-screen.

    Returns None when no rule matches, otherwise a dict with the winning
    category, a heuristic confidence, the matched rules, and `tied`, the list
    of categories sharing the top hit count (len > 1 means ambiguous).
    """
    scores = _keyword_scores(text)
    if not scores:
        return None

    top_count = max(v[0] for v in scores.values())
    tied = [c for c, v in scores.items() if v[0] == top_count]
    category = max(tied, key=lambda c: scores[c][1])
    hit_count, longest, hits = scores[category]

    specificity = min(longest / 20.0, 1.0)
    confidence = min(0.55 + (hit_count - 1) * 0.12 + specificity * 0.15, 0.97)
    if len(tied) > 1:
        confidence = min(confidence, 0.55)
    return {
        "category": category,
        "confidence": round(confidence, 3),
        "matched_rules": hits,
        "tied": tied,
    }


def _zero_shot(text: str, categories: list[str]) -> dict:
    classifier = _load_classifier()
    labels = [EVENT_DESCRIPTIONS[c] for c in categories]
    result = classifier(
        text,
        candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
        truncation=True,
    )
    return {
        LABEL_MAP[label]: round(float(score), 4)
        for label, score in zip(result["labels"], result["scores"])
    }


def classify_event(text: str) -> dict:
    """Return event class, confidence, per-class scores and the method used."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("text must be non-empty")

    keyword_result = _keyword_match(text)

    if keyword_result and len(keyword_result["tied"]) == 1:
        category = keyword_result["category"]
        confidence = keyword_result["confidence"]
        return {
            "event_class": category,
            "confidence": confidence,
            "all_scores": {category: confidence},
            "method": "keyword",
            "matched_rules": keyword_result["matched_rules"],
            "low_confidence": False,
        }

    if keyword_result:
        # Ambiguous keyword evidence: let NLI choose among the tied classes.
        candidates = keyword_result["tied"]
        method = "keyword+zero-shot"
    else:
        candidates = list(EVENT_CLASSES)
        method = "zero-shot"

    try:
        scores = _zero_shot(text, candidates)
    except Exception:
        if not keyword_result:
            raise
        # NLI unavailable: fall back to the keyword tie-break.
        category = keyword_result["category"]
        return {
            "event_class": category,
            "confidence": keyword_result["confidence"],
            "all_scores": {category: keyword_result["confidence"]},
            "method": "keyword",
            "matched_rules": keyword_result["matched_rules"],
            "low_confidence": True,
        }

    category = max(scores, key=scores.get)
    confidence = scores[category]
    abstained = False
    if method == "zero-shot" and category != "Other" and confidence < ZERO_SHOT_MIN_CONFIDENCE:
        category, abstained = "Other", True
    return {
        "event_class": category,
        "confidence": confidence,
        "all_scores": scores,
        "method": method + ("+abstain" if abstained else ""),
        "matched_rules": keyword_result["matched_rules"] if keyword_result else [],
        "low_confidence": abstained,
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
        ("Nike earnings beat estimates as China sales rebound", "Earnings/Corporate"),
        ("Tesla recalls 27,000 Cybertrucks over camera delay", "Earnings/Corporate"),
    ]

    passed_count = 0
    for text, expected in tests:
        result = classify_event(text)
        got = result["event_class"]
        passed = got == expected
        passed_count += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {result['method']:<18} {got:<20} | {text[:70]}")
    print(f"\n{passed_count}/{len(tests)} correct")
