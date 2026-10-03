"""
NER + Entity Linker
Extracts S&P100 ticker mentions from raw text.
Two strategies:
  1. Cashtag scan: finds $NVDA, $AAPL directly (regex, fast)
  2. spaCy NER: finds "NVIDIA", "Apple Inc" and maps to ticker

Known limitation of spaCy en_core_web_sm: misses some abbreviations
like ASML. Handled via manual ALIASES dict below.
"""

import re
import pandas as pd
import spacy
from functools import lru_cache

nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

# Manual aliases for names spaCy misses or maps incorrectly
# Format: "how it appears in text (lowercase)" -> "ticker"
ALIASES = {
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "asml": "ASML",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "berkshire": "BRK.B",
    "berkshire hathaway": "BRK.B",
    "visa": "V",
    "mastercard": "MA",
}


@lru_cache(maxsize=1)
def _load_ticker_map():
    """
    Builds lookup dicts from sp100_tickers.csv + ALIASES.
    Cached so CSV is read only once per process.
    """
    df = pd.read_csv("data/sp100_tickers.csv")

    cashtag_map = {row["ticker"]: row["ticker"] for _, row in df.iterrows()}

    name_map = {}
    for _, row in df.iterrows():
        name_map[row["company"].lower()] = row["ticker"]
        name_map[row["company"].split()[0].lower()] = row["ticker"]

    # Merge manual aliases (these take priority)
    name_map.update(ALIASES)

    return cashtag_map, name_map


def _extract_cashtags(text: str) -> list:
    """Finds $TICKER patterns directly. Fast and high precision."""
    cashtag_map, _ = _load_ticker_map()
    found = re.findall(r'\$([A-Z]{1,5})', text.upper())
    return [t for t in found if t in cashtag_map]


def _extract_named_entities(text: str) -> list:
    """
    Uses spaCy ORG entities then maps to tickers via name_map.
    Falls back to substring match for known aliases.
    """
    _, name_map = _load_ticker_map()
    doc = nlp(text)
    tickers = []

    # spaCy ORG entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            ticker = name_map.get(ent.text.lower())
            if ticker:
                tickers.append(ticker)

    # Substring match for aliases spaCy misses entirely
    text_lower = text.lower()
    for alias, ticker in ALIASES.items():
        if alias in text_lower and ticker not in tickers:
            tickers.append(ticker)

    return tickers


def extract_tickers(text: str) -> list:
    """
    Main function. Returns deduplicated S&P100 tickers found in text.
    Cashtags ($NVDA) are checked first, then company names via spaCy.
    """
    tickers = _extract_cashtags(text) + _extract_named_entities(text)
    return list(dict.fromkeys(tickers))  # deduplicate, preserve order


# Replace the __main__ block with this
if __name__ == "__main__":
    tests = [
        ("$NVDA absolutely crushing it this quarter", ["NVDA"]),
        ("Taiwan tensions hit TSMC and ASML hard", ["TSM", "ASML"]),
        ("Apple announces acquisition of AI startup", ["AAPL"]),
        ("$JPM $GS both down on credit concerns", ["JPM", "GS"]),
        ("No companies mentioned here at all", []),
        ("Google and Facebook face antitrust scrutiny", ["GOOGL", "META"]),
        ("JPMorgan warns on credit spreads", ["JPM"]),
    ]
    all_passed = True
    for text, expected in tests:
        result = extract_tickers(text)
        # Check exact match both ways: nothing missing, nothing extra
        missing = [t for t in expected if t not in result]
        extra = [t for t in result if t not in expected]
        passed = not missing and not extra
        if not passed:
            all_passed = False
        status = "✅" if passed else "❌"
        print(f"  {status} got={result} expected={expected}")
        if missing:
            print(f"     MISSING: {missing}")
        if extra:
            print(f"     EXTRA:   {extra}")

    print(f"\n{'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
