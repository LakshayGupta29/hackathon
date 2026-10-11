"""
NER + Entity Linker.

Extracts S&P 100 ticker mentions from raw text using three signals:
  1. Cashtags ($NVDA) - regex, highest precision.
  2. Company names/aliases ("Nvidia", "Goldman Sachs", "TSMC") - matched on
     word boundaries, so "metals" never becomes META and "Visa restrictions"
     never becomes V.
  3. Plain uppercase tickers ("NVDA shares") for tickers of 3+ letters.

Ambiguous common-word names (Apple, Visa, Meta, Oracle, ...) must be
capitalised, and Visa is rejected in immigration contexts. spaCy is used
when available as an extra check that an ambiguous name is tagged as an
organisation; the linker still works without spaCy installed.
"""

from functools import lru_cache
import re

import pandas as pd

from src.config import data_path

# Curated aliases: lowercase alias -> ticker. Merged with names from the CSV.
ALIASES = {
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "asml": "ASML",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "youtube": "GOOGL",
    "meta": "META",
    "meta platforms": "META",
    "facebook": "META",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "j.p. morgan": "JPM",
    "goldman": "GS",
    "goldman sachs": "GS",
    "berkshire": "BRK.B",
    "berkshire hathaway": "BRK.B",
    "visa": "V",
    "mastercard": "MA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "aws": "AMZN",
    "tesla": "TSLA",
    "boeing": "BA",
    "chevron": "CVX",
    "lockheed": "LMT",
    "lockheed martin": "LMT",
    "unitedhealth": "UNH",
    "johnson & johnson": "JNJ",
    "j&j": "JNJ",
    "procter & gamble": "PG",
    "p&g": "PG",
    "home depot": "HD",
    "merck": "MRK",
    "abbvie": "ABBV",
    "pepsico": "PEP",
    "pepsi": "PEP",
    "coca-cola": "KO",
    "broadcom": "AVGO",
    "costco": "COST",
    "thermo fisher": "TMO",
    "walmart": "WMT",
    "wal-mart": "WMT",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "cisco": "CSCO",
    "accenture": "ACN",
    "abbott laboratories": "ABT",
    "danaher": "DHR",
    "nike": "NKE",
    "nextera": "NEE",
    "texas instruments": "TXN",
    "philip morris": "PM",
    "honeywell": "HON",
    "oracle": "ORCL",
    "united parcel service": "UPS",
    "caterpillar": "CAT",
    "qualcomm": "QCOM",
    "amgen": "AMGN",
    "intuit": "INTU",
    "s&p global": "SPGI",
    "raytheon": "RTX",
    "applied materials": "AMAT",
    "john deere": "DE",
    "deere": "DE",
    "intuitive surgical": "ISRG",
    "american express": "AXP",
    "amex": "AXP",
    "booking holdings": "BKNG",
    "advanced micro devices": "AMD",
    "stryker": "SYK",
    "mondelez": "MDLZ",
    "analog devices": "ADI",
    "gilead": "GILD",
    "marsh mclennan": "MMC",
    "marsh & mclennan": "MMC",
    "tjx": "TJX",
}

# Names that are also ordinary English words: require capitalisation.
AMBIGUOUS_ALIASES = {
    "apple", "visa", "meta", "oracle", "nike", "caterpillar", "intuit",
    "amazon", "deere", "merck", "abbott", "costco", "goldman",
}

# Plain-uppercase tickers that collide with common all-caps words.
PLAIN_TICKER_EXCLUDE = {"CAT", "COST", "NEE", "LOW", "ALL", "KEY", "NOW"}

VISA_NON_COMPANY_AFTER = re.compile(
    r"^\s+(restrictions?|rules?|applications?|holders?|programs?|requirements?|"
    r"policy|policies|waivers?|fees?|bans?|process(ing)?|regime|curbs?|"
    r"backlogs?|overstays?|interviews?|appointments?|lottery)\b",
    re.IGNORECASE,
)
VISA_NON_COMPANY_BEFORE = re.compile(
    r"(h-?1b|work|student|tourist|travel|golden|business|entry|transit|"
    r"immigrant|investor)\s*$",
    re.IGNORECASE,
)

CORPORATE_SUFFIXES = re.compile(
    r"(\.com)?\s+(inc|corporation|corp|company|companies|plc|co|holding|"
    r"holdings|group|international)\.?$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_spacy():
    """Optional spaCy model; returns None if spaCy or the model is missing."""
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_ticker_data():
    df = pd.read_csv(data_path("sp100_tickers.csv"))
    tickers = set(df["ticker"].astype(str))

    alias_map = {}
    for _, row in df.iterrows():
        company = str(row["company"]).strip()
        alias_map[company.lower()] = row["ticker"]
        stripped = re.sub(r"\s+&\s+(co|company)\.?$", "", company, flags=re.IGNORECASE)
        stripped = CORPORATE_SUFFIXES.sub("", stripped).strip()
        if len(stripped) >= 3:
            alias_map[stripped.lower()] = row["ticker"]
    alias_map.update(ALIASES)
    # Drop aliases that are only a generic first word.
    for generic in ("united", "american", "applied", "advanced", "analog",
                    "texas", "home", "s&p", "booking", "abbott"):
        alias_map.pop(generic, None)

    compiled = []
    for alias, ticker in alias_map.items():
        escaped = re.escape(alias)
        if alias in AMBIGUOUS_ALIASES:
            # Case-sensitive, capitalised form only ("Apple", "APPLE").
            variants = {alias.capitalize(), alias.upper()}
            pattern = re.compile(
                r"(?<![\w$])(" + "|".join(re.escape(v) for v in variants) + r")(?![\w-])"
            )
        else:
            pattern = re.compile(rf"(?<![\w$]){escaped}(?!\w)", re.IGNORECASE)
        compiled.append((alias, ticker, pattern))
    # Longest aliases first so "goldman sachs" wins over "goldman".
    compiled.sort(key=lambda item: len(item[0]), reverse=True)

    plain = sorted(
        t for t in tickers
        if len(t.replace(".", "")) >= 3 and t not in PLAIN_TICKER_EXCLUDE
    )
    plain_pattern = re.compile(
        r"(?<![\w$])(" + "|".join(re.escape(t) for t in plain) + r")(?!\w|\.\w)"
    )
    return tickers, compiled, plain_pattern


def _org_spans(text: str):
    nlp = _load_spacy()
    if nlp is None:
        return None
    return [(ent.start_char, ent.end_char) for ent in nlp(text).ents if ent.label_ == "ORG"]


def _inside_org(span, org_spans) -> bool:
    if org_spans is None:
        return True  # spaCy unavailable: rely on capitalisation rule only
    start, end = span
    return any(s <= start and end <= e for s, e in org_spans)


def _is_mostly_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6


def _extract_cashtags(text: str) -> list[tuple[int, str]]:
    tickers, _, _ = _load_ticker_data()
    found = []
    for m in re.finditer(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b", text):
        ticker = m.group(1).upper()
        if ticker in tickers:
            found.append((m.start(), ticker))
    return found


def _extract_names(text: str) -> list[tuple[int, str]]:
    _, compiled, _ = _load_ticker_data()
    org_spans = None
    found = []
    claimed = []  # character spans already matched by a longer alias
    for alias, ticker, pattern in compiled:
        for m in pattern.finditer(text):
            span = m.span()
            if any(s <= span[0] and span[1] <= e for s, e in claimed):
                continue
            if alias == "visa":
                if VISA_NON_COMPANY_AFTER.match(text[span[1]:]) or \
                        VISA_NON_COMPANY_BEFORE.search(text[:span[0]]):
                    continue
            if alias == "oracle" and re.match(r"\s+of\b", text[span[1]:]):
                continue  # "Oracle of Omaha"
            if alias in AMBIGUOUS_ALIASES:
                if org_spans is None:
                    org_spans = _org_spans(text)
                # Sentence-initial ambiguous words need the spaCy ORG check.
                at_start = span[0] == 0 or text[:span[0]].rstrip().endswith((".", ":", "!", "?"))
                if at_start and not _inside_org(span, org_spans):
                    continue
            claimed.append(span)
            found.append((span[0], ticker))
    return found


def _extract_plain_tickers(text: str) -> list[tuple[int, str]]:
    if _is_mostly_uppercase(text):
        return []
    _, _, plain_pattern = _load_ticker_data()
    return [(m.start(), m.group(1)) for m in plain_pattern.finditer(text)]


def extract_tickers(text: str) -> list[str]:
    """Return deduplicated S&P 100 tickers in order of first mention."""
    text = str(text or "")
    hits = _extract_cashtags(text) + _extract_names(text) + _extract_plain_tickers(text)
    ordered = [ticker for _, ticker in sorted(hits, key=lambda h: h[0])]
    return list(dict.fromkeys(ordered))


if __name__ == "__main__":
    tests = [
        ("$NVDA absolutely crushing it this quarter", ["NVDA"]),
        ("Taiwan tensions hit TSMC and ASML hard", ["TSM", "ASML"]),
        ("Apple announces acquisition of AI startup", ["AAPL"]),
        ("$JPM $GS both down on credit concerns", ["JPM", "GS"]),
        ("No companies mentioned here at all", []),
        ("Google and Facebook face antitrust scrutiny", ["GOOGL", "META"]),
        ("JPMorgan warns on credit spreads", ["JPM"]),
        ("Base metals rally as China stimulus lifts demand", []),
        ("US tightens H-1B visa rules; Visa restrictions hit students", []),
        ("Visa Inc beats estimates on cross-border volumes", ["V"]),
        ("The S&P 500 fell 2 percent", []),
    ]
    all_passed = True
    for text, expected in tests:
        result = extract_tickers(text)
        passed = result == expected
        all_passed &= passed
        print(f"  {'OK ' if passed else 'BAD'} got={result} expected={expected} | {text}")
    print("All tests passed" if all_passed else "Some tests failed")
