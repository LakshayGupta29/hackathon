"""
GDELT Event Fetcher
Pulls real historical financial/geopolitical news events from
GDELT's free public API, for expanding our validation set.

GDELT (gdeltproject.org) is a free, open database of global news
events updated every 15 minutes, with full historical archives.
This gives us a genuinely large, unbiased sample instead of
hand-picking 20 headlines, which avoids cherry-picking bias.
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Targeted queries: company name + event type, restricted to
# known financial news domains. Far more relevant than broad
# topic words which match the entire global news index.
FINANCIAL_DOMAINS = (
    "domainis:reuters.com OR domainis:bloomberg.com OR "
    "domainis:cnbc.com OR domainis:wsj.com OR domainis:ft.com"
)

QUERIES = [
    f"NVIDIA earnings ({FINANCIAL_DOMAINS})",
    f"Apple earnings ({FINANCIAL_DOMAINS})",
    f"JPMorgan credit ({FINANCIAL_DOMAINS})",
    f"Goldman Sachs downgrade ({FINANCIAL_DOMAINS})",
    f"Boeing crisis ({FINANCIAL_DOMAINS})",
    f"Tesla delivery ({FINANCIAL_DOMAINS})",
    f"Meta layoffs ({FINANCIAL_DOMAINS})",
    f"Microsoft acquisition ({FINANCIAL_DOMAINS})",
    f"Federal Reserve rate ({FINANCIAL_DOMAINS})",
    f"bank failure collapse ({FINANCIAL_DOMAINS})",
    f"Chevron oil earnings ({FINANCIAL_DOMAINS})",
    f"semiconductor export restriction ({FINANCIAL_DOMAINS})",
]


def fetch_gdelt_events(query: str, max_records: int = 20,
                        start_date: str = "20220101",
                        end_date: str = "20241231") -> list:
    """
    Query GDELT for news articles matching a search term within
    a date range. Returns list of {title, date, url, domain}.
    """
    params = {
        "query": f"{query} sourcelang:eng",
        "mode":  "artlist",
        "maxrecords": max_records,
        "format": "json",
        "startdatetime": f"{start_date}000000",
        "enddatetime":   f"{end_date}235959",
        "sort": "hybridrel",
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [rate limited, waiting {wait}s...]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "headline": a.get("title", ""),
                    "date":     a.get("seendate", "")[:8],
                    "url":      a.get("url", ""),
                    "domain":   a.get("domain", ""),
                }
                for a in articles if a.get("title")
            ]
        except Exception as e:
            print(f"  [GDELT error for '{query}']: {e}")
            return []
    print(f"  [GDELT gave up on '{query}' after {max_retries} retries]")
    return []


def fetch_all(max_per_query: int = 15) -> list:
    """Fetch across all queries, dedupe, format dates."""
    all_events = []
    for q in QUERIES:
        print(f"Querying GDELT: {q}")
        results = fetch_gdelt_events(q, max_records=max_per_query)
        print(f"  -> {len(results)} articles")
        all_events.extend(results)
        time.sleep(5)  # be polite, avoid rate limit between queries

    # Dedup by headline prefix
    seen = set()
    unique = []
    for e in all_events:
        key = e["headline"][:60].lower()
        if key not in seen and e["date"]:
            seen.add(key)
            # Reformat date YYYYMMDD -> YYYY-MM-DD
            d = e["date"]
            if len(d) == 8:
                e["date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                unique.append(e)

    print(f"\nTotal: {len(all_events)}, unique: {len(unique)}")
    return unique


if __name__ == "__main__":
    events = fetch_all()
    Path("data/gdelt_raw_events.json").write_text(
        json.dumps(events, indent=2)
    )
    print(f"\nSaved {len(events)} events to data/gdelt_raw_events.json")
    print("\nSample:")
    for e in events[:10]:
        print(f"  {e['date']} | {e['headline'][:70]}")
