"""
Exposure Calculator
Finds which portfolio positions are exposed to a given event.

Exposure is determined by two criteria (either triggers exposure):
  1. Direct:   position's ticker is in affected_tickers
  2. Indirect: position's sector is in affected_sectors

Direct exposure is weighted more heavily than indirect.
"""

import json
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_portfolio() -> dict:
    return json.loads(Path("data/synthetic_portfolio.json").read_text())


def get_portfolio() -> dict:
    return _load_portfolio()


def calculate_exposure(
    affected_tickers: list,
    affected_sectors: list,
) -> dict:
    """
    Find all portfolio positions exposed to this event.

    Args:
        affected_tickers: tickers directly mentioned in the event
        affected_sectors: sectors impacted by the event class

    Returns:
        {
            "total_portfolio_value": float,
            "exposed_value":         float,
            "exposure_pct":          float,
            "positions": [
                {
                    "id":              str,
                    "type":            str,
                    "obligor":         str,
                    "ticker":          str,
                    "sector":          str,
                    "value":           float,
                    "exposure_type":   "direct" | "indirect",
                    "credit_rating":   str,
                    "duration":        float,
                }
            ]
        }
    """
    portfolio = _load_portfolio()
    total_value = portfolio["total_value"]
    positions = portfolio["positions"]

    exposed = []
    for pos in positions:
        ticker = pos.get("ticker")
        sector = pos.get("sector")

        if ticker and ticker in affected_tickers:
            exposure_type = "direct"
        elif sector and sector in affected_sectors:
            exposure_type = "indirect"
        else:
            continue  # not exposed

        exposed.append({
            "id":            pos["id"],
            "type":          pos["type"],
            "obligor":       pos["obligor"],
            "ticker":        ticker,
            "sector":        sector,
            "value":         pos["value"],
            "exposure_type": exposure_type,
            "credit_rating": pos.get("credit_rating"),
            "duration":      pos.get("duration", 0),
        })

    exposed_value = sum(p["value"] for p in exposed)
    exposure_pct = round(exposed_value / total_value * 100, 2)

    return {
        "total_portfolio_value": total_value,
        "exposed_value":         exposed_value,
        "exposure_pct":          exposure_pct,
        "positions":             exposed,
    }


if __name__ == "__main__":
    tests = [
        {
            "label": "Geopolitical (Taiwan)",
            "tickers": ["TSM", "NVDA", "ASML"],
            "sectors": ["Energy", "Industrials", "Technology", "Financials"],
        },
        {
            "label": "Credit Event",
            "tickers": ["JPM", "GS"],
            "sectors": ["Financials", "Consumer Discretionary", "Energy"],
        },
    ]

    portfolio = get_portfolio()
    print(f"Portfolio total: ${portfolio['total_value']:,.0f}\n")

    for t in tests:
        result = calculate_exposure(t["tickers"], t["sectors"])
        print(f"Event: {t['label']}")
        print(f"  Exposed value: ${result['exposed_value']:,.0f} "
              f"({result['exposure_pct']}% of portfolio)")
        print(f"  Exposed positions ({len(result['positions'])}):")
        for p in result["positions"]:
            print(f"    [{p['exposure_type']:<8}] {p['id']} "
                  f"{p['obligor']:<30} ${p['value']:>12,.0f}")
        print()
