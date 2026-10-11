"""
Portfolio exposure calculator used by the stress-testing engine.

Every position is classified by how directly an event reaches it:
  direct   - the obligor itself is named in the headline       (x1.50)
  sector   - the obligor's sector is on the event's risk path   (x1.00)
  systemic - everything else; market-wide spillover  (x0.50 for systemic
             risk-off events, x0.10 for company-specific events)

Rate and credit-index derivatives (sectors "Rates" / "Credit") are pure
risk-factor instruments and always take the full factor move.

The multiplier scales the event-specific parts of the shock (equity moves
and credit-spread widening). Interest-rate moves are market-wide and hit
every rate-sensitive position in full.
"""

import json
from functools import lru_cache

from src.config import data_path

EXPOSURE_MULTIPLIER = {"direct": 1.50, "sector": 1.00, "systemic": 0.50, "factor": 1.00}
IDIOSYNCRATIC_SPILLOVER = 0.10
FACTOR_SECTORS = {"Rates", "Credit"}


@lru_cache(maxsize=1)
def _load_portfolio() -> dict:
    return json.loads(data_path("synthetic_portfolio.json").read_text())


def get_portfolio() -> dict:
    return _load_portfolio()


def classify_position(pos: dict, ticker_set: set[str], sector_set: set[str]) -> str:
    ticker = pos.get("ticker")
    sector = pos.get("sector")
    if sector in FACTOR_SECTORS:
        return "factor"
    if ticker and str(ticker).upper() in ticker_set:
        return "direct"
    if sector and sector in sector_set:
        return "sector"
    return "systemic"


def calculate_exposure(
    affected_tickers: list[str],
    affected_sectors: list[str],
    portfolio: dict | None = None,
    systemic_multiplier: float | None = None,
) -> dict:
    """Classify every position; exposure % counts direct + sector positions."""
    multipliers = dict(EXPOSURE_MULTIPLIER)
    if systemic_multiplier is not None:
        multipliers["systemic"] = float(systemic_multiplier)
    portfolio = portfolio or _load_portfolio()
    positions = portfolio["positions"]
    total_value = float(portfolio.get("total_value") or sum(float(p["value"]) for p in positions))
    ticker_set = {str(t).upper() for t in affected_tickers}
    sector_set = set(affected_sectors)

    classified = []
    for pos in positions:
        exposure_type = classify_position(pos, ticker_set, sector_set)
        classified.append({
            "id": pos["id"],
            "type": pos["type"],
            "obligor": pos["obligor"],
            "ticker": pos.get("ticker"),
            "sector": pos.get("sector"),
            "value": float(pos["value"]),
            "exposure_type": exposure_type,
            "exposure_multiplier": multipliers[exposure_type],
            "credit_rating": pos.get("credit_rating"),
            "duration": pos.get("duration", 0),
            "spread_duration": pos.get("spread_duration", pos.get("duration", 0)),
            "rate_dv01": pos.get("rate_dv01"),
            "spread_cs01": pos.get("spread_cs01"),
            "direction": pos.get("direction"),
        })

    exposed = [p for p in classified if p["exposure_type"] in {"direct", "sector"}]
    exposed_value = sum(p["value"] for p in exposed)
    return {
        "total_portfolio_value": total_value,
        "exposed_value": exposed_value,
        "exposure_pct": round(exposed_value / total_value * 100, 2) if total_value else 0.0,
        "direct_count": sum(1 for p in classified if p["exposure_type"] == "direct"),
        "sector_count": sum(1 for p in classified if p["exposure_type"] == "sector"),
        "positions": classified,
    }
