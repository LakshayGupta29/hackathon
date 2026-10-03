"""
Shock Generator
Converts event class + impact score into calibrated market shocks.

Base shocks are loaded from data/shock_templates.json.
Impact score (1-10) scales the shock magnitude:
  - Impact 10 → 100% of base shock
  - Impact 5  → 50% of base shock
  - Impact 1  → 10% of base shock

This prevents small events from triggering full stress tests
while keeping the formula transparent and defensible.

Shock types:
  equity_shock        → % change in equity prices
  credit_spread_shock → absolute change in credit spreads (bps as decimal)
  rates_shock         → absolute change in interest rates
  oil_shock           → % change in oil prices
  fx_shock            → % change in FX rates
"""

import json
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_templates() -> dict:
    return json.loads(Path("data/shock_templates.json").read_text())


def _scale_shock(base_value: float, impact_score: float) -> float:
    """
    Scale shock by impact score.
    impact_score 1-10 → scale factor 0.1-1.0
    """
    scale = max(0.1, impact_score / 10.0)
    return round(base_value * scale, 6)


def generate_shocks(event_class: str, impact_score: float) -> dict:
    """
    Generate scaled market shocks for a given event.

    Args:
        event_class:  one of the 5 event categories
        impact_score: 1-10 from impact_scorer

    Returns:
        {
            "event_class":    str,
            "impact_score":   float,
            "scale_factor":   float,
            "shocks": {
                "equity_shock":         float,
                "credit_spread_shock":  float,
                "rates_shock":          float,
                "oil_shock":            float,
                "fx_shock":             float,
            },
            "description": str
        }
    """
    templates = _load_templates()
    template = templates.get(event_class, templates["Macroeconomic"])

    scale = max(0.1, impact_score / 10.0)

    shocks = {
        "equity_shock":        _scale_shock(template["equity_shock"], impact_score),
        "credit_spread_shock": _scale_shock(template["credit_spread_shock"], impact_score),
        "rates_shock":         _scale_shock(template["rates_shock"], impact_score),
        "oil_shock":           _scale_shock(template["oil_shock"], impact_score),
        "fx_shock":            _scale_shock(template["fx_shock"], impact_score),
    }

    return {
        "event_class":  event_class,
        "impact_score": impact_score,
        "scale_factor": round(scale, 2),
        "shocks":       shocks,
        "description":  template["description"],
    }


if __name__ == "__main__":
    tests = [
        ("Geopolitical",       8.5),   # Taiwan crisis
        ("Geopolitical",       3.0),   # minor incident
        ("Credit Event",       7.5),   # significant default
        ("Macroeconomic",      5.0),   # mid-level macro shock
        ("Merger/Acquisition", 4.0),   # M&A deal
    ]

    for event_class, impact in tests:
        result = generate_shocks(event_class, impact)
        print(f"\nEvent: {event_class} | Impact: {impact}/10 "
              f"| Scale: {result['scale_factor']}")
        for shock_name, value in result["shocks"].items():
            if value != 0:
                direction = "▲" if value > 0 else "▼"
                print(f"  {direction} {shock_name:<25} {value:+.4f} "
                      f"({value*100:+.2f}%)")
        print(f"  {result['description']}")
