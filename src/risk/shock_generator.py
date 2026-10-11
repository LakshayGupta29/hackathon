"""
Shock Generator.

Converts (event class, impact score, optional sentiment) into market shocks.

Base shocks live in data/shock_templates.json. The impact score scales the
magnitude linearly: impact 10 -> 100% of the template, impact 5 -> 50%,
floored at 10%.

Systemic classes (Geopolitical, Macroeconomic, Credit Event) are adverse by
construction - that is what a stress scenario is. For idiosyncratic classes
(M&A, Product Launch, Earnings/Corporate) the direction follows the
headline's sentiment when it is known: bad news produces an adverse shock,
good news a favourable one.

Shock keys:
  equity_shock        relative change in equity prices
  credit_spread_shock absolute change in credit spreads (0.01 = 100 bp)
  rates_shock         absolute change in interest rates (0.01 = 100 bp)
  oil_shock           relative change in oil prices
  fx_shock            relative change in USD vs basket (reported only; the
                      synthetic portfolio is USD-denominated)
"""

import json
from functools import lru_cache

from src.config import RISK_OFF_CLASSES, data_path

SHOCK_KEYS = ("equity_shock", "credit_spread_shock", "rates_shock", "oil_shock", "fx_shock")


@lru_cache(maxsize=1)
def _load_templates() -> dict:
    return json.loads(data_path("shock_templates.json").read_text())


def scale_factor(impact_score: float) -> float:
    return max(0.1, min(1.0, float(impact_score) / 10.0))


def _direction(event_class: str, sentiment_score: float | None) -> float:
    """+1 keeps the template sign, -1 flips it (favourable vs adverse)."""
    if event_class in RISK_OFF_CLASSES or sentiment_score is None:
        return 1.0
    template = _load_templates().get(event_class, {})
    template_is_adverse = template.get("equity_shock", 0.0) < 0
    headline_is_negative = float(sentiment_score) < 0
    return 1.0 if template_is_adverse == headline_is_negative else -1.0


def generate_shocks(
    event_class: str,
    impact_score: float,
    sentiment_score: float | None = None,
) -> dict:
    templates = _load_templates()
    template = templates.get(event_class, templates["Macroeconomic"])
    scale = scale_factor(impact_score)
    direction = _direction(event_class, sentiment_score)

    shocks = {
        key: round(float(template.get(key, 0.0)) * scale * direction, 6)
        for key in SHOCK_KEYS
    }
    return {
        "event_class": event_class,
        "impact_score": impact_score,
        "scale_factor": round(scale, 2),
        "direction": "adverse" if (direction > 0) == (template.get("equity_shock", 0) <= 0) else "favourable",
        "shocks": shocks,
        "description": template["description"],
    }


if __name__ == "__main__":
    for event_class, impact, sentiment in [
        ("Geopolitical", 8.5, -0.9),
        ("Geopolitical", 3.0, None),
        ("Credit Event", 7.5, -0.6),
        ("Earnings/Corporate", 7.2, -0.8),
        ("Earnings/Corporate", 7.2, 0.8),
        ("Merger/Acquisition", 4.0, None),
    ]:
        result = generate_shocks(event_class, impact, sentiment)
        print(f"\n{event_class} | impact {impact} | sentiment {sentiment} | "
              f"scale {result['scale_factor']} | {result['direction']}")
        for name, value in result["shocks"].items():
            if value:
                print(f"  {name:<22} {value * 100:+.2f}%")
