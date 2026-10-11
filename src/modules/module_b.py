"""
Module B: strategic, event-driven stress testing of a synthetic wholesale
banking portfolio (loans, bonds, equities, rate and credit derivatives).

Subscribes to Event Classification + Impact Score from the NLP engine.

Trigger rule (case study: "e.g. Geopolitical with Impact Score > 7"):
  impact_score >= 7.0 AND
    (systemic risk-off class with sentiment below +0.30
     OR clearly negative company-specific news, sentiment <= -0.20)
so good news (a defence contract win, a product launch, a takeover premium)
never launches a "stress" test, and "Other" (non-events) never does.

Two modes:
  baseline   (default) every scenario is a what-if applied to the original
             book. This is how bank stress tests work: scenarios are
             hypothetical, not realised losses.
  cumulative scenarios compound (useful to show a sequence of shocks).
"""

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone

from src.config import RISK_OFF_CLASSES, data_path
from src.risk.loss_attribution import attribute_losses
from src.risk.risk_graph import traverse
from src.risk.stress_tester import run_stress_test

STRESS_TRIGGER_IMPACT = 7.0
NEGATIVE_SENTIMENT_TRIGGER = -0.20   # company-specific news must be this negative
POSITIVE_SENTIMENT_VETO = 0.30       # systemic news this positive is not a stress event
MODES = ("baseline", "cumulative")
MAX_HISTORY = 200


def _asset_type_totals(positions: list[dict], key: str) -> dict:
    totals: dict[str, float] = {}
    for p in positions:
        totals[p["type"]] = totals.get(p["type"], 0.0) + float(p[key])
    return {k: round(v, 2) for k, v in totals.items()}


class PortfolioStressTester:
    def __init__(self):
        base = json.loads(data_path("synthetic_portfolio.json").read_text())
        self.base_positions = deepcopy(base["positions"])
        self.positions = deepcopy(base["positions"])  # cumulative-mode state
        self.initial_value = float(base["total_value"])
        self.current_value = self.initial_value
        self.portfolio_name = base.get("portfolio_name", "Synthetic Wholesale Banking Portfolio")
        self.scenario_history: list[dict] = []
        self.last_result: dict = {}
        self._lock = threading.Lock()

    # ── Trigger ──────────────────────────────────────────────
    def should_trigger(
        self,
        impact_score: float,
        event_class: str | None = None,
        sentiment_score: float | None = None,
    ) -> bool:
        if float(impact_score) < STRESS_TRIGGER_IMPACT:
            return False
        if event_class is None:
            return True
        if event_class == "Other":
            return False
        if event_class in RISK_OFF_CLASSES:
            return sentiment_score is None or float(sentiment_score) < POSITIVE_SENTIMENT_VETO
        return sentiment_score is not None and float(sentiment_score) <= NEGATIVE_SENTIMENT_TRIGGER

    # ── Scenario ─────────────────────────────────────────────
    def run_scenario(
        self,
        event_class: str,
        impact_score: float,
        affected_tickers: list,
        affected_sectors: list,
        headline: str = "",
        sentiment_score: float | None = None,
        mode: str = "baseline",
        trigger: str = "manual",
    ) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not affected_sectors:
            affected_sectors = traverse(event_class, affected_tickers)["affected_sectors"]

        with self._lock:
            source_positions = self.positions if mode == "cumulative" else self.base_positions
            book_value = sum(float(p["value"]) for p in source_positions)
            portfolio = {
                "portfolio_name": self.portfolio_name,
                "total_value": book_value,
                "currency": "USD",
                "positions": deepcopy(source_positions),
            }
            stress = run_stress_test(
                affected_tickers=affected_tickers,
                affected_sectors=affected_sectors,
                event_class=event_class,
                impact_score=impact_score,
                portfolio=portfolio,
                sentiment_score=sentiment_score,
            )
            attribution = attribute_losses(stress)

            if mode == "cumulative":
                lookup = {p["id"]: p for p in self.positions}
                for result in stress["positions"]:
                    if result["id"] in lookup:
                        lookup[result["id"]]["value"] = result["value_after"]
                self.current_value = sum(float(p["value"]) for p in self.positions)

            scenario = {
                "scenario_id": len(self.scenario_history) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "trigger": trigger,
                "headline": (headline or "")[:200],
                "event_class": event_class,
                "impact_score": float(impact_score),
                "sentiment_score": sentiment_score,
                "affected_tickers": list(affected_tickers),
                "affected_sectors": list(affected_sectors),
                "portfolio_before": stress["portfolio_before"],
                "portfolio_after": stress["portfolio_after"],
                "total_loss": stress["total_loss"],
                "loss_pct": stress["loss_pct"],
                "gross_loss": attribution["gross_loss"],
                "gross_gain": attribution["gross_gain"],
                "shocks_applied": stress["shocks_applied"],
                "shock_direction": stress["shock_direction"],
                "scale_factor": stress["scale_factor"],
                "scenario_description": stress["scenario_description"],
                "exposure_pct": stress["exposure_pct"],
                "positions": stress["positions"],
                "value_by_asset_type_before": _asset_type_totals(stress["positions"], "value_before"),
                "value_by_asset_type_after": _asset_type_totals(stress["positions"], "value_after"),
                "by_sector": attribution["by_sector"],
                "by_asset_type": attribution["by_asset_type"],
                "by_exposure_type": attribution["by_exposure_type"],
                "top_positions": attribution["top_positions"],
                "hedges": attribution["hedges"],
                "summary_sentence": attribution["summary_sentence"],
            }
            self.scenario_history.append(scenario)
            if len(self.scenario_history) > MAX_HISTORY:
                self.scenario_history = self.scenario_history[-MAX_HISTORY:]
            self.last_result = scenario
            return deepcopy(scenario)

    # ── Read models ──────────────────────────────────────────
    def get_portfolio_summary(self) -> dict:
        history = self.scenario_history
        worst = min(history, key=lambda s: s["total_loss"]) if history else {}
        cumulative_change = self.current_value - self.initial_value
        return {
            "portfolio_name": self.portfolio_name,
            "initial_value": self.initial_value,
            "current_value": round(self.current_value, 2),
            "cumulative_change": round(cumulative_change, 2),
            "cumulative_change_pct": round(cumulative_change / self.initial_value * 100, 3),
            # Backwards-compatible names (cumulative mode).
            "total_loss": round(cumulative_change, 2),
            "total_loss_pct": round(cumulative_change / self.initial_value * 100, 3),
            "scenario_count": len(history),
            "last_event_class": self.last_result.get("event_class", "None"),
            "last_impact": self.last_result.get("impact_score", 0),
            "last_loss": self.last_result.get("total_loss", 0.0),
            "last_loss_pct": self.last_result.get("loss_pct", 0.0),
            "worst_loss": worst.get("total_loss", 0.0),
            "worst_loss_pct": worst.get("loss_pct", 0.0),
            "worst_event_class": worst.get("event_class", "None"),
        }

    def get_history(self) -> list:
        return [
            {
                "scenario_id": s["scenario_id"],
                "timestamp": s["timestamp"],
                "mode": s["mode"],
                "trigger": s["trigger"],
                "headline": s["headline"],
                "event_class": s["event_class"],
                "impact_score": s["impact_score"],
                "total_loss": s["total_loss"],
                "loss_pct": s["loss_pct"],
                "summary": s["summary_sentence"],
            }
            for s in self.scenario_history
        ]

    def get_positions(self, mode: str = "baseline") -> list:
        return deepcopy(self.positions if mode == "cumulative" else self.base_positions)

    def reset(self):
        self.__init__()


stress_tester = PortfolioStressTester()
