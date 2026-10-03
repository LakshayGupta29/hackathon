"""
Module B: Strategic Portfolio Stress Tester
Subscribes to Event Classification + Impact Score from the NLP engine.
Triggers stress tests on high-impact events and tracks portfolio
state over time.

This module wraps the risk engine components into a stateful
portfolio manager that:
  1. Maintains portfolio value over time
  2. Triggers stress tests on high-impact events (impact > 7)
  3. Keeps history of all stress scenarios
  4. Provides before/after comparison for dashboard
"""

import json
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from src.risk.stress_tester    import run_stress_test
from src.risk.loss_attribution import attribute_losses
from src.risk.risk_graph       import traverse

# Threshold to trigger a stress test
STRESS_TRIGGER_IMPACT = 5.0

# Starting portfolio value
INITIAL_VALUE = 100_000_000


class PortfolioStressTester:
    """
    Stateful stress tester. Tracks portfolio through multiple events.
    One instance shared across the API session.
    """

    def __init__(self):
        self.current_value  = float(INITIAL_VALUE)
        self.initial_value  = float(INITIAL_VALUE)
        self.scenario_history: list = []
        self.last_result:     dict  = {}

    def should_trigger(self, impact_score: float) -> bool:
        """Only trigger stress test for high-impact events."""
        return impact_score >= STRESS_TRIGGER_IMPACT

    def run_scenario(
        self,
        event_class:      str,
        impact_score:     float,
        affected_tickers: list,
        affected_sectors: list,
        headline:         str = "",
    ) -> dict:
        """
        Run a stress scenario and update portfolio state.

        Returns full scenario result including attribution.
        """
        # Get sectors from risk graph if not provided
        if not affected_sectors:
            graph = traverse(event_class, affected_tickers)
            affected_sectors = graph["affected_sectors"]

        # Run stress test (always uses full portfolio value)
        stress = run_stress_test(
            affected_tickers=affected_tickers,
            affected_sectors=affected_sectors,
            event_class=event_class,
            impact_score=impact_score,
        )

        # Attribution breakdown
        attribution = attribute_losses(stress)

        # Build full scenario record
        scenario = {
            "scenario_id":       len(self.scenario_history) + 1,
            "timestamp":         datetime.utcnow().isoformat(),
            "headline":          headline[:200],
            "event_class":       event_class,
            "impact_score":      impact_score,
            "portfolio_before":  stress["portfolio_before"],
            "portfolio_after":   stress["portfolio_after"],
            "total_loss":        stress["total_loss"],
            "loss_pct":          stress["loss_pct"],
            "shocks_applied":    stress["shocks_applied"],
            "positions":         stress["positions"],
            "by_sector":         attribution["by_sector"],
            "by_asset_type":     attribution["by_asset_type"],
            "top_positions":     attribution["top_positions"],
            "summary_sentence":  attribution["summary_sentence"],
        }

        self.scenario_history.append(scenario)
        self.last_result = scenario
        self.current_value = stress["portfolio_after"]

        return scenario

    def get_portfolio_summary(self) -> dict:
        """Current portfolio state for dashboard."""
        total_loss = self.current_value - self.initial_value
        return {
            "initial_value":    self.initial_value,
            "current_value":    round(self.current_value, 2),
            "total_loss":       round(total_loss, 2),
            "total_loss_pct":   round(total_loss / self.initial_value * 100, 2),
            "scenario_count":   len(self.scenario_history),
            "last_event_class": self.last_result.get("event_class", "None"),
            "last_impact":      self.last_result.get("impact_score", 0),
        }

    def get_history(self) -> list:
        """All scenario records for timeline chart."""
        return [
            {
                "scenario_id":  s["scenario_id"],
                "timestamp":    s["timestamp"],
                "event_class":  s["event_class"],
                "impact_score": s["impact_score"],
                "loss_pct":     s["loss_pct"],
                "summary":      s["summary_sentence"],
            }
            for s in self.scenario_history
        ]

    def reset(self):
        """Reset for new demo session."""
        self.__init__()


# Global instance shared with API
stress_tester = PortfolioStressTester()


if __name__ == "__main__":
    print("=" * 50)
    print("MODULE B: STRATEGIC PORTFOLIO STRESS TESTER")
    print("=" * 50)

    tester = PortfolioStressTester()

    # Simulate 3 events arriving over time
    events = [
        {
            "event_class":      "Geopolitical",
            "impact_score":     8.5,
            "affected_tickers": ["TSM", "NVDA", "ASML"],
            "affected_sectors": ["Technology", "Energy", "Financials"],
            "headline":         "Taiwan military crisis escalates",
        },
        {
            "event_class":      "Credit Event",
            "impact_score":     7.2,
            "affected_tickers": ["JPM", "GS"],
            "affected_sectors": ["Financials", "Energy"],
            "headline":         "Major bank credit downgrade",
        },
        {
            "event_class":      "Macroeconomic",
            "impact_score":     6.5,  # below threshold, won't trigger
            "affected_tickers": [],
            "affected_sectors": ["Financials"],
            "headline":         "Fed signals rate pause",
        },
    ]

    print("\nProcessing events:\n")
    for evt in events:
        if tester.should_trigger(evt["impact_score"]):
            print(f"  ⚡ TRIGGER: {evt['headline'][:50]}")
            result = tester.run_scenario(**evt)
            print(f"     Loss: ${result['total_loss']:,.0f} "
                  f"({result['loss_pct']}%)")
            print(f"     {result['summary_sentence']}")
        else:
            print(f"  ○  SKIP (impact {evt['impact_score']} "
                  f"< {STRESS_TRIGGER_IMPACT}): {evt['headline'][:40]}")

    print("\nPortfolio Summary:")
    summary = tester.get_portfolio_summary()
    print(f"  Initial value:  ${summary['initial_value']:>15,.2f}")
    print(f"  Current value:  ${summary['current_value']:>15,.2f}")
    print(f"  Total loss:     ${summary['total_loss']:>15,.2f} "
          f"({summary['total_loss_pct']}%)")
    print(f"  Scenarios run:  {summary['scenario_count']}")

    print("\nScenario History:")
    for s in tester.get_history():
        print(f"  #{s['scenario_id']} [{s['event_class']:<20}] "
              f"impact={s['impact_score']} loss={s['loss_pct']}%")

    print("\n✅ Module B complete")
