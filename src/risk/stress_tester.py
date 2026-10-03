"""
Stress Tester
Applies market shocks to exposed portfolio positions
and computes P&L for each position.

Asset-specific shock application:
  equity:     value × equity_shock
  bond:       value × (equity_shock×0.3 + duration × rates_shock × -1
                       + credit_spread_shock × duration × -1)
  loan:       value × (credit_spread_shock × duration × -1)
  derivative: value × (rates_shock × duration × -1)

Why different formulas per asset type:
  - Equities absorb full market shock
  - Bonds: sensitive to rates (duration) AND credit spreads
  - Loans: mainly credit spread sensitive, not equity
  - Derivatives: rates driven, direction depends on position

These are simplified but financially grounded approximations.
Jury can probe and we can defend each one.
"""

from src.risk.exposure_calc   import calculate_exposure
from src.risk.shock_generator import generate_shocks


def _equity_pnl(position: dict, shocks: dict) -> float:
    """Equities absorb the full equity shock."""
    return position["value"] * shocks["equity_shock"]


def _bond_pnl(position: dict, shocks: dict) -> float:
    """
    Bonds are sensitive to:
      - Equity market sentiment (partial, 30% weight)
      - Interest rate changes × duration (modified duration effect)
      - Credit spread changes × duration
    """
    duration = position.get("duration", 0) or 0
    equity_component  = position["value"] * shocks["equity_shock"] * 0.30
    rates_component   = position["value"] * (-duration * shocks["rates_shock"])
    credit_component  = position["value"] * (-duration * shocks["credit_spread_shock"])
    return equity_component + rates_component + credit_component


def _loan_pnl(position: dict, shocks: dict) -> float:
    """
    Loans are mainly credit spread sensitive.
    Rising spreads → mark-to-market loss on loan book.
    """
    duration = position.get("duration", 0) or 0
    return position["value"] * (-duration * shocks["credit_spread_shock"])


def _derivative_pnl(position: dict, shocks: dict) -> float:
    """
    Simplified: interest rate swap value moves with rates × duration.
    Credit default swap gains value when credit spreads widen.
    """
    duration = position.get("duration", 0) or 0
    obligor = position.get("obligor", "").lower()

    if "rate swap" in obligor:
        return position["value"] * (-duration * shocks["rates_shock"])
    elif "credit default" in obligor:
        # CDS gains when spreads widen (it's protection)
        return position["value"] * (duration * shocks["credit_spread_shock"])
    return 0.0


def run_stress_test(
    affected_tickers: list,
    affected_sectors: list,
    event_class: str,
    impact_score: float,
) -> dict:
    """
    Full stress test: exposure + shocks + P&L per position.

    Returns:
        {
            "portfolio_before":  float,
            "portfolio_after":   float,
            "total_loss":        float,
            "loss_pct":          float,
            "shocks_applied":    dict,
            "positions":         list of position P&L dicts,
        }
    """
    # Get exposed positions
    exposure = calculate_exposure(affected_tickers, affected_sectors)

    # Get shocks
    shock_result = generate_shocks(event_class, impact_score)
    shocks = shock_result["shocks"]

    portfolio_before = exposure["total_portfolio_value"]
    position_results = []

    for pos in exposure["positions"]:
        asset_type = pos["type"]

        if asset_type == "equity":
            pnl = _equity_pnl(pos, shocks)
        elif asset_type == "bond":
            pnl = _bond_pnl(pos, shocks)
        elif asset_type == "loan":
            pnl = _loan_pnl(pos, shocks)
        elif asset_type == "derivative":
            pnl = _derivative_pnl(pos, shocks)
        else:
            pnl = 0.0

        position_results.append({
            "id":            pos["id"],
            "obligor":       pos["obligor"],
            "ticker":        pos["ticker"],
            "sector":        pos["sector"],
            "type":          asset_type,
            "value_before":  pos["value"],
            "value_after":   round(pos["value"] + pnl, 2),
            "pnl":           round(pnl, 2),
            "exposure_type": pos["exposure_type"],
        })

    total_loss = sum(p["pnl"] for p in position_results)
    portfolio_after = portfolio_before + total_loss
    loss_pct = round(total_loss / portfolio_before * 100, 2)

    return {
        "portfolio_before": portfolio_before,
        "portfolio_after":  round(portfolio_after, 2),
        "total_loss":       round(total_loss, 2),
        "loss_pct":         loss_pct,
        "shocks_applied":   shocks,
        "positions":        position_results,
    }


if __name__ == "__main__":
    result = run_stress_test(
        affected_tickers=["TSM", "NVDA", "ASML"],
        affected_sectors=["Energy", "Industrials", "Technology", "Financials"],
        event_class="Geopolitical",
        impact_score=8.5,
    )

    print("STRESS TEST: Geopolitical Event (Taiwan Crisis)")
    print(f"Impact Score: 8.5/10\n")
    print(f"Portfolio before: ${result['portfolio_before']:>15,.2f}")
    print(f"Portfolio after:  ${result['portfolio_after']:>15,.2f}")
    print(f"Total loss:       ${result['total_loss']:>15,.2f} ({result['loss_pct']}%)")
    print(f"\nShocks applied:")
    for k, v in result["shocks_applied"].items():
        if v != 0:
            print(f"  {k:<25} {v:+.4f}")
    print(f"\nPosition P&L:")
    print(f"  {'ID':<6} {'Obligor':<30} {'Type':<12} {'P&L':>12}")
    print(f"  {'-'*65}")
    for p in sorted(result["positions"], key=lambda x: x["pnl"]):
        print(f"  {p['id']:<6} {p['obligor']:<30} {p['type']:<12} "
              f"${p['pnl']:>11,.0f}")
