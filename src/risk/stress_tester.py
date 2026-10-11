"""
Scenario stress testing with first-order sensitivities.

Per asset type (m = exposure multiplier, beta = rating spread beta):
  equity     dV = V * equity_shock * m   (+ V * oil_shock * 0.4 for Energy)
  bond       dV = V * (-D * rates_shock - SD * credit_spread_shock * m * beta)
  loan       dV = V * (-SD * credit_spread_shock * m * beta)   (floating rate)
  IRS        dV = DV01 * rates_shock_bp * (-1 receive-fixed / +1 pay-fixed)
  CDS        dV = CS01 * spread_shock_bp * (+1 protection buyer / -1 seller)

Interest-rate moves are market-wide, so every bond and swap is repriced.
Event-specific equity and spread shocks spill over to unrelated names at
x0.50 for systemic events and only x0.10 for company-specific events.
Credit spreads widen more for lower-rated obligors (rating beta), the way
spreads behave in real sell-offs.
"""

from src.config import RISK_OFF_CLASSES
from src.risk.exposure_calc import EXPOSURE_MULTIPLIER, IDIOSYNCRATIC_SPILLOVER, calculate_exposure
from src.risk.shock_generator import generate_shocks

# Relative credit-spread sensitivity by rating (BBB = 1.0).
RATING_SPREAD_BETA = {
    "AAA": 0.60, "AA": 0.75, "A": 0.90, "BBB": 1.00,
    "BB": 1.40, "B": 1.80, "CCC": 2.50,
}
OIL_EQUITY_BETA = 0.40


def rating_beta(rating: str | None) -> float:
    if not rating:
        return 1.0
    base = str(rating).upper().rstrip("+-")
    return RATING_SPREAD_BETA.get(base, 1.0)


def _equity_pnl(pos: dict, shocks: dict) -> float:
    pnl = pos["value"] * shocks["equity_shock"] * pos["exposure_multiplier"]
    if pos.get("sector") == "Energy":
        pnl += pos["value"] * shocks.get("oil_shock", 0.0) * OIL_EQUITY_BETA
    return pnl


def _bond_pnl(pos: dict, shocks: dict) -> float:
    duration = float(pos.get("duration") or 0)
    spread_duration = float(pos.get("spread_duration") or duration)
    spread_move = shocks["credit_spread_shock"] * pos["exposure_multiplier"] * rating_beta(pos.get("credit_rating"))
    return pos["value"] * (-duration * shocks["rates_shock"] - spread_duration * spread_move)


def _loan_pnl(pos: dict, shocks: dict) -> float:
    spread_duration = float(pos.get("spread_duration") or pos.get("duration") or 0)
    spread_move = shocks["credit_spread_shock"] * pos["exposure_multiplier"] * rating_beta(pos.get("credit_rating"))
    return pos["value"] * (-spread_duration * spread_move)


def _derivative_pnl(pos: dict, shocks: dict) -> float:
    obligor = str(pos.get("obligor", "")).lower()
    direction = str(pos.get("direction") or "").lower()

    if "rate swap" in obligor:
        sign = -1.0 if direction in {"receive_fixed", "receiver"} else 1.0
        dv01 = pos.get("rate_dv01")
        if dv01 is not None:
            return float(dv01) * shocks["rates_shock"] * 10_000.0 * sign
        return pos["value"] * float(pos.get("duration") or 0) * shocks["rates_shock"] * sign

    if "credit default" in obligor:
        sign = 1.0 if direction in {"protection_buyer", "buyer"} else -1.0
        cs01 = pos.get("spread_cs01")
        if cs01 is not None:
            return float(cs01) * shocks["credit_spread_shock"] * 10_000.0 * sign
        return pos["value"] * float(pos.get("duration") or 0) * shocks["credit_spread_shock"] * sign

    return 0.0


PNL_FUNCTIONS = {
    "equity": _equity_pnl,
    "bond": _bond_pnl,
    "loan": _loan_pnl,
    "derivative": _derivative_pnl,
}


def run_stress_test(
    affected_tickers: list[str],
    affected_sectors: list[str],
    event_class: str,
    impact_score: float,
    portfolio: dict | None = None,
    sentiment_score: float | None = None,
) -> dict:
    """Apply one scenario to a supplied portfolio state (all positions)."""
    systemic = EXPOSURE_MULTIPLIER["systemic"] if event_class in RISK_OFF_CLASSES else IDIOSYNCRATIC_SPILLOVER
    exposure = calculate_exposure(
        affected_tickers, affected_sectors, portfolio=portfolio, systemic_multiplier=systemic
    )
    shock_info = generate_shocks(event_class, impact_score, sentiment_score)
    shocks = shock_info["shocks"]

    portfolio_before = exposure["total_portfolio_value"]
    position_results = []
    for pos in exposure["positions"]:
        pnl = PNL_FUNCTIONS.get(pos["type"], lambda *_: 0.0)(pos, shocks)
        value_after = pos["value"] + pnl
        if pos["type"] != "derivative":
            value_after = max(0.0, value_after)
            pnl = value_after - pos["value"]
        position_results.append({
            "id": pos["id"],
            "obligor": pos["obligor"],
            "ticker": pos["ticker"],
            "sector": pos["sector"],
            "type": pos["type"],
            "credit_rating": pos.get("credit_rating"),
            "value_before": round(pos["value"], 2),
            "value_after": round(value_after, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / pos["value"] * 100, 3) if pos["value"] else 0.0,
            "exposure_type": pos["exposure_type"],
            "exposure_multiplier": pos["exposure_multiplier"],
            "duration": pos.get("duration", 0),
            "spread_duration": pos.get("spread_duration", 0),
        })

    total_pnl = sum(p["pnl"] for p in position_results)
    return {
        "portfolio_before": round(portfolio_before, 2),
        "portfolio_after": round(portfolio_before + total_pnl, 2),
        "total_loss": round(total_pnl, 2),
        "loss_pct": round(total_pnl / portfolio_before * 100, 3) if portfolio_before else 0.0,
        "shocks_applied": shocks,
        "shock_direction": shock_info["direction"],
        "scale_factor": shock_info["scale_factor"],
        "scenario_description": shock_info["description"],
        "positions": position_results,
        "exposure_pct": exposure["exposure_pct"],
        "exposed_value": exposure["exposed_value"],
    }
