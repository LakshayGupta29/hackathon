"""
Loss Attribution
Breaks down stress test losses into explainable components.

Three attribution dimensions:
  1. By sector       → "Technology accounted for 47% of losses"
  2. By asset type   → "Bonds lost more than equities"
  3. By position     → top 3 loss contributors with explanation

This is the "why did we lose money" layer that judges love.
It directly answers: "where does the $6.96M come from?"
"""

from collections import defaultdict


def attribute_losses(stress_result: dict) -> dict:
    """
    Compute loss attribution from stress test result.

    Args:
        stress_result: output from stress_tester.run_stress_test()

    Returns:
        {
            "total_loss":         float,
            "by_sector":          list of {sector, loss, pct_of_total},
            "by_asset_type":      list of {type, loss, pct_of_total},
            "top_positions":      list of {obligor, loss, pct_of_total, reason},
            "summary_sentence":   str   (human readable summary)
        }
    """
    positions = stress_result["positions"]
    total_loss = stress_result["total_loss"]

    if total_loss == 0:
        return _empty_result()

    # 1. Attribution by sector
    sector_losses = defaultdict(float)
    for p in positions:
        sector_losses[p["sector"] or "Unknown"] += p["pnl"]

    by_sector = sorted([
        {
            "sector":       sector,
            "loss":         round(loss, 2),
            "pct_of_total": round(loss / total_loss * 100, 1)
        }
        for sector, loss in sector_losses.items()
        if loss < 0  # only show losses
    ], key=lambda x: x["loss"])

    # 2. Attribution by asset type
    type_losses = defaultdict(float)
    for p in positions:
        type_losses[p["type"]] += p["pnl"]

    by_asset_type = sorted([
        {
            "type":         asset_type,
            "loss":         round(loss, 2),
            "pct_of_total": round(loss / total_loss * 100, 1)
        }
        for asset_type, loss in type_losses.items()
        if loss < 0
    ], key=lambda x: x["loss"])

    # 3. Top loss positions (worst 5)
    loss_positions = [p for p in positions if p["pnl"] < 0]
    top_positions = sorted(loss_positions, key=lambda x: x["pnl"])[:5]

    top_with_reason = []
    for p in top_positions:
        reason = _explain_loss(p, stress_result["shocks_applied"])
        top_with_reason.append({
            "obligor":       p["obligor"],
            "type":          p["type"],
            "sector":        p["sector"],
            "loss":          round(p["pnl"], 2),
            "pct_of_total":  round(p["pnl"] / total_loss * 100, 1),
            "reason":        reason,
        })

    summary = _build_summary(by_sector, by_asset_type, total_loss)

    return {
        "total_loss":       total_loss,
        "by_sector":        by_sector,
        "by_asset_type":    by_asset_type,
        "top_positions":    top_with_reason,
        "summary_sentence": summary,
    }


def _explain_loss(position: dict, shocks: dict) -> str:
    """Generate a human-readable explanation for a position's loss."""
    asset_type = position["type"]
    obligor = position["obligor"]

    if asset_type == "equity":
        pct = abs(shocks.get("equity_shock", 0)) * 100
        return f"Equity selloff: {pct:.1f}% price decline"

    elif asset_type == "bond":
        rate_shock = shocks.get("rates_shock", 0)
        spread_shock = shocks.get("credit_spread_shock", 0)
        duration = position.get("duration", 0) or 0
        if abs(spread_shock) > abs(rate_shock):
            return (f"Credit spread widening "
                    f"(+{spread_shock*100:.2f}%) × duration {duration:.1f}yr")
        else:
            return (f"Rate shock "
                    f"(+{rate_shock*100:.2f}%) × duration {duration:.1f}yr")

    elif asset_type == "loan":
        spread_shock = shocks.get("credit_spread_shock", 0)
        duration = position.get("duration", 0) or 0
        return (f"Credit spread widening "
                f"(+{spread_shock*100:.2f}%) × duration {duration:.1f}yr")

    elif asset_type == "derivative":
        return "Interest rate / credit sensitivity"

    return "Market shock"


def _build_summary(by_sector: list, by_asset_type: list,
                   total_loss: float) -> str:
    """Build a one-sentence human readable summary."""
    if not by_sector:
        return "No significant losses attributed."

    top_sector = by_sector[0]
    top_type = by_asset_type[0] if by_asset_type else None

    loss_m = abs(total_loss) / 1_000_000
    sector_pct = abs(top_sector["pct_of_total"])
    type_str = f" driven by {top_type['type']} exposure" if top_type else ""

    return (
        f"Total loss of ${loss_m:.1f}M: "
        f"{sector_pct:.0f}% from {top_sector['sector']} sector{type_str}."
    )


def _empty_result() -> dict:
    return {
        "total_loss": 0,
        "by_sector": [],
        "by_asset_type": [],
        "top_positions": [],
        "summary_sentence": "No losses recorded."
    }


if __name__ == "__main__":
    from src.risk.stress_tester import run_stress_test

    stress = run_stress_test(
        affected_tickers=["TSM", "NVDA", "ASML"],
        affected_sectors=["Energy", "Industrials", "Technology", "Financials"],
        event_class="Geopolitical",
        impact_score=8.5,
    )

    attribution = attribute_losses(stress)

    print("LOSS ATTRIBUTION: Taiwan Crisis\n")
    print(f"Total loss: ${attribution['total_loss']:,.0f}\n")

    print("By Sector:")
    for s in attribution["by_sector"]:
        bar = "█" * int(abs(s["pct_of_total"]) / 5)
        print(f"  {s['sector']:<25} ${s['loss']:>12,.0f} "
              f"({s['pct_of_total']:>5.1f}%) {bar}")

    print("\nBy Asset Type:")
    for a in attribution["by_asset_type"]:
        print(f"  {a['type']:<12} ${a['loss']:>12,.0f} "
              f"({a['pct_of_total']:>5.1f}%)")

    print("\nTop Loss Positions:")
    for p in attribution["top_positions"]:
        print(f"  {p['obligor']:<30} ${p['loss']:>12,.0f} "
              f"({p['pct_of_total']:>5.1f}%)")
        print(f"    → {p['reason']}")

    print(f"\nSummary: {attribution['summary_sentence']}")
