"""Explainable loss attribution for a stress scenario."""

from collections import defaultdict

from src.risk.stress_tester import rating_beta


def _breakdown(positions: list[dict], key: str, gross_loss: float) -> list[dict]:
    losses = defaultdict(float)
    for p in positions:
        losses[p.get(key) or "Unknown"] += -p["pnl"]
    rows = [
        {
            key: name,
            "loss": round(loss, 2),
            "pct_of_gross_loss": round(loss / gross_loss * 100, 1),
            # Kept for older dashboard/clients.
            "pct_of_total": round(loss / gross_loss * 100, 1),
        }
        for name, loss in losses.items()
    ]
    return sorted(rows, key=lambda r: r["loss"], reverse=True)


def attribute_losses(stress_result: dict) -> dict:
    positions = stress_result["positions"]
    net_pnl = float(stress_result["total_loss"])
    loss_positions = [p for p in positions if p["pnl"] < 0]
    gain_positions = [p for p in positions if p["pnl"] > 0]
    gross_loss = -sum(p["pnl"] for p in loss_positions)
    gross_gain = sum(p["pnl"] for p in gain_positions)

    if gross_loss <= 0:
        return {
            "total_loss": round(net_pnl, 2),
            "gross_loss": 0.0,
            "gross_gain": round(gross_gain, 2),
            "by_sector": [],
            "by_asset_type": [],
            "by_exposure_type": [],
            "top_positions": [],
            "hedges": _hedges(gain_positions),
            "summary_sentence": (
                f"The scenario produced a net gain of ${net_pnl:,.0f}."
                if net_pnl > 0 else "No position generated a loss."
            ),
        }

    by_sector = _breakdown(loss_positions, "sector", gross_loss)
    by_asset_type = _breakdown(loss_positions, "type", gross_loss)
    by_exposure_type = _breakdown(loss_positions, "exposure_type", gross_loss)

    top_positions = [
        {
            "obligor": p["obligor"],
            "type": p["type"],
            "sector": p["sector"],
            "exposure_type": p["exposure_type"],
            "loss": round(-p["pnl"], 2),
            "pct_of_gross_loss": round(-p["pnl"] / gross_loss * 100, 1),
            "reason": _explain(p, stress_result["shocks_applied"]),
        }
        for p in sorted(loss_positions, key=lambda x: x["pnl"])[:5]
    ]

    if net_pnl < 0:
        summary = (
            f"Net loss ${-net_pnl:,.0f} ({stress_result['loss_pct']:+.2f}%): gross losses "
            f"${gross_loss:,.0f}, hedges/gains offset ${gross_gain:,.0f}. "
            f"Largest contributor: {by_sector[0]['sector']} "
            f"({by_sector[0]['pct_of_gross_loss']:.0f}% of gross loss)."
        )
    else:
        summary = (
            f"Gross losses ${gross_loss:,.0f} were more than offset by "
            f"${gross_gain:,.0f} of gains (net +${net_pnl:,.0f})."
        )

    return {
        "total_loss": round(net_pnl, 2),
        "gross_loss": round(gross_loss, 2),
        "gross_gain": round(gross_gain, 2),
        "by_sector": by_sector,
        "by_asset_type": by_asset_type,
        "by_exposure_type": by_exposure_type,
        "top_positions": top_positions,
        "hedges": _hedges(gain_positions),
        "summary_sentence": summary,
    }


def _hedges(gain_positions: list[dict]) -> list[dict]:
    return [
        {"obligor": p["obligor"], "type": p["type"], "gain": round(p["pnl"], 2)}
        for p in sorted(gain_positions, key=lambda x: -x["pnl"])
    ]


def _explain(p: dict, shocks: dict) -> str:
    m = p.get("exposure_multiplier", 1.0)
    exposure = f"{p.get('exposure_type', 'n/a')} exposure x{m:.2f}"
    if p["type"] == "equity":
        return f"Equity shock {shocks.get('equity_shock', 0) * 100:+.1f}% ({exposure})"
    if p["type"] in {"bond", "loan"}:
        spread_bp = shocks.get("credit_spread_shock", 0) * 10_000
        beta = rating_beta(p.get("credit_rating"))
        sd = p.get("spread_duration") or p.get("duration") or 0
        text = (f"Spread {spread_bp:+.0f}bp x rating beta {beta:.2f} "
                f"({p.get('credit_rating') or 'NR'}) x spread duration {sd:.1f}y, {exposure}")
        if p["type"] == "bond":
            rate_bp = shocks.get("rates_shock", 0) * 10_000
            text = f"Rates {rate_bp:+.0f}bp x duration {p.get('duration', 0):.1f}y; " + text
        return text
    if p["type"] == "derivative":
        return "Risk-factor sensitivity (DV01/CS01) to the stressed rate or spread move"
    return "Stress factor applied"
