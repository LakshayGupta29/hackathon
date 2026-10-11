"""
Time-causal historical validation for the NLP risk engine.

The archive contains no publication timestamps, so evaluation deliberately
avoids event-day closes. Each headline is processed in chronological order
using only headline/source information. Market outcomes are measured from the
first trading-day open strictly after the event date to the close of T+1/T+3/T+5
post-event trading days. Results are observational, not causal estimates.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.config import EVENT_CLASSES, data_path, output_path
from src.engine.pipeline import process, reset_novelty_memory

SPLIT_FILES = {
    "tuning": "master_tuning_set.json",
    "holdout": "master_holdout_set.json",
}


def _load_split(split: str) -> list[dict]:
    return json.loads(data_path(SPLIT_FILES[split]).read_text())["events"]


def _load_all_events() -> list[dict]:
    events = _load_split("tuning") + _load_split("holdout")
    return sorted(events, key=lambda e: (e["date"], e["id"]))


def _download_prices(events: list[dict]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    import yfinance as yf

    tickers = sorted({ticker for event in events for ticker in event.get("affected_tickers", [])})
    if not tickers:
        return {}, {}

    event_dates = [pd.Timestamp(e["date"]).normalize() for e in events]
    start = min(event_dates) - pd.Timedelta(days=10)
    end = max(event_dates) + pd.Timedelta(days=20)

    raw = yf.download(
        tickers + ["SPY"],
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        return {}, {}

    if isinstance(raw.columns, pd.MultiIndex):
        opens = raw["Open"]
        closes = raw["Close"]
    else:
        opens = raw[["Open"]]
        closes = raw[["Close"]]

    opens.index = pd.DatetimeIndex(opens.index).tz_localize(None).normalize()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    return opens, closes


def _forward_return(
    opens: pd.Series,
    closes: pd.Series,
    event_date: str,
    horizon: int,
) -> float | None:
    """Return from next post-event session open to the close at horizon."""
    event_dt = pd.Timestamp(event_date).normalize()
    common = closes.index.intersection(opens.index)
    future_dates = common[common > event_dt]
    if len(future_dates) < horizon:
        return None
    base_date = future_dates[0]
    target_date = future_dates[horizon - 1]
    base_price = opens.get(base_date)
    target_price = closes.get(target_date)
    if pd.isna(base_price) or pd.isna(target_price) or float(base_price) == 0:
        return None
    return (float(target_price) - float(base_price)) / float(base_price)


def _get_event_outcome(event: dict, opens: dict[str, pd.Series], closes: dict[str, pd.Series]) -> dict:
    tickers = [t for t in event.get("affected_tickers", []) if t in opens and t in closes]
    rows = []

    for ticker in tickers:
        values = {}
        abnormal = {}
        for horizon in [1, 3, 5]:
            raw_move = _forward_return(opens[ticker], closes[ticker], event["date"], horizon)
            spy_move = _forward_return(opens["SPY"], closes["SPY"], event["date"], horizon) if "SPY" in opens else None
            key = f"t+{horizon}"
            if raw_move is not None:
                values[key] = round(raw_move * 100, 2)
            if raw_move is not None and spy_move is not None:
                abnormal[key] = round((raw_move - spy_move) * 100, 2)
        if values:
            rows.append({"ticker": ticker, "moves": values, "abnormal_moves": abnormal})

    event_mean = {}
    abnormal_mean = {}
    for horizon in ["t+1", "t+3", "t+5"]:
        raw_values = [r["moves"][horizon] for r in rows if horizon in r["moves"]]
        adj_values = [r["abnormal_moves"][horizon] for r in rows if horizon in r["abnormal_moves"]]
        if raw_values:
            event_mean[horizon] = round(float(np.mean(raw_values)), 2)
        if adj_values:
            abnormal_mean[horizon] = round(float(np.mean(adj_values)), 2)

    return {
        "tickers": rows,
        "event_mean_move": event_mean,
        "event_mean_abnormal_move": abnormal_mean,
    }


def _safe_corr(x: list[float], y: list[float], fn) -> dict:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"r": None, "p_value": None, "n": len(x)}
    r, p = fn(x, y)
    return {"r": round(float(r), 4), "p_value": round(float(p), 4), "n": len(x)}


def run_validation(target_split: str = "holdout") -> dict:
    if target_split not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {target_split}")

    all_events = _load_all_events()
    target_ids = {e["id"] for e in _load_split(target_split)}
    reset_novelty_memory()
    opens_raw, closes_raw = _download_prices(all_events)
    if not hasattr(opens_raw, "columns") or not hasattr(closes_raw, "columns"):
        raise RuntimeError("No price data returned by yfinance; check your internet connection")
    opens = {ticker: opens_raw[ticker].dropna() for ticker in opens_raw.columns}
    closes = {ticker: closes_raw[ticker].dropna() for ticker in closes_raw.columns}

    results = []
    for event in all_events:
        prediction = process(event["headline"], source=event.get("source", "news:unknown"))
        outcome = _get_event_outcome(event, opens, closes)
        results.append({
            "id": event["id"],
            "split": "holdout" if event["id"] in target_ids else "tuning",
            "date": event["date"],
            "headline": event["headline"],
            "source": event.get("source", ""),
            "expected_event_class": event["event_class_expected"],
            "predicted": {
                "sentiment_score": prediction["sentiment_score"],
                "sentiment_label": prediction["sentiment_label"],
                "event_class": prediction["event_class"],
                "impact_score": prediction["impact_score"],
                "novelty_score": prediction["novelty_score"],
                "detected_tickers": prediction["affected_tickers"],
            },
            "outcome": outcome,
        })

    scored = [r for r in results if r["id"] in target_ids]
    summary = _compute_summary(scored)
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_methodology": {
            "split": target_split,
            "events_processed_chronologically": True,
            "prediction_uses_future_prices": False,
            "price_data_source": "yfinance",
            "entry_rule": "first trading-day open strictly after event date",
            "outcome_horizons": ["t+1", "t+3", "t+5"],
            "primary_direction_measure": "sign of event-level mean T+5 abnormal return",
            "note": "No publication timestamps are available. Outcomes are forward-only observational measures, not causal effect estimates.",
        },
        "events": scored,
        "summary": summary,
    }

    output_path(f"validation_results_{target_split}.json").write_text(json.dumps(output, indent=2))
    outcome_rows = []
    for r in scored:
        outcome_rows.append({
            "id": r["id"],
            "date": r["date"],
            "sentiment_score": r["predicted"]["sentiment_score"],
            "impact_score": r["predicted"]["impact_score"],
            "novelty_score": r["predicted"]["novelty_score"],
            "event_class_pred": r["predicted"]["event_class"],
            "event_class_expected": r["expected_event_class"],
            "mean_t5_move_pct": r["outcome"]["event_mean_move"].get("t+5"),
            "mean_t5_abnormal_move_pct": r["outcome"]["event_mean_abnormal_move"].get("t+5"),
        })
    output_path(f"validation_outcomes_{target_split}.csv").write_text(pd.DataFrame(outcome_rows).to_csv(index=False))
    return output


def _compute_summary(results: list[dict]) -> dict:
    n = len(results)
    class_matches = [
        r["predicted"]["event_class"] == r["expected_event_class"]
        for r in results
    ]
    class_counts = {}
    for r in results:
        cls = r["expected_event_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    majority_count = max(class_counts.values(), default=0)

    direction_matches = 0
    direction_total = 0
    raw_direction_matches = 0
    raw_direction_total = 0
    abnormal_positive = 0
    abnormal_negative = 0
    abs_high = []
    abs_low = []
    impact_vals = []
    realized_abs = []
    novelty_vals = []

    for r in results:
        pred = r["predicted"]
        abnormal = r["outcome"]["event_mean_abnormal_move"].get("t+5")
        raw = r["outcome"]["event_mean_move"].get("t+5")

        if abnormal is not None:
            pred_sign = np.sign(pred["sentiment_score"])
            actual_sign = np.sign(abnormal)
            if pred_sign != 0 and actual_sign != 0:
                direction_total += 1
                direction_matches += int(pred_sign == actual_sign)
                if actual_sign > 0:
                    abnormal_positive += 1
                else:
                    abnormal_negative += 1

        if raw is not None:
            raw_pred_sign = np.sign(pred["sentiment_score"])
            raw_actual_sign = np.sign(raw)
            if raw_pred_sign != 0 and raw_actual_sign != 0:
                raw_direction_total += 1
                raw_direction_matches += int(raw_pred_sign == raw_actual_sign)

            if pred["impact_score"] >= 7:
                abs_high.append(abs(raw))
            else:
                abs_low.append(abs(raw))
            impact_vals.append(pred["impact_score"])
            realized_abs.append(abs(raw))
            novelty_vals.append(pred["novelty_score"])

    confusion = {
        expected: {predicted: 0 for predicted in EVENT_CLASSES}
        for expected in EVENT_CLASSES
    }
    for r in results:
        expected = r["expected_event_class"]
        predicted = r["predicted"]["event_class"]
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1
    per_class_recall = {
        cls: round(row[cls] / sum(row.values()), 4) if sum(row.values()) else None
        for cls, row in confusion.items()
    }

    return {
        "n_events": n,
        "classification_confusion_matrix": confusion,
        "classification_recall_by_class": per_class_recall,
        "classification_accuracy": round(sum(class_matches) / n, 4) if n else 0.0,
        "classification_correct": int(sum(class_matches)),
        "classification_majority_baseline": round(majority_count / n, 4) if n else 0.0,
        "direction_match_rate_abnormal_t5": round(direction_matches / direction_total, 4) if direction_total else None,
        "direction_correct_abnormal_t5": direction_matches,
        "direction_sample_size_abnormal_t5": direction_total,
        "direction_random_baseline": 0.50,
        "direction_majority_baseline": round(max(abnormal_positive, abnormal_negative) / direction_total, 4) if direction_total else None,
        "direction_actual_positive": abnormal_positive,
        "direction_actual_negative": abnormal_negative,
        "direction_match_rate_raw_t5": round(raw_direction_matches / raw_direction_total, 4) if raw_direction_total else None,
        "direction_correct_raw_t5": raw_direction_matches,
        "direction_sample_size_raw_t5": raw_direction_total,
        "high_impact_abs_move_avg_pct": round(float(np.mean(abs_high)), 3) if abs_high else None,
        "low_impact_abs_move_avg_pct": round(float(np.mean(abs_low)), 3) if abs_low else None,
        "impact_vs_abs_t5_pearson": _safe_corr(impact_vals, realized_abs, pearsonr),
        "impact_vs_abs_t5_spearman": _safe_corr(impact_vals, realized_abs, spearmanr),
        "novelty_vs_abs_t5_pearson": _safe_corr(novelty_vals, realized_abs, pearsonr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tuning", "holdout"], default="holdout")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.split)["summary"], indent=2))
