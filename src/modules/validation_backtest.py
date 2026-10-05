"""
Validation Backtest
Proves the risk engine's predictive validity using real historical
events with known, verifiable outcomes.

Methodology (strictly time-causal):
  1. Feed each historical headline through our NLP pipeline
     exactly as a live event would arrive
  2. Pipeline outputs: sentiment, event class, impact score, novelty
  3. Fetch REAL subsequent price moves (T+1, T+3, T+5 days) via yfinance
  4. Compare: did our impact/sentiment predictions align with what
     actually happened in the market?

No future information is used at prediction time - the pipeline only
ever sees the headline text, nothing else. This is what makes the
validation meaningful rather than circular.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.engine.pipeline import process, reset_novelty_memory


def _load_historical_events() -> list:
    data = json.loads(Path("data/historical_events.json").read_text())
    return data["events"]


def _get_price_moves(ticker: str, event_date: str) -> dict:
    """
    Fetch REAL price moves after the event date.
    Returns % change at T+1, T+3, T+5 trading days relative to
    the close on event_date (or nearest available date).
    """
    import yfinance as yf

    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    start = event_dt - timedelta(days=5)
    end   = event_dt + timedelta(days=12)  # buffer for T+5 trading days

    try:
        hist = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if hist.empty or len(hist) < 2:
            return {}

        # Normalize index to date-only for matching
        hist.index = hist.index.tz_localize(None)
        dates = hist.index

        # Find the trading day on/after event_date (base price)
        base_candidates = dates[dates >= event_dt]
        if len(base_candidates) == 0:
            return {}
        base_date = base_candidates[0]
        base_idx  = list(dates).index(base_date)
        base_price = hist.loc[base_date, "Close"]

        moves = {}
        for label, offset in [("t+1", 1), ("t+3", 3), ("t+5", 5)]:
            target_idx = base_idx + offset
            if target_idx < len(hist):
                target_price = hist.iloc[target_idx]["Close"]
                pct_change = (target_price - base_price) / base_price
                moves[label] = round(float(pct_change) * 100, 2)

        return moves

    except Exception as e:
        print(f"    [price fetch error for {ticker}]: {e}")
        return {}


def run_validation() -> dict:
    """
    Run the full validation backtest.

    Returns:
        {
            "events": [
                {
                    "date": str,
                    "headline": str,
                    "predicted": {sentiment, event_class, impact, novelty},
                    "affected_tickers": {
                        ticker: {t+1, t+3, t+5}  # real % moves
                    }
                }
            ],
            "summary": {
                "total_events": int,
                "direction_match_rate": float,  # did sentiment direction
                                                 # match t+5 price direction
                "high_impact_avg_move": float,  # avg |move| for impact >= 6
                "low_impact_avg_move": float,   # avg |move| for impact < 6
            }
        }
    """
    reset_novelty_memory()  # fresh state for clean validation
    historical_events = _load_historical_events()

    results = []
    print(f"Running validation on {len(historical_events)} historical events...\n")

    for evt in historical_events:
        print(f"  Processing: {evt['date']} - {evt['headline'][:50]}")

        # Run through OUR pipeline exactly as a live event would arrive
        prediction = process(evt["headline"], source=evt["source"])

        # Fetch REAL price moves (time-causal: only after event date)
        ticker_moves = {}
        for ticker in evt["affected_tickers"]:
            moves = _get_price_moves(ticker, evt["date"])
            if moves:
                ticker_moves[ticker] = moves

        results.append({
            "event_id":         evt["id"],
            "date":             evt["date"],
            "headline":         evt["headline"],
            "predicted": {
                "sentiment_score": prediction["sentiment_score"],
                "sentiment_label": prediction["sentiment_label"],
                "event_class":     prediction["event_class"],
                "impact_score":    prediction["impact_score"],
                "novelty_score":   prediction["novelty_score"],
                "detected_tickers": prediction["affected_tickers"],
            },
            "expected_class":    evt["event_class_expected"],
            "ticker_moves":      ticker_moves,
        })

    summary = _compute_summary(results)

    output = {"events": results, "summary": summary}
    Path("data/validation_results.json").write_text(
        json.dumps(output, indent=2)
    )
    return output


def _compute_summary(results: list) -> dict:
    """Compute aggregate validation metrics."""
    direction_matches = 0
    direction_total = 0
    high_impact_moves = []
    low_impact_moves = []
    class_matches = 0
    class_total = 0

    for r in results:
        pred = r["predicted"]
        sentiment_sign = 1 if pred["sentiment_score"] > 0 else (
            -1 if pred["sentiment_score"] < 0 else 0
        )

        for ticker, moves in r["ticker_moves"].items():
            if "t+5" not in moves:
                continue
            actual_move = moves["t+5"]
            actual_sign = 1 if actual_move > 0 else (-1 if actual_move < 0 else 0)

            if sentiment_sign != 0 and actual_sign != 0:
                direction_total += 1
                if sentiment_sign == actual_sign:
                    direction_matches += 1

            if pred["impact_score"] >= 6:
                high_impact_moves.append(abs(actual_move))
            else:
                low_impact_moves.append(abs(actual_move))

        class_total += 1
        # Loose match: our class vs expected class
        if pred["event_class"] == r["expected_class"]:
            class_matches += 1

    direction_match_rate = round(
        direction_matches / direction_total, 3
    ) if direction_total > 0 else 0.0

    high_avg = round(sum(high_impact_moves) / len(high_impact_moves), 2) \
        if high_impact_moves else 0.0
    low_avg = round(sum(low_impact_moves) / len(low_impact_moves), 2) \
        if low_impact_moves else 0.0

    class_accuracy = round(class_matches / class_total, 3) if class_total else 0.0

    return {
        "total_events":           len(results),
        "direction_match_rate":   direction_match_rate,
        "direction_sample_size":  direction_total,
        "high_impact_avg_move":   high_avg,
        "low_impact_avg_move":    low_avg,
        "event_class_accuracy":   class_accuracy,
    }


if __name__ == "__main__":
    output = run_validation()

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    for r in output["events"]:
        pred = r["predicted"]
        print(f"\n{r['date']} | {r['headline'][:60]}")
        print(f"  Predicted: {pred['event_class']} | "
              f"sentiment={pred['sentiment_label']} ({pred['sentiment_score']:+.2f}) | "
              f"impact={pred['impact_score']}/10 | novelty={pred['novelty_score']:.2f}")
        for ticker, moves in r["ticker_moves"].items():
            t5 = moves.get("t+5", "N/A")
            print(f"    {ticker}: actual T+5 move = {t5}%")

    s = output["summary"]
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Total events analyzed:        {s['total_events']}")
    print(f"Sentiment direction match:    {s['direction_match_rate']*100:.1f}% "
          f"(n={s['direction_sample_size']})")
    print(f"Avg |move| when impact >= 6:  {s['high_impact_avg_move']}%")
    print(f"Avg |move| when impact < 6:   {s['low_impact_avg_move']}%")
    print(f"Event classification match:   {s['event_class_accuracy']*100:.1f}%")
