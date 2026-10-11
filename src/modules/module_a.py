"""
Module A: event-driven tactical index rebalancer.

Subscribes to the NLP engine's Sentiment Score. For each stock in a
15-name mock index (drawn from the S&P 100):
  positive sentiment -> weight up, negative sentiment -> weight down.

Controls that keep it realistic and stable:
  - dead-band: |sentiment| < 0.15 is treated as noise (no trade)
  - smoothing: an exponential move toward the target weight (ALPHA)
  - bounds: 2% <= weight <= 15%, weights always sum to 1
  - turnover cap: at most 3% one-way portfolio turnover per signal
  - decay: tilts fade back toward equal weight with a 5-day half-life,
    so stale news stops driving the portfolio
"""

import argparse
import json
import math
import threading
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import PIPELINE_VERSION, data_path, output_path

INDEX_STOCKS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "JPM": "Financials",
    "GS": "Financials",
    "CVX": "Energy",
    "BA": "Industrials",
    "LMT": "Industrials",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "TSLA": "Consumer Discretionary",
    "AMZN": "Consumer Discretionary",
    "TSM": "Technology",
}

EQUAL_WEIGHT = 1.0 / len(INDEX_STOCKS)
MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.15
SENSITIVITY = 0.05
ALPHA = 0.3
MAX_TURNOVER = 0.03          # max one-way portfolio turnover per signal update
SENTIMENT_DEADBAND = 0.15    # |score| below this is ignored
HALF_LIFE_DAYS = 5.0         # tilt decay toward equal weight
DEFAULT_COST_BPS = 5.0       # one-way transaction cost for the backtest
MAX_HISTORY = 2000


def _to_utc(ts) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


class IndexRebalancer:
    """Stateful 15-name rebalancer with smoothing, bounds, turnover cap and decay."""

    def __init__(self, start_time=None):
        self.weights = {t: EQUAL_WEIGHT for t in INDEX_STOCKS}
        self.history: list[dict] = []
        self._clock = _to_utc(start_time)
        self._lock = threading.Lock()
        self._record("init")

    # ── Core mechanics ────────────────────────────────────────
    def _decay_to(self, as_of: datetime) -> None:
        elapsed_days = (as_of - self._clock).total_seconds() / 86_400.0
        if elapsed_days > 0:
            factor = 0.5 ** (elapsed_days / HALF_LIFE_DAYS)
            # Convex combination of two valid weight vectors: sum and bounds hold.
            self.weights = {
                t: EQUAL_WEIGHT + (w - EQUAL_WEIGHT) * factor
                for t, w in self.weights.items()
            }
            self._clock = as_of

    def advance(self, as_of=None) -> None:
        """Apply time decay up to `as_of` without trading on a signal."""
        with self._lock:
            self._decay_to(_to_utc(as_of))

    def update(self, ticker: str, sentiment_score: float, source: str = "", as_of=None) -> dict:
        if ticker not in INDEX_STOCKS:
            return {}
        score = float(sentiment_score)

        with self._lock:
            self._decay_to(_to_utc(as_of))
            old_weights = self.weights.copy()
            old_weight = old_weights[ticker]

            if abs(score) < SENTIMENT_DEADBAND:
                return {
                    "ticker": ticker,
                    "skipped": "deadband",
                    "old_weight": round(old_weight, 6),
                    "new_weight": round(old_weight, 6),
                    "delta": 0.0,
                    "portfolio_turnover": 0.0,
                }

            raw_delta = float(np.clip(score * SENSITIVITY, -MAX_TURNOVER, MAX_TURNOVER))
            target_weight = old_weight + raw_delta
            smoothed = ALPHA * target_weight + (1.0 - ALPHA) * old_weight
            proposed = old_weights.copy()
            proposed[ticker] = float(np.clip(smoothed, MIN_WEIGHT, MAX_WEIGHT))
            self.weights = proposed
            self._rebalance_bounds()

            turnover = 0.5 * sum(abs(self.weights[t] - old_weights[t]) for t in INDEX_STOCKS)
            if turnover > MAX_TURNOVER:
                scale = MAX_TURNOVER / turnover
                self.weights = {
                    t: old_weights[t] + scale * (self.weights[t] - old_weights[t])
                    for t in INDEX_STOCKS
                }
                turnover = MAX_TURNOVER

            self._record(f"{ticker}:{score:+.2f}", source=source)
            return {
                "ticker": ticker,
                "old_weight": round(old_weight, 6),
                "new_weight": round(self.weights[ticker], 6),
                "delta": round(self.weights[ticker] - old_weight, 6),
                "portfolio_turnover": round(turnover, 6),
            }

    def _rebalance_bounds(self) -> None:
        """Renormalise to sum 1 while respecting floor/ceiling."""
        weights = self.weights.copy()
        for _ in range(50):
            weights = {t: float(np.clip(w, MIN_WEIGHT, MAX_WEIGHT)) for t, w in weights.items()}
            total = sum(weights.values())
            if abs(total - 1.0) < 1e-12:
                break
            if total > 1.0:
                free = [t for t, w in weights.items() if w > MIN_WEIGHT + 1e-15]
            else:
                free = [t for t, w in weights.items() if w < MAX_WEIGHT - 1e-15]
            if not free:
                break
            free_total = sum(weights[t] for t in free)
            target_free = free_total + (1.0 - total)
            scale = target_free / free_total
            for t in free:
                weights[t] *= scale
        total = sum(weights.values())
        self.weights = {t: w / total for t, w in weights.items()}

    def _record(self, trigger: str, source: str = "") -> None:
        self.history.append({
            "timestamp": self._clock.isoformat(),
            "trigger": trigger,
            "source": source,
            "weights": {t: round(w, 6) for t, w in self.weights.items()},
        })
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    # ── Read models ───────────────────────────────────────────
    def get_weights(self) -> dict:
        return deepcopy(self.weights)

    def get_history(self, limit: int | None = None) -> list:
        history = self.history if not limit else self.history[-int(limit):]
        return deepcopy(history)

    def reset(self) -> None:
        self.__init__()


# ── Historical event replay (backtest) ───────────────────────

def _signal_cache_name() -> str:
    return f"module_a_signals_v{PIPELINE_VERSION}.json"


def build_event_signals(dataset_path: str | None = None, rebuild: bool = False) -> list[dict]:
    """
    Run the frozen NLP pipeline over the chronological historical event set
    and cache the signals for the 15 index names. The cache is keyed by the
    pipeline version so NLP changes always trigger a rebuild.
    """
    cache = output_path(_signal_cache_name())
    if cache.exists() and not rebuild and dataset_path is None:
        return json.loads(cache.read_text())

    if dataset_path is None:
        events = []
        for name in ("master_tuning_set.json", "master_holdout_set.json"):
            events.extend(json.loads(data_path(name).read_text())["events"])
    else:
        events = json.loads(data_path(dataset_path).read_text())["events"]
    events = sorted(events, key=lambda e: (e["date"], e["id"]))

    from src.engine.pipeline import process, reset_novelty_memory
    reset_novelty_memory()

    signals = []
    for event in events:
        prediction = process(event["headline"], source=event.get("source", "news:unknown"))
        for ticker in prediction["affected_tickers"]:
            if ticker in INDEX_STOCKS:
                signals.append({
                    "event_id": event["id"],
                    "date": event["date"],
                    "headline": event["headline"],
                    "ticker": ticker,
                    "sentiment_score": prediction["sentiment_score"],
                    "event_class": prediction["event_class"],
                    "impact_score": prediction["impact_score"],
                })

    if dataset_path is None:
        cache.write_text(json.dumps(signals, indent=2))
    return signals


def _max_drawdown(path: pd.Series) -> float:
    return float((path / path.cummax() - 1.0).min())


def _annualised(series: pd.Series) -> tuple[float, float]:
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    vol = std * math.sqrt(252)
    sharpe = float(series.mean()) / std * math.sqrt(252) if std > 0 else 0.0
    return vol, sharpe


def run_backtest(
    days: int | None = None,
    sentiment_events: list | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    save: bool = True,
) -> dict:
    """
    Event-driven backtest on real prices (yfinance).

    Signals are applied at the first trading-day OPEN strictly after the
    headline date (publication times are unknown, so the event-day reaction
    is never used). Daily return = open-to-close, weights decay between
    signals, and transaction costs are charged on turnover.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for the Module A backtest") from exc

    if sentiment_events is None:
        sentiment_events = build_event_signals()
    if not sentiment_events:
        raise RuntimeError("No index-relevant event signals available for backtest")

    ticker_list = list(INDEX_STOCKS)
    event_dates = [pd.Timestamp(e["date"]).normalize() for e in sentiment_events if e.get("date")]
    latest, earliest = max(event_dates), min(event_dates)
    end = latest + pd.Timedelta(days=10)
    start = earliest - pd.Timedelta(days=10)
    if days and days > 0:
        start = max(start, end - pd.Timedelta(days=days))
    active_events = [e for e in sentiment_events if pd.Timestamp(e["date"]).normalize() >= start]

    raw = yf.download(
        ticker_list,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("No price data returned by yfinance")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Unexpected yfinance response format (expected multi-ticker columns)")

    opens = raw["Open"]
    closes = raw["Close"]
    opens.index = pd.DatetimeIndex(opens.index).tz_localize(None).normalize()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    available = [t for t in ticker_list if t in opens.columns and t in closes.columns]
    opens = opens[available].dropna(axis=1, how="any")
    closes = closes[list(opens.columns)]
    available = list(opens.columns)
    if not available:
        raise RuntimeError("Price data had no complete ticker series")
    intraday = (closes / opens - 1.0).dropna(how="any")

    signal_by_day: dict[pd.Timestamp, dict[str, list[float]]] = {}
    for event in active_events:
        if event["ticker"] not in available:
            continue
        event_dt = pd.Timestamp(event["date"]).normalize()
        future_days = intraday.index[intraday.index > event_dt]
        if len(future_days) == 0:
            continue
        signal_by_day.setdefault(future_days[0], {}).setdefault(event["ticker"], []).append(
            float(event["sentiment_score"])
        )

    rebalancer = IndexRebalancer(start_time=intraday.index[0])
    cost = cost_bps / 10_000.0
    gross, net, bench, turnover_daily, dates, weight_path = [], [], [], [], [], []
    signals_applied = 0

    for date, row in intraday.iterrows():
        before = rebalancer.get_weights()
        rebalancer.advance(date)
        for ticker, scores in signal_by_day.get(date, {}).items():
            result = rebalancer.update(ticker, float(np.mean(scores)), source="replay", as_of=date)
            if result and "skipped" not in result:
                signals_applied += 1
        after = rebalancer.get_weights()
        turnover = 0.5 * sum(abs(after[t] - before[t]) for t in available)

        day_return = sum(after[t] * float(row[t]) for t in available) / sum(after[t] for t in available)
        gross.append(day_return)
        net.append(day_return - 2.0 * turnover * cost)  # buy + sell legs
        bench.append(float(row[available].mean()))
        turnover_daily.append(turnover)
        dates.append(date.strftime("%Y-%m-%d"))
        weight_path.append({t: round(after[t], 6) for t in INDEX_STOCKS})

    idx = pd.to_datetime(dates)
    gross_s, net_s, bench_s = (pd.Series(v, index=idx) for v in (gross, net, bench))
    gross_path, net_path, bench_path = ((1 + s).cumprod() for s in (gross_s, net_s, bench_s))
    active = net_s - bench_s
    vol, sharpe = _annualised(net_s)
    _, bench_sharpe = _annualised(bench_s)
    active_std = float(active.std(ddof=1)) if len(active) > 1 else 0.0
    information_ratio = float(active.mean()) / active_std * math.sqrt(252) if active_std > 0 else 0.0

    result = {
        "method": "event_driven_nlp_replay",
        "pipeline_version": PIPELINE_VERSION,
        "signal_application": "first trading-day open strictly after event date",
        "return_definition": "open-to-close daily returns; weights decay between signals",
        "dates": dates,
        "sentiment_returns": [round(float(v - 1.0), 6) for v in net_path],
        "sentiment_returns_gross": [round(float(v - 1.0), 6) for v in gross_path],
        "equalweight_returns": [round(float(v - 1.0), 6) for v in bench_path],
        "weights_history": weight_path,
        "sentiment_final": round(float(net_path.iloc[-1] - 1.0), 4),
        "sentiment_final_gross": round(float(gross_path.iloc[-1] - 1.0), 4),
        "equalweight_final": round(float(bench_path.iloc[-1] - 1.0), 4),
        "excess_return": round(float(net_path.iloc[-1] - bench_path.iloc[-1]), 4),
        "sharpe_ratio": round(sharpe, 4),
        "benchmark_sharpe_ratio": round(bench_sharpe, 4),
        "information_ratio": round(information_ratio, 4),
        "annualized_volatility": round(vol, 4),
        "max_drawdown": round(_max_drawdown(net_path), 4),
        "benchmark_max_drawdown": round(_max_drawdown(bench_path), 4),
        "average_daily_turnover": round(float(np.mean(turnover_daily)), 5),
        "transaction_cost_bps": cost_bps,
        "transaction_costs_included": cost_bps > 0,
        "tickers_used": available,
        "calendar_days_requested": days,
        "trading_days": len(dates),
        "event_signal_count": len(active_events),
        "signals_applied": signals_applied,
        "note": (
            "Event-replay demonstration of the live rebalancing logic on real prices "
            "with frozen NLP parameters. Small event sample: not evidence of "
            "tradable alpha."
        ),
    }
    if save:
        output_path("backtest_results.json").write_text(json.dumps(result, indent=2))
    return result


rebalancer = IndexRebalancer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module A historical event replay")
    parser.add_argument("--rebuild", action="store_true", help="re-run the NLP pipeline on the event set")
    parser.add_argument("--days", type=int, default=0, help="limit replay window (0 = full)")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args()

    signals = build_event_signals(rebuild=args.rebuild)
    print(f"{len(signals)} index-relevant signals")
    res = run_backtest(days=args.days, sentiment_events=signals, cost_bps=args.cost_bps)
    keys = ["sentiment_final", "sentiment_final_gross", "equalweight_final", "excess_return",
            "sharpe_ratio", "information_ratio", "max_drawdown", "average_daily_turnover",
            "trading_days", "signals_applied"]
    print(json.dumps({k: res[k] for k in keys}, indent=2))
