#!/usr/bin/env python3
"""
Tweet signal study (Module A at scale).

Uses the public Kaggle dataset "Tweet Sentiment's Impact on Stock Returns"
(thedevastator): tweets about specific stocks with dates and forward returns.

  1. Map each tweet's stock to the 15 Module A index names; keep only those.
  2. Sample up to --max-tweets tweets (capped per stock-day so busy days do
     not dominate) and score them with the project's fine-tuned FinBERT.
  3. Signal test: daily mean sentiment per stock vs the dataset's forward
     1/2/3/7-day returns (Spearman rank IC, hit rate, top-vs-bottom tercile),
     with the dataset's own TextBlob/LSTM polarity as baselines and the
     sample split chronologically into two halves.
  4. Module A replay: the same frozen rebalancer (no parameter tuning) driven
     by the daily signals on real prices (yfinance), vs equal weight.

Raw data stays in data/tweets/ (git-ignored). Outputs:
  data/tweet_signals_daily.csv     small daily extract (committed)
  data/tweet_study_results.json    all metrics
  data/tweet_scores_cache.csv      per-tweet scores (cache, git-ignored)

Usage:
  python scripts/tweet_signal_study.py                 # auto-finds data/tweets/*reduced*.csv
  python scripts/tweet_signal_study.py --max-tweets 8000 --skip-backtest
"""

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import data_path, output_path  # noqa: E402
from src.modules.module_a import INDEX_STOCKS, SENTIMENT_DEADBAND  # noqa: E402

SEED = 42
PER_STOCK_DAY_CAP = 12

# Extra names seen in social data for the 15 index stocks.
NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "google": "GOOGL", "alphabet": "GOOGL",
    "facebook": "META", "meta": "META", "jpmorgan": "JPM", "jp morgan": "JPM", "jpmorgan chase": "JPM",
    "goldman sachs": "GS", "goldman": "GS", "chevron": "CVX", "boeing": "BA", "lockheed martin": "LMT",
    "lockheed": "LMT", "johnson & johnson": "JNJ", "johnson and johnson": "JNJ", "j&j": "JNJ",
    "unitedhealth": "UNH", "united health": "UNH", "tesla": "TSLA", "amazon": "AMZN",
    "tsmc": "TSM", "taiwan semiconductor": "TSM",
}


def find_input(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    hits = sorted(glob.glob(str(data_path("tweets") / "*reduced*.csv"))) or \
        sorted(glob.glob(str(data_path("tweets") / "*.csv")))
    if not hits:
        sys.exit("No CSV found in data/tweets/. Unzip the Kaggle archive there first.")
    return Path(hits[0])


def pick(columns, *patterns):
    for pattern in patterns:
        for c in columns:
            if re.fullmatch(pattern, c.strip(), flags=re.IGNORECASE):
                return c
    return None


def map_stock(value) -> str | None:
    v = str(value).strip()
    up = v.upper().lstrip("$")
    if up in INDEX_STOCKS:
        return up
    if up == "FB":
        return "META"
    if up == "GOOG":
        return "GOOGL"
    low = re.sub(r"\s+(inc|corp|corporation|co|company|plc)\.?$", "", v.lower()).strip()
    return NAME_TO_TICKER.get(low)


def spearman(x: pd.Series, y: pd.Series) -> dict:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 20 or x[mask].nunique() < 3:
        return {"ic": None, "p_value": None, "n": n}
    from scipy.stats import spearmanr
    r, p = spearmanr(x[mask], y[mask])
    return {"ic": round(float(r), 4), "p_value": round(float(p), 4), "n": n}


def signal_metrics(df: pd.DataFrame, score_col: str, return_cols: list[str]) -> dict:
    out = {}
    for rc in return_cols:
        sub = df[[score_col, rc]].dropna()
        res = spearman(sub[score_col], sub[rc])
        # Zero returns (non-trading days in the source data) carry no direction: exclude them.
        directional = sub[(sub[score_col].abs() >= SENTIMENT_DEADBAND) & (sub[rc] != 0)]
        res["zero_return_share"] = round(float((sub[rc] == 0).mean()), 4)
        if len(directional) >= 20:
            hits = np.sign(directional[score_col]) == np.sign(directional[rc])
            res["hit_rate"] = round(float(hits.mean()), 4)
            res["hit_n"] = int(len(directional))
            res["up_base_rate"] = round(float((directional[rc] > 0).mean()), 4)
        if len(sub) >= 30:
            lo, hi = sub[score_col].quantile([1 / 3, 2 / 3])
            top = sub[sub[score_col] >= hi][rc].mean()
            bottom = sub[sub[score_col] <= lo][rc].mean()
            res["top_minus_bottom_tercile_return_pct"] = round(float((top - bottom) * 100), 4)
        out[rc] = res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input")
    parser.add_argument("--max-tweets", type=int, default=20000)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    path = find_input(args.input)
    print(f"Reading {path}")
    try:
        raw = pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError as exc:
        # The full Kaggle file contains a few malformed rows (stray quotes / newlines).
        print(f"Fast parser failed ({str(exc)[:60]}...); retrying with the tolerant parser, skipping bad lines")
        raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
        print(f"Loaded {len(raw):,} rows")
    cols = list(raw.columns)
    text_col = pick(cols, r"tweet", r"text", r"content", r"body")
    stock_col = pick(cols, r"stock", r"ticker", r"symbol", r"company")
    date_col = pick(cols, r"date", r"created_at", r"datetime", r"timestamp")
    return_cols = [c for c in cols if re.fullmatch(r"\d+_day_return", c.strip(), flags=re.IGNORECASE)]
    baseline_cols = [c for c in cols if re.search(r"polarity", c, flags=re.IGNORECASE)]
    if not (text_col and stock_col and date_col):
        sys.exit(f"Could not detect text/stock/date columns in {cols}")
    print(f"Columns: text={text_col} stock={stock_col} date={date_col} returns={return_cols} baselines={baseline_cols}")

    df = raw[[text_col, stock_col, date_col] + return_cols + baseline_cols].copy()
    df.columns = ["text", "stock", "date"] + return_cols + baseline_cols
    mapping = {v: map_stock(v) for v in df["stock"].dropna().unique()}
    df["ticker"] = df["stock"].map(mapping)
    unmapped = sorted({str(k) for k, v in mapping.items() if v is None})[:15]
    df = df[df["ticker"].notna()].copy()
    if df.empty:
        sys.exit(f"No tweets matched the 15 index stocks. Stock values look like: {unmapped}")

    sample_dates = df["date"].astype(str).head(200)
    dayfirst = sample_dates.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").any() and \
        sample_dates.str.extract(r"^(\d{1,2})/")[0].dropna().astype(int).max() > 12
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=bool(dayfirst)).dt.tz_localize(None).dt.normalize()
    df = df[df["date"].notna()]
    for rc in return_cols + baseline_cols:
        df[rc] = pd.to_numeric(df[rc], errors="coerce")
    df["text"] = df["text"].astype(str).str.replace(r"https?://\S+", "", regex=True).str.strip()
    df = df[df["text"].str.len() >= 15].drop_duplicates(subset=["text", "ticker"])
    print(f"{len(df):,} tweets on index stocks, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    # Cap per stock-day, then sample down to the budget.
    df = df.sample(frac=1.0, random_state=SEED)
    df = df.groupby(["ticker", "date"], group_keys=False).head(PER_STOCK_DAY_CAP)
    if len(df) > args.max_tweets:
        df = df.sample(n=args.max_tweets, random_state=SEED)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    cache_file = output_path("tweet_scores_cache.csv")
    cache = {}
    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        cache = dict(zip(cached["text"], cached["sentiment"]))
    todo = [t for t in df["text"].unique() if t not in cache]
    print(f"Scoring {len(todo):,} tweets with the fine-tuned model ({len(cache):,} cached)...")
    if todo:
        from tqdm import tqdm
        from src.engine.sentiment import analyze_batch, model_status
        chunk = 512
        for start in tqdm(range(0, len(todo), chunk)):
            batch = todo[start:start + chunk]
            for text, res in zip(batch, analyze_batch(batch, mode="finetuned", batch_size=32)):
                cache[text] = res["score"]
            pd.DataFrame({"text": list(cache), "sentiment": list(cache.values())}).to_csv(cache_file, index=False)
        model_name = model_status()["model"]
    else:
        model_name = "cached"
    df["sentiment"] = df["text"].map(cache)

    # Daily aggregation per stock.
    agg = {"sentiment": "mean", "text": "count"}
    agg.update({rc: "mean" for rc in return_cols})
    agg.update({bc: "mean" for bc in baseline_cols})
    daily = df.groupby(["ticker", "date"]).agg(agg).rename(columns={"text": "n_tweets"}).reset_index()
    daily = daily.sort_values(["date", "ticker"]).reset_index(drop=True)
    daily_out = daily.copy()
    daily_out["date"] = daily_out["date"].dt.strftime("%Y-%m-%d")
    daily_out.round(6).to_csv(output_path("tweet_signals_daily.csv"), index=False)

    # Signal tests: full sample, chronological halves, baselines.
    cut = daily["date"].quantile(0.5)
    halves = {"first_half": daily[daily["date"] <= cut], "second_half": daily[daily["date"] > cut]}
    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": path.name,
        "sentiment_model": model_name,
        "tweets_scored": int(len(df)),
        "stock_days": int(len(daily)),
        "tickers": sorted(daily["ticker"].unique().tolist()),
        "date_range": [str(daily["date"].min().date()), str(daily["date"].max().date())],
        "per_stock_day_cap": PER_STOCK_DAY_CAP,
        "deadband": SENTIMENT_DEADBAND,
        "unmapped_stock_examples": unmapped,
        "signal_test": {
            "ours_full": signal_metrics(daily, "sentiment", return_cols),
            **{f"ours_{k}": signal_metrics(v, "sentiment", return_cols) for k, v in halves.items()},
            **{f"baseline_{bc}": signal_metrics(daily, bc, return_cols) for bc in baseline_cols},
        },
        "note": ("Forward returns are those supplied with the dataset, measured from the tweet date; "
                 "rank IC = Spearman correlation between daily mean sentiment and forward return."),
    }

    if not args.skip_backtest:
        events = [
            {"event_id": f"tw_{i}", "date": r.date.strftime("%Y-%m-%d"), "ticker": r.ticker,
             "sentiment_score": float(r.sentiment)}
            for i, r in enumerate(daily.itertuples()) if abs(r.sentiment) >= SENTIMENT_DEADBAND
        ]
        print(f"Module A replay with {len(events):,} daily signals (frozen parameters)...")
        try:
            from src.modules.module_a import run_backtest
            bt = run_backtest(sentiment_events=events, save=False)
            keep = ["sentiment_final", "sentiment_final_gross", "equalweight_final", "excess_return",
                    "sharpe_ratio", "benchmark_sharpe_ratio", "information_ratio", "max_drawdown",
                    "benchmark_max_drawdown", "average_daily_turnover", "trading_days", "signals_applied",
                    "transaction_cost_bps"]
            results["module_a_replay"] = {k: bt[k] for k in keep}
            bt["signal_source"] = "tweet_daily_sentiment"
            output_path("tweet_backtest_results.json").write_text(json.dumps(bt, indent=2))
        except Exception as exc:  # network / yfinance issues should not lose the signal test
            results["module_a_replay"] = {"error": str(exc)}

    output_path("tweet_study_results.json").write_text(json.dumps(results, indent=2))
    printable = {k: v for k, v in results.items() if k != "unmapped_stock_examples"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
