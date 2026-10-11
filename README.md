# Financial Risk Intelligence Platform — S&P Global & Crisil Campus Hackathon 2026

A modular financial-risk prototype that ingests unstructured financial headlines/social replay data, converts them into structured risk signals, and feeds those signals into tactical index rebalancing and wholesale-portfolio stress testing.

## Current architecture

`RSS / social replay -> NLP risk engine -> risk graph -> Module A / Module B -> FastAPI -> Streamlit`

The NLP engine combines fine-tuned FinBERT sentiment, hybrid keyword + zero-shot event classification (the five case-study examples plus Earnings/Corporate, and an "Other" class for low-confidence non-events), sentence embeddings, novelty and duplicate detection, complete-linkage event fusion, entity/ticker linking, and an explainable 1–10 impact score. Signals are served over a REST API / WebSocket and appended to `data/signals_live.jsonl`.

## Quickstart

Tested target: Python 3.11+ on Linux/macOS/Windows with a CPU-friendly configuration. The first model inference downloads Hugging Face models, so internet access is required for first run.

```bash
git clone <your-public-repo-url>
cd <repo-folder>
bash scripts/setup.sh
source .venv/bin/activate
```

Point the app at your fine-tuned sentiment model (falls back to `ProsusAI/finbert` if it cannot be loaded):

```bash
export HF_FINETUNED_MODEL="<your-hf-username>/finbert-financial-sentiment-v2"
```

Start the API:

```bash
python -m uvicorn src.api.main:app --reload
```

Start the dashboard in another terminal:

```bash
source .venv/bin/activate
streamlit run src/dashboard/app.py
```

The API docs are available at `http://127.0.0.1:8000/docs`. Set `DISABLE_LIVE_FEEDS=1` to run without the RSS/social streams; if no RSS feed is reachable the app replays bundled historical headlines.

## Offline sanity tests

```bash
python scripts/check_repo.py
python -m pytest -q
```

These 48 tests (unit + API end-to-end) use small fake models, so they need no model downloads, network or yfinance. Regenerate every result file with `bash scripts/reproduce_results.sh`.

## Data

All project datasets are synthetic or publicly sourced. No proprietary S&P Global or Crisil client data is used. The repository includes the data needed for the prototype and validation experiments.

The historical validation manifests are split into:

- `data/master_tuning_set.json` — development/tuning events
- `data/master_holdout_set.json` — locked holdout events

The corrected validation script processes events chronologically and only uses market prices after the event date. Because the archive has no publication timestamps, it deliberately avoids using event-day closes and does **not** claim causal impact.

## Fine-tuning status

`notebooks/03_finetune_sentiment.ipynb` contains the corrected training procedure:

- source labels `0=negative, 1=positive, 2=neutral` are remapped to native FinBERT ids `0=positive, 1=negative, 2=neutral`
- a stratified train/validation split is used for checkpoint selection
- the locked test set is evaluated only after training
- train/test normalized text overlaps are removed before fitting
- class-weighted cross-entropy is retained for the imbalanced sentiment labels

The old pre-fix sentiment result files are intentionally not included. Re-run the notebook before publishing final metrics.

## Validation and backtesting

```bash
python -m src.modules.validation_backtest --split tuning
python -m src.modules.validation_backtest --split holdout
```

Module A uses the same frozen NLP pipeline on the historical event manifests and applies a signal at the first trading-day open after the event date, with a sentiment dead-band, 2–15% weight bounds, a 3% turnover cap, a 5-day decay back to equal weight and 5 bp transaction costs. Its result is an event-replay demonstration, not evidence of tradable alpha.

```bash
python -m src.modules.module_a --rebuild
```

## Stress-testing model

Module B uses a synthetic wholesale portfolio containing loans, bonds, equities, and derivatives. Equity shocks use percentage price moves. Bond and loan mark-to-market shocks use first-order duration / spread-duration approximations, with credit-spread betas by rating. Interest-rate moves reprice every rate-sensitive position; event-specific equity and spread shocks hit directly named obligors at x1.5, affected sectors at x1.0 and the rest of the book at x0.5 (x0.1 for company-specific events). IRS/CDS positions use explicit DV01/CS01 sensitivities.

A stress test triggers automatically when `impact_score >= 7.0` and the event is a systemic risk-off class (Geopolitical, Macroeconomic, Credit Event) or clearly negative company news. Scenarios are what-ifs on the base book by default (`baseline` mode); `cumulative` mode compounds them. Near-duplicate headlines never trigger a second test.

## Important limitations

- The historical event sample is small and should not be used to claim statistical significance.
- The event taxonomy is constrained to five case-study categories, so some historical headlines are inherently ambiguous.
- Social input is a replay/simulation stream, not a live X/Twitter API integration.
- News publication timestamps are not available in the historical validation files, limiting causal claims.
- Module A backtesting excludes transaction costs and uses a short event sample.
- Loan credit losses are simplified to spread-duration mark-to-market; a production bank model would also model PD/LGD/default transitions.
