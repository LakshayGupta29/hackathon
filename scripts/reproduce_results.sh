#!/usr/bin/env bash
# Regenerate every result file the dashboard and the deck use.
# Needs internet (Hugging Face models on first run, yfinance prices).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 Offline tests"
python -m pytest -q

echo "== 2/5 Sentiment evaluation on the locked test set -> data/eval_results.json"
python -m src.engine.evaluator --mode all

echo "== 3/5 Historical validation -> data/validation_results_{tuning,holdout}.json"
python -m src.modules.validation_backtest --split tuning
python -m src.modules.validation_backtest --split holdout

echo "== 4/5 Module A event replay -> data/backtest_results.json"
python -m src.modules.module_a --rebuild

if [ -f data/fresh_test_set.csv ]; then
  echo "== 5/5 Fresh unseen test set -> data/fresh_test_results.json"
  python scripts/fresh_test_set.py score
fi

echo "Done. Commit the updated files in data/."
