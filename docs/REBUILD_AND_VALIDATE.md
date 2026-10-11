# Rebuild and Validate Before Final Submission

Operational checklist (not the presentation documentation).

## 1. Retrain the sentiment model (Kaggle GPU)

Open `notebooks/03_finetune_sentiment.ipynb`, set the Kaggle secrets
`HF_TOKEN` and `HF_USERNAME` (your own Hugging Face account), and run it top
to bottom.

- The locked test set is never passed to `Trainer` as `eval_dataset`; a
  stratified 10% validation split selects the checkpoint.
- Train/test text overlaps are removed before training.
- Labels are remapped to native FinBERT ids (0=positive, 1=negative, 2=neutral).
- The notebook saves `finetune_metrics.json`. Download it and commit it to `data/`.

Then point the app at your model:

```bash
export HF_FINETUNED_MODEL="<your-hf-username>/finbert-financial-sentiment-v2"
```

If the model cannot be loaded, the app falls back to `ProsusAI/finbert` and
the dashboard header shows "(fallback)".

## 2. Regenerate every result file

```bash
bash scripts/reproduce_results.sh
```

This runs, in order:

| Step | Command | Output |
|---|---|---|
| Offline tests | `python -m pytest -q` | 48 tests, no network needed |
| Sentiment evaluation | `python -m src.engine.evaluator --mode all` | `data/eval_results.json` |
| Historical validation | `python -m src.modules.validation_backtest --split tuning` / `--split holdout` | `data/validation_results_*.json`, `data/validation_outcomes_*.csv` |
| Module A event replay | `python -m src.modules.module_a --rebuild` | `data/backtest_results.json`, `data/module_a_signals_v2.1.0.json` |

Commit the regenerated files so the dashboard's "Model & Results" tab and
the backtest button work instantly during the demo.

## 3. Label set notes

The validation sets use six event classes (the model can also abstain to "Other"). `Earnings/Corporate` was added
because company-specific news (earnings, guidance, deliveries, layoffs,
recalls, strikes, analyst actions) is the most common headline type and was
previously forced into `Product Launch` or `Credit Event`. Every relabelled
event keeps `event_class_original` and a `label_note`.

The keyword rules were written from general financial vocabulary, but the
headlines in the holdout file had already been seen. To report a strictly
unseen number, add 20-30 freshly collected headlines (for example with
`src/modules/gdelt_fetcher.py`), label them before running the model, and
evaluate on those.

## 4. Wording for results

- Validation is forward-only and observational (no publication timestamps):
  do not describe it as causal.
- The Module A replay is a demonstration of the live logic on real prices
  with 5 bp costs; the event sample is small, so it is not evidence of alpha.
- Impact weights and severity priors are transparent domain priors, not
  fitted coefficients.
