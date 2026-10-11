#!/usr/bin/env python3
"""Fast offline submission sanity checks; does not download models or market data."""
from pathlib import Path
import compileall
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "requirements.txt",
    ROOT / "src",
    ROOT / "data",
    ROOT / "notebooks/03_finetune_sentiment.ipynb",
    ROOT / "data/master_tuning_set.json",
    ROOT / "data/master_holdout_set.json",
    ROOT / "data/test_set.csv",
    ROOT / "data/synthetic_portfolio.json",
    ROOT / "data/shock_templates.json",
    ROOT / "data/risk_graph_static.json",
    ROOT / "tests/test_core.py",
    ROOT / "tests/test_api.py",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    print("Missing required paths:", missing)
    sys.exit(1)

ok = compileall.compile_dir(str(ROOT / "src"), quiet=1)
if not ok:
    print("Python compilation failed")
    sys.exit(1)

for notebook in [ROOT / "notebooks/03_finetune_sentiment.ipynb"]:
    nb = json.loads(notebook.read_text())
    assert nb.get("nbformat")

print("Repository sanity checks passed.")
print("Run full unit tests with: python -m pytest -q")
print("Regenerate every result file with: bash scripts/reproduce_results.sh")
