#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon

echo
echo "Environment ready."
echo "Activate: source .venv/bin/activate"
echo "Run API: python -m uvicorn src.api.main:app --reload"
echo "Run dashboard: streamlit run src/dashboard/app.py"
