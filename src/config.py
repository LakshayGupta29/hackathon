"""Project-wide paths, constants and environment helpers."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Bump whenever NLP logic changes so cached downstream signals are rebuilt.
PIPELINE_VERSION = "2.1.0"

# Canonical event taxonomy. The case study lists five example categories;
# "Earnings/Corporate" is added because company-specific news (earnings,
# guidance, layoffs, recalls, strikes, analyst actions) is the most common
# headline type and was previously forced into "Product Launch".
# "Other" absorbs market commentary and non-events (personal finance, crypto
# price chatter, lifestyle pieces) and low-confidence classifications, so they
# never move Module A or trigger Module B.
EVENT_CLASSES = [
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Product Launch",
    "Earnings/Corporate",
    "Other",
]

# Event classes treated as systemic risk-off scenarios by Module B.
RISK_OFF_CLASSES = {"Geopolitical", "Macroeconomic", "Credit Event"}


def data_path(name: str) -> Path:
    """Absolute path of an input file inside the repository's data directory."""
    return DATA_DIR / name


def output_path(name: str) -> Path:
    """
    Absolute path for generated outputs (caches, results, live signal log).

    Defaults to data/ so results ship with the repo. Tests point
    RISK_ENGINE_OUTPUT_DIR at a temporary folder to keep the repo clean.
    """
    base = Path(os.getenv("RISK_ENGINE_OUTPUT_DIR", str(DATA_DIR)))
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
