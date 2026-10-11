"""
Offline test fixtures.

The real models (FinBERT, DeBERTa NLI, MiniLM) are replaced with small
deterministic fakes so the whole suite runs in seconds without network
access or GPU. Generated outputs go to a temporary directory.
"""

import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RISK_ENGINE_OUTPUT_DIR", tempfile.mkdtemp(prefix="risk_engine_test_"))
os.environ.setdefault("DISABLE_LIVE_FEEDS", "1")

NEG = {"fall", "falls", "fell", "plunge", "crisis", "war", "missile", "default", "collapse",
       "warn", "warns", "miss", "misses", "cut", "cuts", "recall", "slump", "fears", "hike"}
POS = {"beat", "beats", "surge", "surges", "record", "strong", "rally", "soar", "soars",
       "growth", "gain", "gains", "crushing"}


class FakeSentimentPipeline:
    """Mimics a transformers text-classification pipeline with top_k=None."""

    def __call__(self, texts, **kwargs):
        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            neg, pos = len(words & NEG), len(words & POS)
            if neg > pos:
                probs = {"positive": 0.04, "negative": 0.90, "neutral": 0.06}
            elif pos > neg:
                probs = {"positive": 0.88, "negative": 0.04, "neutral": 0.08}
            else:
                probs = {"positive": 0.10, "negative": 0.10, "neutral": 0.80}
            out.append([{"label": k, "score": v} for k, v in probs.items()])
        return out


class FakeZeroShot:
    """Always prefers the first candidate label (deterministic)."""

    def __call__(self, text, candidate_labels, **kwargs):
        n = len(candidate_labels)
        scores = [0.6] + [0.4 / (n - 1)] * (n - 1) if n > 1 else [1.0]
        return {"labels": list(candidate_labels), "scores": scores}


class FakeEncoder:
    """Hashing bag-of-words embedder (normalised, 384-d)."""

    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return self._vec(texts)
        return np.vstack([self._vec(t) for t in texts])


@pytest.fixture(scope="session", autouse=True)
def fake_models():
    """Session-wide so module-scoped fixtures (the API client) also use fakes."""
    from src.engine import embedder, event_classifier, sentiment

    patcher = pytest.MonkeyPatch()
    patcher.setattr(sentiment, "_load_finetuned", lambda: FakeSentimentPipeline())
    patcher.setattr(sentiment, "_load_finbert", lambda: FakeSentimentPipeline())
    patcher.setattr(event_classifier, "_load_classifier", lambda: FakeZeroShot())
    patcher.setattr(embedder, "_load_model", lambda: FakeEncoder())
    yield
    patcher.undo()
