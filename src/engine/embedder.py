"""
Sentence embedding utilities used for novelty scoring and event fusion.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-d, normalised).

Thresholds are heuristic and environment-overridable:
  SAME_EVENT_THRESHOLD (0.40): two headlines are clustered as the same story
    only if EVERY pair in the cluster is at least this similar (complete
    linkage), so unrelated headlines cannot chain together.
  DUPLICATE_THRESHOLD (0.80): a new item this similar to a recent item is a
    near-verbatim repeat (syndicated copy, replayed post) and is not allowed
    to move Module A weights or trigger another Module B stress test.
"""

from functools import lru_cache

import numpy as np

from src.config import env_float

SAME_EVENT_THRESHOLD = env_float("FUSION_SIMILARITY_THRESHOLD", 0.40)
DUPLICATE_THRESHOLD = env_float("DUPLICATE_SIMILARITY_THRESHOLD", 0.80)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(text: str) -> np.ndarray:
    """Embed one text into a normalised 384-dimensional vector."""
    return np.asarray(_load_model().encode(text, normalize_embeddings=True))


def embed_batch(texts: list) -> np.ndarray:
    """Embed multiple texts in one forward pass."""
    return np.asarray(
        _load_model().encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for normalised vectors."""
    return float(np.dot(a, b))


if __name__ == "__main__":
    sentences = [
        "Military exercises near Taiwan rattle semiconductor stocks",
        "Taiwan Strait tensions escalate as China conducts drills",
        "Federal Reserve raises interest rates by 50 basis points",
    ]
    embeddings = embed_batch(sentences)
    print(f"Same-event similarity : {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"Cross-event similarity: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"Fusion threshold      : {SAME_EVENT_THRESHOLD:.2f}")
    print(f"Duplicate threshold   : {DUPLICATE_THRESHOLD:.2f}")
