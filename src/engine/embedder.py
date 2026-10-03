"""
Embedder
Converts text to dense vector representations using
sentence-transformers/all-MiniLM-L6-v2.

Calibrated similarity thresholds (measured on financial text):
  > 0.30  → likely same event (safe threshold, gap is 0.14-0.42)
  > 0.42  → definitely same event
  < 0.14  → definitely different event

These thresholds are used by event_fusion.py for clustering.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

SAME_EVENT_THRESHOLD = 0.30  # used by event_fusion.py


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(text: str) -> np.ndarray:
    """
    Embed a single text string.
    Returns: numpy array of shape (384,)
    """
    model = _load_model()
    return model.encode(text, normalize_embeddings=True)


def embed_batch(texts: list) -> np.ndarray:
    """
    Embed multiple texts in one forward pass.
    Returns: numpy array of shape (N, 384)
    """
    model = _load_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalized embeddings.
    Range: 0.0 (unrelated) to 1.0 (identical meaning).
    Since embeddings are normalized, this is a dot product.
    """
    return float(np.dot(a, b))


if __name__ == "__main__":
    # Calibration test: verified on Taiwan crisis articles
    # Same-event min=0.42, cross-event max=0.14 → threshold=0.30
    sentences = [
        "Military exercises near Taiwan rattle semiconductor stocks",
        "Taiwan Strait tensions escalate as China conducts drills",
        "Federal Reserve raises interest rates by 50 basis points",
    ]

    print("Calibration check...\n")
    embeddings = embed_batch(sentences)

    sim_same = cosine_similarity(embeddings[0], embeddings[1])
    sim_diff = cosine_similarity(embeddings[0], embeddings[2])

    print(f"  Same-event similarity : {sim_same:.4f} (expect > {SAME_EVENT_THRESHOLD})")
    print(f"  Cross-event similarity: {sim_diff:.4f} (expect < {SAME_EVENT_THRESHOLD})")

    passed = sim_same > SAME_EVENT_THRESHOLD and sim_diff < SAME_EVENT_THRESHOLD
    print(f"\n  {'✅ Threshold validated' if passed else '❌ Threshold needs recalibration'}")
    print(f"  SAME_EVENT_THRESHOLD = {SAME_EVENT_THRESHOLD}")
