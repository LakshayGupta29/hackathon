"""
Novelty Scorer.

Measures how new an event is compared with recently seen events:
    novelty = 1 - max cosine similarity to the last MAX_MEMORY events

  >= 0.65  high novelty   (new story)
  >= 0.35  medium novelty (development of an existing story)
  <  0.35  low novelty    (repetition)

Every event is stored in a sliding window; the scorer is thread-safe because
the API scores events from worker threads.
"""

import threading

import numpy as np

from src.engine.embedder import embed

HIGH_NOVELTY = 0.65
MEDIUM_NOVELTY = 0.35


class NoveltyScorer:
    """Stateful scorer shared across one pipeline session."""

    MAX_MEMORY = 100

    def __init__(self):
        self._past_embeddings: list[np.ndarray] = []
        self._past_headlines: list[str] = []
        self._lock = threading.Lock()

    def score(self, text: str, embedding=None) -> dict:
        if embedding is None:
            embedding = embed(text)

        with self._lock:
            if not self._past_embeddings:
                self._store(embedding, text)
                return self._result(1.0, None, 0.0)

            matrix = np.vstack(self._past_embeddings)
            similarities = matrix @ np.asarray(embedding)
            idx = int(np.argmax(similarities))
            max_sim = float(similarities[idx])
            most_similar_text = self._past_headlines[idx]
            self._store(embedding, text)

        novelty = round(max(0.0, min(1.0, 1.0 - max_sim)), 4)
        return self._result(novelty, most_similar_text, max_sim)

    def _store(self, embedding: np.ndarray, text: str):
        self._past_embeddings.append(np.asarray(embedding))
        self._past_headlines.append(text)
        if len(self._past_embeddings) > self.MAX_MEMORY:
            self._past_embeddings = self._past_embeddings[-self.MAX_MEMORY:]
            self._past_headlines = self._past_headlines[-self.MAX_MEMORY:]

    @staticmethod
    def _result(novelty: float, most_similar: str | None, max_sim: float) -> dict:
        if novelty >= HIGH_NOVELTY:
            label = "high"
        elif novelty >= MEDIUM_NOVELTY:
            label = "medium"
        else:
            label = "low"
        return {
            "novelty_score": novelty,
            "novelty_label": label,
            "most_similar_past": most_similar,
            "max_similarity": round(max_sim, 4),
        }

    def reset(self):
        with self._lock:
            self._past_embeddings = []
            self._past_headlines = []

    @property
    def memory_size(self) -> int:
        return len(self._past_embeddings)


if __name__ == "__main__":
    scorer = NoveltyScorer()
    events = [
        "Military exercises near Taiwan rattle semiconductor stocks",
        "Taiwan Strait tensions escalate as China conducts drills",
        "Taiwan crisis deepens, TSMC shares fall sharply",
        "Federal Reserve raises interest rates by 50 basis points",
        "Fed signals further rate hikes as inflation persists",
        "Apple reports record quarterly earnings beating estimates",
    ]
    for text in events:
        result = scorer.score(text)
        print(f"  [{result['novelty_label'].upper():<6}] {result['novelty_score']:.3f} | {text[:60]}")
    print(f"\nMemory size: {scorer.memory_size}")
