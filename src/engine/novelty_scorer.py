"""
Novelty Scorer
Measures how new an event is compared to previously seen events.

Maintains an in-memory store of past event embeddings.
For each new event, computes max cosine similarity vs all past events.
novelty = 1 - max_similarity

Range:
  1.0 = completely new, never seen anything like this
  0.0 = identical to a past event (pure repetition)

Threshold:
  > 0.65 = high novelty  (new event, trigger downstream actions)
  > 0.35 = medium novelty (update to existing event)
  < 0.35 = low novelty   (repetition, suppress downstream actions)
"""

import numpy as np
from src.engine.embedder import embed, cosine_similarity

# Novelty thresholds
HIGH_NOVELTY    = 0.65
MEDIUM_NOVELTY  = 0.35


class NoveltyScorer:
    """
    Stateful scorer that remembers past events.
    One instance should be shared across the pipeline session.
    """

    def __init__(self):
        self._past_embeddings = []   # list of np.ndarray
        self._past_headlines  = []   # for debugging

    def score(self, text: str) -> dict:
        """
        Score the novelty of a new event.

        Args:
            text: headline or representative text of the event
        Returns:
            {
                "novelty_score": float (0-1),
                "novelty_label": "high" | "medium" | "low",
                "most_similar_past": str | None,
                "max_similarity": float
            }
        """
        embedding = embed(text)

        # No past events yet — first event is always maximally novel
        if not self._past_embeddings:
            self._store(embedding, text)
            return self._result(1.0, None, 0.0)

        # Find most similar past event
        similarities = [
            cosine_similarity(embedding, past)
            for past in self._past_embeddings
        ]
        max_sim = max(similarities)
        most_similar_idx = int(np.argmax(similarities))
        most_similar_text = self._past_headlines[most_similar_idx]

        novelty = round(max(0.0, min(1.0, 1.0 - max_sim)), 4)

        # Always store - sliding window handles memory size
        # Removing the novelty gate which caused collapse in long sessions
        # (gate prevented diverse events from entering memory, making
        #  everything look similar to the small fixed set that did enter)
        self._store(embedding, text)

        return self._result(novelty, most_similar_text, max_sim)

    MAX_MEMORY = 100  # sliding window - only compare against last 100 events

    def _store(self, embedding: np.ndarray, text: str):
        self._past_embeddings.append(embedding)
        self._past_headlines.append(text)
        # Evict oldest entries beyond sliding window
        if len(self._past_embeddings) > self.MAX_MEMORY:
            self._past_embeddings = self._past_embeddings[-self.MAX_MEMORY:]
            self._past_headlines  = self._past_headlines[-self.MAX_MEMORY:]

    def _result(self, novelty: float, most_similar: str, max_sim: float) -> dict:
        if novelty >= HIGH_NOVELTY:
            label = "high"
        elif novelty >= MEDIUM_NOVELTY:
            label = "medium"
        else:
            label = "low"

        return {
            "novelty_score":      novelty,
            "novelty_label":      label,
            "most_similar_past":  most_similar,
            "max_similarity":     round(max_sim, 4)
        }

    def reset(self):
        """Clear memory. Used between replay sessions."""
        self._past_embeddings = []
        self._past_headlines  = []

    @property
    def memory_size(self) -> int:
        return len(self._past_embeddings)


if __name__ == "__main__":
    scorer = NoveltyScorer()

    events = [
        "Military exercises near Taiwan rattle semiconductor stocks",
        "Taiwan Strait tensions escalate as China conducts drills",   # similar
        "Taiwan crisis deepens, TSMC shares fall sharply",            # similar
        "Federal Reserve raises interest rates by 50 basis points",   # new topic
        "Fed signals further rate hikes as inflation persists",        # similar to Fed
        "Apple reports record quarterly earnings beating estimates",   # new topic
    ]

    print("Novelty scoring sequence:\n")
    for text in events:
        result = scorer.score(text)
        label = result["novelty_label"].upper()
        score = result["novelty_score"]
        print(f"  [{label:<6}] {score:.3f} | {text[:60]}")
        if result["most_similar_past"]:
            print(f"           most similar: {result['most_similar_past'][:55]}")

    print(f"\nMemory size: {scorer.memory_size} stored events")
    print("(Only novel events stored, repetitions suppressed)")
