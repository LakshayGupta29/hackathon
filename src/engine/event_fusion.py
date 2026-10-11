"""
Event Fusion / Deduplication.

Groups reports of the same underlying story using sentence embeddings and
agglomerative clustering with COMPLETE linkage on cosine distance.

Why complete linkage instead of DBSCAN: DBSCAN chains points (A~B and B~C
puts A and C together even if they are unrelated). Complete linkage only
merges groups whose *every* pair clears the similarity threshold.

Example:
    result = fuse_events([{"event_id": "1", "headline": "..."}, ...])
    for event in result["events"]:
        print(event["event_cluster_id"])
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.engine.embedder import SAME_EVENT_THRESHOLD, embed_batch


def _cluster_labels(embeddings: np.ndarray, similarity_threshold: float) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="complete",
        distance_threshold=1.0 - float(similarity_threshold),
    )
    return model.fit_predict(embeddings)


def fuse_events(
    events: list[dict],
    similarity_threshold: float = SAME_EVENT_THRESHOLD,
    min_samples: int = 2,
) -> dict:
    """Cluster a batch of events by semantic similarity."""
    if not events:
        return {"events": [], "clusters": [], "similarity_threshold": similarity_threshold}

    texts = [e.get("headline") or e.get("text") or "" for e in events]
    embeddings = np.asarray(embed_batch(texts), dtype=float)
    raw_labels = _cluster_labels(embeddings, similarity_threshold)

    members: dict[int, list[int]] = {}
    for idx, label in enumerate(raw_labels):
        members.setdefault(int(label), []).append(idx)

    clusters = []
    cluster_of: dict[int, dict] = {}
    for label, indices in sorted(members.items(), key=lambda kv: kv[1][0]):
        if len(indices) < min_samples:
            continue
        centroid = embeddings[indices].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        rep_idx = indices[int(np.argmax(embeddings[indices] @ centroid))]
        cluster = {
            "cluster_id": f"cluster_{len(clusters)}",
            "size": len(indices),
            "representative_index": rep_idx,
            "event_ids": [events[i].get("event_id") for i in indices],
            "representative_headline": texts[rep_idx],
        }
        clusters.append(cluster)
        for i in indices:
            cluster_of[i] = cluster

    annotated = []
    for idx, event in enumerate(events):
        out = dict(event)
        cluster = cluster_of.get(idx)
        if cluster is None:
            out["event_cluster_id"] = f"singleton_{idx}"
            out["cluster_size"] = 1
            out["is_cluster_representative"] = True
        else:
            out["event_cluster_id"] = cluster["cluster_id"]
            out["cluster_size"] = cluster["size"]
            out["is_cluster_representative"] = cluster["representative_index"] == idx
        annotated.append(out)

    return {
        "events": annotated,
        "clusters": clusters,
        "similarity_threshold": round(float(similarity_threshold), 4),
        "linkage": "complete",
        "min_samples": min_samples,
        "cluster_count": len(clusters),
        "noise_count": sum(1 for i in range(len(events)) if i not in cluster_of),
    }


# Friendly alias for callers.
cluster_events = fuse_events


if __name__ == "__main__":
    sample = [
        {"event_id": "1", "headline": "Taiwan drills rattle chip stocks"},
        {"event_id": "2", "headline": "China military exercises raise Taiwan tensions"},
        {"event_id": "3", "headline": "Federal Reserve raises rates by 50 bps"},
    ]
    result = fuse_events(sample)
    for e in result["events"]:
        print(e["event_cluster_id"], e["cluster_size"], e["headline"])
