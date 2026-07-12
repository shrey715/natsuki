"""Reciprocal Rank Fusion: combine multiple ranked lists without needing
their scores to be on comparable scales (BM25 scores and cosine
similarities aren't)."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """rankings: one ranked list of doc_ids per retriever (best first).
    Returns doc_ids sorted by fused score, descending.

    RRF score for a doc = sum over rankers of 1 / (k + rank), rank 0-indexed.
    Standard choice is k=60 (from the original RRF paper); it downweights
    the exact rank position so one retriever's #1 pick doesn't
    automatically dominate.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
