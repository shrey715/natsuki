"""Candidate generation + feature extraction for the learned reranker.

For a query, the candidate set is the union of BM25's and dense retrieval's
top-`fanout` hits. Each candidate gets a feature vector built from where (and
how well) each retriever ranked it -- this is what LambdaMART learns to
combine, instead of RRF's fixed 1/(k+rank) formula.
"""

from __future__ import annotations

import numpy as np

from natsuki.bm25 import BM25
from natsuki.embeddings import embed_query
from natsuki.index.dense_index import DenseIndex

FEATURE_NAMES = ["bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip", "in_both"]


def candidate_features(
    query: str,
    bm25: BM25,
    dense: DenseIndex,
    fanout: int = 100,
    query_vector: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Returns {doc_id: feature_vector} for the union of both retrievers' top-fanout hits.

    query_vector can be supplied directly (e.g. in tests) to avoid depending
    on the real embedding model; defaults to embed_query(query).
    """
    if query_vector is None:
        query_vector = embed_query(query)

    bm25_hits = bm25.search(query, top_k=fanout)
    dense_hits = dense.search(query_vector, top_k=fanout)

    bm25_score = dict(bm25_hits)
    bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_hits, start=1)}
    dense_score = dict(dense_hits)
    dense_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_hits, start=1)}

    candidate_ids = set(bm25_score) | set(dense_score)
    features: dict[str, np.ndarray] = {}
    for doc_id in candidate_ids:
        in_bm25 = doc_id in bm25_score
        in_dense = doc_id in dense_score
        features[doc_id] = np.array(
            [
                bm25_score.get(doc_id, 0.0),
                1.0 / bm25_rank[doc_id] if in_bm25 else 0.0,
                dense_score.get(doc_id, 0.0),
                1.0 / dense_rank[doc_id] if in_dense else 0.0,
                1.0 if (in_bm25 and in_dense) else 0.0,
            ],
            dtype=np.float32,
        )
    return features
