"""IR eval metrics implemented from scratch: NDCG@k, MRR, Recall@k.

qrels:   {query_id: {doc_id: relevance_grade}}
results: {query_id: [doc_id, ...]}  -- ranked, most relevant first
"""

from __future__ import annotations

import math


def _dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevant: dict[str, int], ranked_doc_ids: list[str], k: int) -> float:
    top = ranked_doc_ids[:k]
    gains = [relevant.get(doc_id, 0) for doc_id in top]
    dcg = _dcg(gains)

    ideal_gains = sorted(relevant.values(), reverse=True)[:k]
    idcg = _dcg(ideal_gains)

    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(relevant: dict[str, int], ranked_doc_ids: list[str]) -> float:
    for i, doc_id in enumerate(ranked_doc_ids):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevant: dict[str, int], ranked_doc_ids: list[str], k: int) -> float:
    total_relevant = sum(1 for rel in relevant.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    top = set(ranked_doc_ids[:k])
    hit = sum(1 for doc_id, rel in relevant.items() if rel > 0 and doc_id in top)
    return hit / total_relevant


def evaluate(
    qrels: dict[str, dict[str, int]],
    results: dict[str, list[str]],
    k: int = 10,
) -> dict[str, float]:
    """Average NDCG@k, MRR, Recall@k over all queries present in both qrels and results."""
    query_ids = [qid for qid in results if qid in qrels]
    if not query_ids:
        raise ValueError("No overlapping query ids between qrels and results")

    ndcgs, rrs, recalls = [], [], []
    for qid in query_ids:
        relevant = qrels[qid]
        ranked = results[qid]
        ndcgs.append(ndcg_at_k(relevant, ranked, k))
        rrs.append(reciprocal_rank(relevant, ranked))
        recalls.append(recall_at_k(relevant, ranked, k))

    n = len(query_ids)
    return {
        f"ndcg@{k}": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
        f"recall@{k}": sum(recalls) / n,
        "num_queries": n,
    }
