"""Thin wrapper around ir_datasets for loading BEIR-style corpora/queries/qrels."""

from __future__ import annotations

from collections.abc import Iterator


def load_corpus(dataset_id: str) -> Iterator[tuple[str, str]]:
    """Yields (doc_id, text) pairs. Title (if present) is prepended to the body."""
    import ir_datasets

    dataset = ir_datasets.load(dataset_id)
    for doc in dataset.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        full_text = f"{title}. {text}" if title else text
        yield doc.doc_id, full_text


def load_queries(dataset_id: str) -> dict[str, str]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_id)
    return {q.query_id: q.text for q in dataset.queries_iter()}


def load_qrels(dataset_id: str) -> dict[str, dict[str, int]]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_id)
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels
