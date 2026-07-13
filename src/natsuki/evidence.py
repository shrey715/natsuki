"""Evidence-sentence extraction: given a query and a retrieved document,
pick the sentence(s) most relevant to the query instead of returning the
whole document. Reuses the same embedding model as dense retrieval, at
sentence granularity.

Note: BEIR's SciFact strips the original dataset's sentence-level
evidence annotations (each claim's SUPPORT/CONTRADICT rationale
sentences), so there's no ground truth available here to benchmark
against -- this is a heuristic embedding-based sentence ranker, not a
validated extractive system. The sentence splitter is a plain regex, not
a proper tokenizer, so it will mishandle abbreviations ("Fig. 1", "e.g.")
common in scientific text.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def rank_sentences(query_vector: np.ndarray, sentence_vectors: np.ndarray) -> list[tuple[int, float]]:
    """Returns (sentence_index, cosine_similarity), best first."""
    sims = sentence_vectors @ query_vector
    order = np.argsort(-sims)
    return [(int(i), float(sims[i])) for i in order]


def extract_evidence(
    query: str,
    doc_text: str,
    top_n: int = 1,
    embed_query_fn: Callable[[str], np.ndarray] | None = None,
    embed_documents_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[tuple[str, float]]:
    """Splits doc_text into sentences and returns the top_n most relevant
    to query, each as (sentence, cosine_similarity). embed_*_fn are
    injectable so this is testable without the real embedding model."""
    from natsuki.embeddings import embed_documents, embed_query

    embed_query_fn = embed_query_fn or embed_query
    embed_documents_fn = embed_documents_fn or embed_documents

    sentences = split_sentences(doc_text)
    if not sentences:
        return []

    query_vector = embed_query_fn(query)
    sentence_vectors = embed_documents_fn(sentences)
    ranked = rank_sentences(query_vector, sentence_vectors)
    return [(sentences[i], score) for i, score in ranked[:top_n]]
