"""BM25 ranking over an InvertedIndex, scored from scratch (Robertson/Sparck-Jones IDF)."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from natsuki.index.inverted_index import InvertedIndex
from natsuki.tokenizer import tokenize


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        if not index._finalized:
            raise RuntimeError("Index must be finalized before scoring")
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        df = self.index.doc_frequency(term)
        n = self.index.N
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_terms = set(tokenize(query))
        scores: dict[int, float] = defaultdict(float)

        for term in query_terms:
            postings = self.index.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for internal_id, tf in postings:
                dl = self.index.doc_lengths[internal_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.index.avgdl)
                scores[internal_id] += idf * (tf * (self.k1 + 1)) / denom

        top = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        return [(self.index.doc_ids[internal_id], score) for internal_id, score in top]
