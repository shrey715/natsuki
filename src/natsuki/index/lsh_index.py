"""Locality-sensitive hashing for cosine similarity, from scratch.

Random hyperplane hashing (SimHash-style): each of L hash tables projects
every vector onto num_bits random hyperplanes and keeps the sign bits as
a bucket key. Two vectors landing in the same bucket in any table become
search candidates; candidates get exactly reranked by real cosine
similarity before returning the top-k. This is the technique that should
actually reduce the fraction of the corpus touched per query at this
dimensionality (384), unlike the KD-tree in kdtree_index.py -- see the
README for the measured comparison.

Recall/latency tradeoff: more bits per table -> smaller buckets -> fewer
candidates but lower recall; more tables -> higher recall (more chances
to collide) at the cost of more memory and hashing work.
"""

from __future__ import annotations

import gzip
import pickle
from collections import defaultdict

import numpy as np

from natsuki.index.dense_index import DenseIndex


class LSHIndex:
    def __init__(
        self,
        hyperplanes: list[np.ndarray],
        tables: list[dict[bytes, list[int]]],
        doc_ids: list[str],
        vectors: np.ndarray,
    ):
        self.hyperplanes = hyperplanes  # one (num_bits, dim) array per table
        self.tables = tables
        self.doc_ids = doc_ids
        self.vectors = vectors

    @classmethod
    def build(
        cls,
        dense: DenseIndex,
        num_tables: int = 8,
        num_bits: int = 12,
        seed: int = 0,
    ) -> "LSHIndex":
        rng = np.random.default_rng(seed)
        dim = dense.vectors.shape[1]
        hyperplanes = [rng.standard_normal((num_bits, dim)).astype(np.float32) for _ in range(num_tables)]

        tables: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(num_tables)]
        for table_idx, plane in enumerate(hyperplanes):
            codes = _hash_batch(dense.vectors, plane)
            for doc_idx, code in enumerate(codes):
                tables[table_idx][code].append(doc_idx)

        return cls(
            hyperplanes=hyperplanes,
            tables=[dict(t) for t in tables],
            doc_ids=list(dense.doc_ids),
            vectors=dense.vectors,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        results, _ = self.search_with_stats(query_vector, top_k)
        return results

    def search_with_stats(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[list[tuple[str, float]], int]:
        """Same as search(), but also returns the candidate-set size --
        the fraction of the corpus actually compared, which is the real
        payoff metric for a hash-bucket approach."""
        candidate_ids: set[int] = set()
        for table, plane in zip(self.tables, self.hyperplanes):
            code = _hash_one(query_vector, plane)
            candidate_ids.update(table.get(code, ()))

        if not candidate_ids:
            return [], 0

        candidates = np.fromiter(candidate_ids, dtype=np.int64)
        sims = self.vectors[candidates] @ query_vector
        top_k = min(top_k, len(candidates))
        order = np.argpartition(-sims, top_k - 1)[:top_k]
        order = order[np.argsort(-sims[order])]
        results = [(self.doc_ids[int(candidates[i])], float(sims[i])) for i in order]
        return results, len(candidates)

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "LSHIndex":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def _hash_batch(vectors: np.ndarray, plane: np.ndarray) -> list[bytes]:
    bits = (vectors @ plane.T) >= 0  # (N, num_bits) booleans
    packed = np.packbits(bits, axis=1)
    return [row.tobytes() for row in packed]


def _hash_one(vector: np.ndarray, plane: np.ndarray) -> bytes:
    bits = (plane @ vector) >= 0  # (num_bits,) booleans
    return np.packbits(bits).tobytes()
