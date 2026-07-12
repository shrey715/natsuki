"""Flat (brute-force) dense vector index. Vectors are unit-normalized, so
dot product == cosine similarity. Fine at the doc counts used here (<= a
few million); an HNSW/approximate index is the natural follow-up once
corpus size makes exhaustive search too slow."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DenseIndex:
    doc_ids: list[str] = field(default_factory=list)
    vectors: np.ndarray | None = None  # (N, dim) float32, unit-normalized

    @classmethod
    def build(
        cls,
        corpus: Iterable[tuple[str, str]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> "DenseIndex":
        from tqdm import tqdm

        from natsuki.embeddings import embed_documents

        doc_ids: list[str] = []
        vector_batches: list[np.ndarray] = []
        batch_ids: list[str] = []
        batch_texts: list[str] = []

        def flush():
            if not batch_texts:
                return
            vecs = embed_documents(batch_texts, batch_size=batch_size)
            vector_batches.append(vecs)
            doc_ids.extend(batch_ids)

        items = list(corpus)
        iterator = tqdm(items, desc="embedding", unit="doc") if show_progress else items
        for doc_id, text in iterator:
            batch_ids.append(doc_id)
            batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                flush()
                batch_ids, batch_texts = [], []
        flush()

        vectors = np.vstack(vector_batches) if vector_batches else np.zeros((0, 0), dtype=np.float32)
        return cls(doc_ids=doc_ids, vectors=vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.doc_ids) == 0:
            return []
        scores = self.vectors @ query_vector
        top_k = min(top_k, len(self.doc_ids))
        top_idx = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        # Pass an open file handle so numpy doesn't silently append ".npz"
        # to the path (which would break callers that pass an exact path).
        with open(path, "wb") as f:
            np.savez_compressed(
                f,
                vectors=self.vectors,
                doc_ids=np.array(self.doc_ids, dtype=object),
            )

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True)
            return cls(doc_ids=list(data["doc_ids"]), vectors=data["vectors"])
