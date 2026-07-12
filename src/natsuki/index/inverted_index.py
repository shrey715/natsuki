"""From-scratch inverted index: term -> postings list of (internal_doc_id, tf)."""

from __future__ import annotations

import gzip
import pickle
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from natsuki.tokenizer import tokenize


@dataclass
class InvertedIndex:
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    N: int = 0
    avgdl: float = 0.0
    _finalized: bool = False
    _doc_id_to_internal: dict[str, int] = field(default_factory=dict)

    def add_document(self, doc_id: str, text: str) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add documents after finalize()")
        if doc_id in self._doc_id_to_internal:
            raise ValueError(f"Duplicate doc_id: {doc_id}")

        internal_id = len(self.doc_ids)
        self._doc_id_to_internal[doc_id] = internal_id
        self.doc_ids.append(doc_id)

        tokens = tokenize(text)
        self.doc_lengths.append(len(tokens))

        term_freqs = Counter(tokens)
        for term, tf in term_freqs.items():
            self.postings.setdefault(term, []).append((internal_id, tf))

    def finalize(self) -> None:
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.doc_lengths) / self.N) if self.N else 0.0
        self._finalized = True

    def doc_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def internal_id(self, doc_id: str) -> int | None:
        return self._doc_id_to_internal.get(doc_id)

    def save(self, path: str) -> None:
        if not self._finalized:
            raise RuntimeError("Call finalize() before save()")
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "InvertedIndex":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def build_index(corpus: Iterable[tuple[str, str]], show_progress: bool = True) -> InvertedIndex:
    """corpus: iterable of (doc_id, text)."""
    from tqdm import tqdm

    index = InvertedIndex()
    iterator = tqdm(corpus, desc="indexing", unit="doc") if show_progress else corpus
    for doc_id, text in iterator:
        index.add_document(doc_id, text)
    index.finalize()
    return index
