"""Tests use hand-crafted vectors, not the real embedding model -- keeps the
suite fast and offline. Embedding correctness is checked separately/manually
(see natsuki.embeddings), since it requires downloading an ONNX model."""

import numpy as np
import pytest

from natsuki.index.dense_index import DenseIndex


def _toy_index() -> DenseIndex:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # d1: pure "x"
            [0.0, 1.0, 0.0],  # d2: pure "y"
            [0.70710678, 0.70710678, 0.0],  # d3: halfway between x and y
        ],
        dtype=np.float32,
    )
    return DenseIndex(doc_ids=["d1", "d2", "d3"], vectors=vectors)


def test_search_ranks_by_cosine_similarity():
    index = _toy_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=3)
    ranked_ids = [doc_id for doc_id, _ in results]
    assert ranked_ids[0] == "d1"
    assert ranked_ids[-1] == "d2"


def test_search_scores_are_cosine_values():
    index = _toy_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = dict(index.search(query, top_k=3))
    assert results["d1"] == pytest.approx(1.0)
    assert results["d2"] == pytest.approx(0.0)
    assert results["d3"] == pytest.approx(0.70710678, abs=1e-5)


def test_top_k_is_capped_at_corpus_size():
    index = _toy_index()
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert len(index.search(query, top_k=100)) == 3


def test_empty_index_returns_empty():
    index = DenseIndex(doc_ids=[], vectors=np.zeros((0, 3), dtype=np.float32))
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert index.search(query, top_k=10) == []


def test_save_load_roundtrip(tmp_path):
    index = _toy_index()
    path = tmp_path / "dense.npz"
    index.save(str(path))
    loaded = DenseIndex.load(str(path))

    assert loaded.doc_ids == index.doc_ids
    np.testing.assert_allclose(loaded.vectors, index.vectors)
