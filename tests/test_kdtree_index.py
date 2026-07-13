import numpy as np
import pytest

from natsuki.index.dense_index import DenseIndex
from natsuki.index.kdtree_index import KDTreeIndex


def _random_dense_index(n=200, dim=8, seed=0) -> DenseIndex:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    doc_ids = [f"d{i}" for i in range(n)]
    return DenseIndex(doc_ids=doc_ids, vectors=vectors)


def test_matches_brute_force_exactly():
    # KD-tree search is exact, not approximate -- it must agree with
    # brute-force cosine similarity search doc-for-doc, in order.
    dense = _random_dense_index()
    kdtree = KDTreeIndex.build(dense)
    query = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    query /= np.linalg.norm(query)

    expected = [doc_id for doc_id, _ in dense.search(query, top_k=10)]
    actual = [doc_id for doc_id, _ in kdtree.search(query, top_k=10)]
    assert actual == expected


def test_scores_match_brute_force():
    dense = _random_dense_index()
    kdtree = KDTreeIndex.build(dense)
    query = np.random.default_rng(2).standard_normal(8).astype(np.float32)
    query /= np.linalg.norm(query)

    expected = dict(dense.search(query, top_k=10))
    actual = dict(kdtree.search(query, top_k=10))
    for doc_id, score in actual.items():
        assert score == pytest.approx(expected[doc_id], abs=1e-4)


def test_search_with_stats_reports_visited_nodes():
    dense = _random_dense_index()
    kdtree = KDTreeIndex.build(dense)
    query = np.random.default_rng(3).standard_normal(8).astype(np.float32)
    query /= np.linalg.norm(query)

    results, visited = kdtree.search_with_stats(query, top_k=10)
    assert len(results) == 10
    assert 0 < visited <= dense.vectors.shape[0]


def test_high_dimensional_search_visits_most_nodes():
    # The documented negative result: at high dimensionality relative to
    # tree depth, pruning barely helps, so most nodes get visited anyway.
    dense = _random_dense_index(n=500, dim=384, seed=5)
    kdtree = KDTreeIndex.build(dense)
    query = np.random.default_rng(6).standard_normal(384).astype(np.float32)
    query /= np.linalg.norm(query)

    _, visited = kdtree.search_with_stats(query, top_k=10)
    assert visited > 0.8 * dense.vectors.shape[0]


def test_single_point_tree():
    dense = DenseIndex(doc_ids=["only"], vectors=np.array([[1.0, 0.0]], dtype=np.float32))
    kdtree = KDTreeIndex.build(dense)
    results = kdtree.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
    assert results == [("only", pytest.approx(1.0, abs=1e-4))]


def test_save_load_roundtrip(tmp_path):
    dense = _random_dense_index()
    kdtree = KDTreeIndex.build(dense)
    query = np.random.default_rng(4).standard_normal(8).astype(np.float32)
    query /= np.linalg.norm(query)

    path = str(tmp_path / "kdtree.pkl.gz")
    kdtree.save(path)
    loaded = KDTreeIndex.load(path)

    assert loaded.search(query, top_k=10) == kdtree.search(query, top_k=10)
