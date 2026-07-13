import numpy as np
import pytest

from natsuki.index.dense_index import DenseIndex
from natsuki.index.lsh_index import LSHIndex


def _random_dense_index(n=500, dim=32, seed=0) -> DenseIndex:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    doc_ids = [f"d{i}" for i in range(n)]
    return DenseIndex(doc_ids=doc_ids, vectors=vectors)


def test_finds_exact_match_for_a_stored_vector():
    # Querying with a vector identical to one already indexed must land
    # in the same bucket in every table (hashing is deterministic), so
    # that document should always come back as the top hit.
    dense = _random_dense_index()
    lsh = LSHIndex.build(dense, num_tables=8, num_bits=8, seed=1)

    query = dense.vectors[42].copy()
    results = lsh.search(query, top_k=5)
    assert results[0][0] == "d42"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_candidate_set_is_smaller_than_full_corpus():
    # The whole point of hashing: a query should only be compared against
    # a subset of the corpus, not all of it.
    dense = _random_dense_index(n=2000, dim=64, seed=2)
    lsh = LSHIndex.build(dense, num_tables=4, num_bits=10, seed=3)

    query = np.random.default_rng(4).standard_normal(64).astype(np.float32)
    query /= np.linalg.norm(query)

    _, num_candidates = lsh.search_with_stats(query, top_k=10)
    assert 0 < num_candidates < dense.vectors.shape[0]


def test_recall_against_brute_force_is_reasonably_high():
    # LSH is approximate, so exact agreement isn't guaranteed, but with
    # generous tables/bits on a fixed seed it should recover most of the
    # true top-10 nearest neighbors.
    dense = _random_dense_index(n=1000, dim=32, seed=5)
    lsh = LSHIndex.build(dense, num_tables=16, num_bits=6, seed=6)

    hits = 0
    num_queries = 20
    rng = np.random.default_rng(7)
    for _ in range(num_queries):
        query = rng.standard_normal(32).astype(np.float32)
        query /= np.linalg.norm(query)

        true_top10 = {doc_id for doc_id, _ in dense.search(query, top_k=10)}
        lsh_top10 = {doc_id for doc_id, _ in lsh.search(query, top_k=10)}
        hits += len(true_top10 & lsh_top10)

    recall = hits / (num_queries * 10)
    assert recall > 0.5


def test_more_bits_shrinks_candidate_sets():
    # More hash bits per table -> smaller buckets -> fewer candidates.
    dense = _random_dense_index(n=2000, dim=64, seed=8)
    query = np.random.default_rng(9).standard_normal(64).astype(np.float32)
    query /= np.linalg.norm(query)

    coarse = LSHIndex.build(dense, num_tables=4, num_bits=4, seed=10)
    fine = LSHIndex.build(dense, num_tables=4, num_bits=14, seed=10)

    _, coarse_candidates = coarse.search_with_stats(query, top_k=10)
    _, fine_candidates = fine.search_with_stats(query, top_k=10)
    assert fine_candidates <= coarse_candidates


def test_save_load_roundtrip(tmp_path):
    dense = _random_dense_index(n=100, dim=16, seed=11)
    lsh = LSHIndex.build(dense, num_tables=4, num_bits=6, seed=12)

    path = str(tmp_path / "lsh.pkl.gz")
    lsh.save(path)
    loaded = LSHIndex.load(path)

    query = dense.vectors[0]
    assert lsh.search(query, top_k=5) == loaded.search(query, top_k=5)


def test_no_candidates_returns_empty():
    dense = DenseIndex(doc_ids=["a"], vectors=np.array([[1.0, 0.0]], dtype=np.float32))
    # A huge number of bits with only one point means the query is very
    # unlikely to collide with anything if it points the opposite way.
    lsh = LSHIndex.build(dense, num_tables=1, num_bits=20, seed=13)
    results, count = lsh.search_with_stats(np.array([-1.0, 0.0], dtype=np.float32), top_k=5)
    assert count == 0
    assert results == []
