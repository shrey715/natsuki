import numpy as np
import pytest

from natsuki.bm25 import BM25
from natsuki.features import candidate_features
from natsuki.index.dense_index import DenseIndex
from natsuki.index.inverted_index import build_index

CORPUS = [
    ("d1", "cats are great pets"),
    ("d2", "dogs are loyal companions"),
    ("d3", "the stock market rallied today"),
]


def _bm25():
    return BM25(build_index(CORPUS, show_progress=False))


def _dense():
    # Hand-crafted vectors: d1 and d2 close together, d3 far away.
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return DenseIndex(doc_ids=["d1", "d2", "d3"], vectors=vectors / np.linalg.norm(vectors, axis=1, keepdims=True))


def test_candidate_set_is_union_of_both_retrievers():
    bm25 = _bm25()
    dense = _dense()
    query_vector = np.array([1.0, 0.0], dtype=np.float32)

    features = candidate_features("cats pets", bm25, dense, fanout=3, query_vector=query_vector)

    # d1 matches BM25 (has "cats"/"pets") and is closest in dense space.
    assert "d1" in features
    assert features["d1"][4] == 1.0  # in_both


def test_doc_only_in_dense_has_zero_bm25_features():
    bm25 = _bm25()
    dense = _dense()
    # query with no term overlap with any doc -> BM25 returns nothing
    query_vector = np.array([1.0, 0.0], dtype=np.float32)

    features = candidate_features("zzz_no_overlap", bm25, dense, fanout=3, query_vector=query_vector)

    assert features["d1"][0] == 0.0  # bm25_score
    assert features["d1"][1] == 0.0  # bm25_rank_recip
    assert features["d1"][3] > 0.0  # dense_rank_recip
    assert features["d1"][4] == 0.0  # not in both


def test_feature_vector_length_matches_names():
    from natsuki.features import FEATURE_NAMES

    bm25 = _bm25()
    dense = _dense()
    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    features = candidate_features("cats", bm25, dense, fanout=3, query_vector=query_vector)
    for vec in features.values():
        assert len(vec) == len(FEATURE_NAMES)
