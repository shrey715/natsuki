from natsuki.hybrid import reciprocal_rank_fusion


def test_agreement_boosts_rank():
    # "b" is ranked highly by both retrievers -> should win despite
    # never being #1 in either individual list.
    bm25_ranking = ["a", "b", "c"]
    dense_ranking = ["b", "c", "a"]
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    assert fused[0][0] == "b"


def test_single_ranking_preserves_order():
    ranking = ["x", "y", "z"]
    fused = reciprocal_rank_fusion([ranking])
    assert [doc_id for doc_id, _ in fused] == ["x", "y", "z"]


def test_doc_only_in_one_list_still_included():
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    doc_ids = {doc_id for doc_id, _ in fused}
    assert doc_ids == {"a", "b", "c"}


def test_empty_rankings_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
