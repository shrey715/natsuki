import pytest

from natsuki.eval import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank

QRELS_Q1 = {"d1": 1, "d2": 0, "d3": 2}


def test_ndcg_perfect_ranking_is_one():
    ranked = ["d3", "d1", "d2"]  # already sorted by relevance desc
    assert ndcg_at_k(QRELS_Q1, ranked, k=3) == pytest.approx(1.0)


def test_ndcg_penalizes_suboptimal_order():
    ranked = ["d1", "d3", "d2"]  # d3 (rel=2) should be first, isn't
    score = ndcg_at_k(QRELS_Q1, ranked, k=3)
    assert 0.0 < score < 1.0
    assert score == pytest.approx(0.79665, abs=1e-4)


def test_reciprocal_rank_finds_first_relevant():
    assert reciprocal_rank(QRELS_Q1, ["d2", "d1", "d3"]) == pytest.approx(0.5)
    assert reciprocal_rank(QRELS_Q1, ["d1", "d2", "d3"]) == pytest.approx(1.0)
    assert reciprocal_rank({"d1": 0}, ["d1"]) == 0.0


def test_recall_at_k():
    # 2 relevant docs total (d1, d3); top-2 misses d3
    assert recall_at_k(QRELS_Q1, ["d1", "d2", "d3"], k=1) == pytest.approx(0.5)
    assert recall_at_k(QRELS_Q1, ["d1", "d2", "d3"], k=3) == pytest.approx(1.0)


def test_evaluate_aggregates_over_queries():
    qrels = {"q1": QRELS_Q1, "q2": {"e1": 1}}
    results = {"q1": ["d3", "d1", "d2"], "q2": ["e1"]}
    metrics = evaluate(qrels, results, k=3)
    assert metrics["num_queries"] == 2
    assert metrics["ndcg@3"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall@3"] == pytest.approx(1.0)


def test_evaluate_raises_on_no_overlap():
    with pytest.raises(ValueError):
        evaluate({"q1": QRELS_Q1}, {"q2": ["d1"]}, k=3)
