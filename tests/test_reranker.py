import numpy as np

from natsuki.reranker import LambdaMARTReranker


def _synthetic_training_set(num_queries: int = 30, docs_per_query: int = 8, seed: int = 0):
    """Feature 0 is a strong relevance signal, feature 1 is pure noise --
    a working ranker should learn to weight feature 0 much more heavily."""
    rng = np.random.default_rng(seed)
    X, y, group = [], [], []
    for _ in range(num_queries):
        signal = rng.random(docs_per_query)
        noise = rng.random(docs_per_query)
        X.append(np.stack([signal, noise], axis=1))
        # top-2 by signal are relevant
        labels = np.zeros(docs_per_query, dtype=int)
        labels[np.argsort(-signal)[:2]] = 1
        y.append(labels)
        group.append(docs_per_query)
    return np.vstack(X), np.concatenate(y), group


def test_fit_and_predict_shapes():
    X, y, group = _synthetic_training_set()
    reranker = LambdaMARTReranker(n_estimators=20)
    reranker.fit(X, y, group)
    scores = reranker.predict(X)
    assert scores.shape == (X.shape[0],)


def test_predict_before_fit_raises():
    reranker = LambdaMARTReranker()
    try:
        reranker.predict(np.zeros((3, 2)))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_learns_to_weight_signal_over_noise():
    # Uses the booster's raw feature_importance directly rather than
    # LambdaMARTReranker.feature_importance(), since that method's names
    # come from the real 5-feature FEATURE_NAMES and this synthetic
    # dataset only has 2 (signal, noise) -- not a real feature set.
    X, y, group = _synthetic_training_set(num_queries=50)
    reranker = LambdaMARTReranker(n_estimators=50)
    reranker.fit(X, y, group)

    raw_importance = reranker.model.booster_.feature_importance(importance_type="gain")
    signal_importance, noise_importance = raw_importance[0], raw_importance[1]
    assert signal_importance > noise_importance


def test_rank_returns_sorted_doc_ids():
    X, y, group = _synthetic_training_set()
    reranker = LambdaMARTReranker(n_estimators=20)
    reranker.fit(X, y, group)

    features = {"a": X[0], "b": X[1], "c": X[2]}
    ranked = reranker.rank(features)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert {doc_id for doc_id, _ in ranked} == {"a", "b", "c"}


def test_save_load_roundtrip(tmp_path):
    X, y, group = _synthetic_training_set()
    reranker = LambdaMARTReranker(n_estimators=20)
    reranker.fit(X, y, group)
    original_scores = reranker.predict(X)

    path = str(tmp_path / "model.txt")
    reranker.save(path)
    loaded = LambdaMARTReranker.load(path)
    loaded_scores = loaded.predict(X)

    np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)
