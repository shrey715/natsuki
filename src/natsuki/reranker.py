"""LambdaMART learned reranker (LightGBM), trained on CPU.

Replaces RRF's fixed 1/(k+rank) fusion formula with weights learned from
labeled (query, doc, relevance) data -- the model decides how much to trust
BM25 vs. dense signals, rather than a hand-picked constant.
"""

from __future__ import annotations

import numpy as np

from natsuki.features import FEATURE_NAMES


class LambdaMARTReranker:
    def __init__(self, **lgbm_params):
        import lightgbm as lgb

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 5,
            **lgbm_params,
        }
        self.model = lgb.LGBMRanker(**params)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: list[int],
    ) -> None:
        """X: (N, num_features), y: (N,) relevance labels, group: query group sizes summing to N."""
        self.model.fit(X, y, group=group)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() (or load a trained model) before predict()")
        return np.asarray(self.model.predict(X))

    def rank(self, features: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        """features: {doc_id: feature_vector}. Returns doc_ids sorted by predicted score, descending."""
        if not features:
            return []
        doc_ids = list(features.keys())
        X = np.stack([features[d] for d in doc_ids])
        scores = self.predict(X)
        order = np.argsort(-scores)
        return [(doc_ids[i], float(scores[i])) for i in order]

    def feature_importance(self) -> dict[str, float]:
        booster = self.model.booster_ if hasattr(self.model, "booster_") else self.model
        importances = booster.feature_importance(importance_type="gain")
        return dict(zip(FEATURE_NAMES, importances.tolist()))

    def save(self, path: str) -> None:
        self.model.booster_.save_model(path)

    @classmethod
    def load(cls, path: str) -> "LambdaMARTReranker":
        import lightgbm as lgb

        # lgb.Booster.predict() has the same call signature as
        # LGBMRanker.predict(), so it drops straight into self.model here.
        instance = cls.__new__(cls)
        instance.model = lgb.Booster(model_file=path)
        instance._fitted = True
        return instance
