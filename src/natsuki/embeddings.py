"""Local, CPU-friendly document/query embeddings via fastembed (ONNX runtime).

Deliberately not the Mistral API here: embedding is a bulk, per-document
operation that scales with corpus size, so it belongs on local free compute.
Mistral is reserved for per-query LLM work (see natsuki.config).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ~130MB, ONNX, CPU-friendly

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=DEFAULT_MODEL)
    return _model


def embed_documents(texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
    """Returns an (N, dim) float32 array. Uses passage_embed so fastembed applies
    whatever model-specific prefixing bge-small expects for passages (none, in
    this model's case -- only queries get an instruction prefix)."""
    model = _get_model()
    vectors = list(model.passage_embed(list(texts), batch_size=batch_size))
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    vector = next(model.query_embed(text))
    return np.asarray(vector, dtype=np.float32)
