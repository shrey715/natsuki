"""KD-tree nearest-neighbor search, from scratch.

Built on unit-normalized vectors, so ranking by Euclidean distance is
equivalent to ranking by cosine similarity: for unit vectors a, b,
||a-b||^2 = 2 - 2*cos_sim(a,b), a monotonic function of cos_sim. That lets
the tree use plain Euclidean branch-and-bound and still answer a cosine
similarity query correctly.

Included mainly as a documented negative result: KD-trees prune well in
low dimensions, but the number of levels a tree can usefully split on is
bounded by its depth (~log2(N)), which is tiny relative to the embedding
dimension (384) here. Most dimensions never get split on, so most branches
can't be pruned, and search degrades toward visiting most of the tree --
see natsuki/index/lsh_index.py for the approach that actually helps at
this dimensionality, and the README for the measured comparison.
"""

from __future__ import annotations

import gzip
import heapq
import pickle

import numpy as np

from natsuki.index.dense_index import DenseIndex


class _Node:
    __slots__ = ("point", "doc_id", "axis", "left", "right")

    def __init__(self, point: np.ndarray, doc_id: str, axis: int):
        self.point = point
        self.doc_id = doc_id
        self.axis = axis
        self.left: _Node | None = None
        self.right: _Node | None = None


def _build(points: np.ndarray, doc_ids: list[str], indices: np.ndarray, depth: int, dim: int) -> _Node | None:
    if len(indices) == 0:
        return None

    axis = depth % dim
    order = indices[np.argsort(points[indices, axis])]
    mid = len(order) // 2
    median_idx = order[mid]

    node = _Node(points[median_idx], doc_ids[median_idx], axis)
    node.left = _build(points, doc_ids, order[:mid], depth + 1, dim)
    node.right = _build(points, doc_ids, order[mid + 1 :], depth + 1, dim)
    return node


class KDTreeIndex:
    def __init__(self, root: _Node | None, dim: int, size: int):
        self.root = root
        self.dim = dim
        self.size = size

    @classmethod
    def build(cls, dense: DenseIndex) -> "KDTreeIndex":
        n, dim = dense.vectors.shape
        indices = np.arange(n)
        root = _build(dense.vectors, dense.doc_ids, indices, depth=0, dim=dim)
        return cls(root=root, dim=dim, size=n)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        results, _ = self.search_with_stats(query_vector, top_k)
        return results

    def search_with_stats(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[list[tuple[str, float]], int]:
        """Same as search(), but also returns the number of tree nodes
        visited -- the metric that shows whether pruning is actually
        doing anything at this dimensionality."""
        # Max-heap on distance, via negation, so the worst of the current
        # top-k sits at heap[0] and can be evicted in O(log k).
        heap: list[tuple[float, str]] = []
        visited = 0

        def recurse(node: _Node | None) -> None:
            nonlocal visited
            if node is None:
                return
            visited += 1

            dist_sq = float(np.sum((node.point - query_vector) ** 2))
            if len(heap) < top_k:
                heapq.heappush(heap, (-dist_sq, node.doc_id))
            elif dist_sq < -heap[0][0]:
                heapq.heapreplace(heap, (-dist_sq, node.doc_id))

            axis = node.axis
            diff = query_vector[axis] - node.point[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            recurse(near)

            # Only descend into the far branch if the splitting
            # hyperplane is closer than our current worst candidate --
            # otherwise nothing on the far side can possibly be closer.
            if len(heap) < top_k or diff * diff < -heap[0][0]:
                recurse(far)

        recurse(self.root)

        # sort by distance ascending (closest/most-similar first)
        results = sorted(heap, key=lambda h: -h[0])
        # convert squared Euclidean distance back to cosine similarity:
        # for unit vectors, ||a-b||^2 = 2 - 2*cos_sim -> cos_sim = 1 - dist_sq/2
        return [(doc_id, 1.0 - (-neg_dist_sq) / 2.0) for neg_dist_sq, doc_id in results], visited

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "KDTreeIndex":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
