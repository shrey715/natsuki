"""CLI: build a BM25 index from a public IR dataset, then evaluate it."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _cmd_build_index(args: argparse.Namespace) -> None:
    from natsuki.data import load_corpus
    from natsuki.index import build_index

    corpus = load_corpus(args.dataset)
    t0 = time.time()
    index = build_index(corpus)
    elapsed = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(out_path))

    print(
        f"Indexed {index.N} docs (avgdl={index.avgdl:.1f}, "
        f"vocab={len(index.postings)}) in {elapsed:.1f}s -> {out_path}"
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from natsuki.bm25 import BM25
    from natsuki.data import load_qrels, load_queries
    from natsuki.eval import evaluate
    from natsuki.index import InvertedIndex

    index = InvertedIndex.load(args.index)
    bm25 = BM25(index)

    queries = load_queries(args.dataset)
    qrels = load_qrels(args.dataset)

    results: dict[str, list[str]] = {}
    t0 = time.time()
    for qid, qtext in tqdm(queries.items(), desc="querying", unit="query"):
        hits = bm25.search(qtext, top_k=max(args.k, 100))
        results[qid] = [doc_id for doc_id, _ in hits]
    elapsed = time.time() - t0

    metrics = evaluate(qrels, results, k=args.k)
    avg_latency_ms = 1000 * elapsed / len(queries)

    print(f"\n{len(queries)} queries, {avg_latency_ms:.2f} ms/query avg")
    for name, val in metrics.items():
        if name == "num_queries":
            continue
        print(f"{name}: {val:.4f}")


def _cmd_search(args: argparse.Namespace) -> None:
    from natsuki.bm25 import BM25
    from natsuki.index import InvertedIndex

    index = InvertedIndex.load(args.index)
    bm25 = BM25(index)
    hits = bm25.search(args.query, top_k=args.k)
    for rank, (doc_id, score) in enumerate(hits, 1):
        print(f"{rank:>3}  {score:.4f}  {doc_id}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="natsuki")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build a BM25 index from an ir_datasets id")
    p_build.add_argument("--dataset", required=True, help="e.g. beir/scifact/test")
    p_build.add_argument("--out", required=True, help="output path, e.g. indexes/scifact.index.gz")
    p_build.set_defaults(func=_cmd_build_index)

    p_eval = sub.add_parser("evaluate", help="Evaluate a built index against a dataset's qrels")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--index", required=True)
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.set_defaults(func=_cmd_evaluate)

    p_search = sub.add_parser("search", help="Run a single ad-hoc query against a built index")
    p_search.add_argument("--index", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--k", type=int, default=10)
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
