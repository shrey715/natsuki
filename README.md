# natsuki

A hybrid BM25 + dense retrieval engine with a learned reranker, built from
scratch (no BM25/IR library dependency) and evaluated against public
benchmarks with real relevance judgments.

## Status

**Phase 1 (done):** from-scratch inverted index + BM25.

- `src/natsuki/tokenizer.py` — regex tokenizer + stopword filtering.
- `src/natsuki/index/inverted_index.py` — term -> postings (doc_id, tf),
  gzip+pickle persistence.
- `src/natsuki/bm25.py` — Robertson/Sparck-Jones IDF, standard BM25 scoring
  (k1=1.5, b=0.75), top-k via heap.
- `src/natsuki/eval.py` — NDCG@k, MRR, Recall@k implemented from scratch
  (no pytrec_eval — see note below).
- `src/natsuki/data.py` — loads any `ir_datasets` corpus (BEIR sets, MS
  MARCO, etc.) into `(doc_id, text)` / queries / qrels.

**Phase 2 (done):** dense retrieval + hybrid fusion.

- `src/natsuki/embeddings.py` — local CPU embeddings via `fastembed`
  (`BAAI/bge-small-en-v1.5`, ONNX runtime, no torch/GPU dependency).
- `src/natsuki/index/dense_index.py` — flat cosine-similarity vector index.
- `src/natsuki/hybrid.py` — reciprocal rank fusion (RRF, k=60) combining
  BM25 and dense rankings.
- `src/natsuki/cli.py` — `natsuki build-index`, `build-dense-index`,
  `evaluate --mode {bm25,dense,hybrid}`, `search`.

**Validated on BEIR/SciFact** (5,183 docs, 300 test queries, real qrels):

| mode   | ndcg@10 | mrr    | recall@10 | ms/query |
|--------|---------|--------|-----------|----------|
| bm25   | 0.6692  | 0.6432 | 0.7856    | 0.74     |
| dense  | 0.7215  | 0.6932 | 0.8396    | 42.10    |
| hybrid | 0.7245  | 0.6988 | 0.8379    | 48.19    |

The published BEIR paper reports a BM25 baseline of **NDCG@10 ≈ 0.665** on
SciFact — this from-scratch implementation lands at 0.6692, matching the
reference almost exactly. That's the correctness check for the whole
tokenizer -> index -> BM25 pipeline before anything else gets layered on.

Dense retrieval alone beats BM25 by a wide margin here (SciFact is a
scientific-claim-matching task where semantic similarity carries a lot of
the signal), and RRF fusion edges out dense-only on both NDCG and MRR —
combining the two ranking signals helps even when one is individually much
stronger, at roughly the cost of the slower retriever (dense embedding of
the query dominates hybrid latency; BM25's own contribution is sub-ms).

Document embedding on this CPU-only laptop ran at **~5.3 docs/sec**
(5,183 docs in ~16 minutes) — a real throughput number for the "why the
architecture looks like this" hardware-notes section below, and the basis
for the ~500k-1M document corpus-size target (that throughput puts a
full 1M-doc embed job at roughly two days of wall-clock time, so a larger
run needs either a faster model, batching optimization, or to be treated
as a genuine overnight/multi-day batch job rather than an interactive step).

**Not yet built:**

- LambdaMART (LightGBM) learned reranker on top of BM25+dense candidates.
- Query understanding / LLM-judge eval via Mistral API.
- Real-time incremental indexing.
- Permission-aware filtering (deliberately descoped for now).

## Why no pytrec_eval

`pytrec_eval` failed to compile on this machine — its bundled `trec_eval`
C source is old-style C89 (implicit qsort comparator casts) that trips
strict prototype checking on newer GCC. Rather than patch upstream C, NDCG/
MRR/Recall@k are implemented directly in `natsuki/eval.py`, matching the
standard formulas (DCG with 2^rel-1 gain, log2 discount). The SciFact
number above matching the published BEIR baseline is the validation that
these are correct.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in MISTRAL_API_KEY
```

`.env` is gitignored — never commit it. `MISTRAL_API_KEY` isn't used yet
(Phase 1 is pure classical IR, no LLM calls); it's wired into
`src/natsuki/config.py` ready for Phase 2 (dense embeddings / reranking /
query understanding).

## Usage

```bash
# Build a BM25 index from any ir_datasets id
uv run natsuki build-index --dataset beir/scifact/test --out indexes/scifact.index.gz

# Build a local dense (embedding) index for the same corpus
uv run natsuki build-dense-index --dataset beir/scifact/test --out indexes/scifact.dense.npz

# Evaluate against that dataset's real relevance judgments
uv run natsuki evaluate --dataset beir/scifact/test --mode bm25 --index indexes/scifact.index.gz --k 10
uv run natsuki evaluate --dataset beir/scifact/test --mode dense --dense-index indexes/scifact.dense.npz --k 10
uv run natsuki evaluate --dataset beir/scifact/test --mode hybrid \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz --k 10

# Ad-hoc single BM25 query
uv run natsuki search --index indexes/scifact.index.gz --query "your query here" --k 10
```

## Tests

```bash
uv run pytest -q
```

22 unit tests covering tokenizer edge cases, BM25 ranking behavior (term
frequency ordering, no-match handling, unfinalized-index guard), dense
index search/persistence, reciprocal rank fusion, and eval metrics against
hand-computed NDCG/MRR/Recall values. Dense-index tests use hand-crafted
vectors rather than the real embedding model, so the suite stays fast and
offline — embedding correctness itself was checked manually (see
`natsuki/embeddings.py` docstring) and validated indirectly by the SciFact
NDCG numbers above matching the published baseline.

## Hardware notes (why the architecture looks like this)

Built on a laptop with no discrete GPU (Intel i7-12650H, 14GB RAM,
integrated graphics only) — no local model fine-tuning or high-throughput
inference. Decisions this drove:

- **Document embeddings** (bulk, one-time cost, scales with corpus size):
  `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX runtime), *not* the
  Mistral API — embedding hundreds of thousands of documents through a
  paid API is slow and expensive; this is a local, free job. Measured
  throughput: ~5.3 docs/sec single-threaded on this CPU.
- **Query-time LLM work** (query understanding, reranking prompts,
  LLM-as-judge eval): Mistral API — call volume is per-query, not
  per-document, so it's cheap and doesn't need local compute.
- **Learned reranker**: LambdaMART (LightGBM) trained on CPU — no GPU
  dependency, trains in minutes even on a few hundred thousand rows.

Target corpus size for the full hybrid pipeline: at ~5.3 docs/sec, a
500k-document embed job is ~26 hours and a 1M-document job ~2 days —
both feasible as an overnight/multi-day batch job, but they set the
practical ceiling. Next optimization to revisit if a larger corpus is
needed: increase `fastembed`'s ONNX thread count (currently
single-threaded despite 16 CPU threads being available) before assuming a
smaller model or corpus subsample is required.
