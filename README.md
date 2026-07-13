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

**Phase 3 (done):** learned reranker.

- `src/natsuki/features.py` — candidate set = union of BM25's and dense
  retrieval's top-fanout hits; features = each retriever's score and
  reciprocal rank (0 if not retrieved by that method) plus an in_both flag.
- `src/natsuki/reranker.py` — LambdaMART (`lightgbm.LGBMRanker`, CPU),
  trained on labeled (query, doc) candidates instead of hand-picking a
  fusion formula.
- `src/natsuki/cli.py` — `natsuki train-reranker`, `evaluate --mode rerank`.

**Validated on BEIR/SciFact** (5,183 docs, 300 test queries, real qrels;
reranker trained on the separate 809-query train split, no leakage):

| mode   | ndcg@10 | mrr    | recall@10 | ms/query |
|--------|---------|--------|-----------|----------|
| bm25   | 0.6692  | 0.6432 | 0.7856    | 0.74     |
| dense  | 0.7215  | 0.6932 | 0.8396    | 42.10    |
| hybrid | 0.7245  | 0.6988 | 0.8379    | 48.19    |
| rerank | 0.7459  | 0.7144 | 0.8762    | 52.38    |

The reranker beats hybrid RRF on every metric: NDCG@10 0.7459 vs 0.7245,
MRR 0.7144 vs 0.6988, Recall@10 0.8762 vs 0.8379 — a clean monotonic
improvement across all four modes (bm25 < dense < hybrid < rerank).
Trained on 809 queries / 138,103 (query, doc) candidate pairs (906
positive) in ~40s. Feature importance (gain) came out heavily skewed
toward `dense_rank_recip` (~103.6k) over everything else (bm25 rank ~6.1k,
bm25/dense raw scores ~2.2-2.4k, in_both ~0.4k) — consistent with dense
retrieval's much stronger standalone numbers above; the model learned
that skew from data rather than it being hand-tuned in.

The published BEIR paper reports a BM25 baseline of **NDCG@10 ≈ 0.665** on
SciFact. This from-scratch implementation lands at 0.6692, matching the
reference almost exactly. That's the correctness check for the whole
tokenizer -> index -> BM25 pipeline before anything else gets layered on.

Dense retrieval alone beats BM25 by a wide margin here (SciFact is a
scientific-claim-matching task where semantic similarity carries a lot of
the signal), and RRF fusion edges out dense-only on both NDCG and MRR —
combining the two ranking signals helps even when one is individually much
stronger, at roughly the cost of the slower retriever (dense embedding of
the query dominates hybrid latency; BM25's own contribution is sub-ms).

Document embedding on a CPU-only laptop ran at **~5.3 docs/sec**
(5,183 docs in ~16 minutes) — a real throughput number for the "why the
architecture looks like this" hardware-notes section below, and the basis
for the ~500k-1M document corpus-size target (that throughput puts a
full 1M-doc embed job at roughly two days of wall-clock time, so a larger
run needs either a faster model, batching optimization, or to be treated
as a genuine overnight/multi-day batch job rather than an interactive step).

**Phase 4 (done):** LLM query expansion via the Mistral API.

- `src/natsuki/llm.py` — thin wrapper around `mistralai`'s chat API.
  Note: this SDK build (2.6.0) ships with no root-level
  `mistralai/__init__.py`; the working import is
  `from mistralai.client import Mistral`, not the documented
  `from mistralai import Mistral`.
- `src/natsuki/query_understanding.py` — `expand_query()` appends
  Mistral-generated related terms/synonyms to a query before retrieval,
  falling back to the original query unchanged on any API failure.
- `src/natsuki/cli.py` — `evaluate --expand-query`, `evaluate --limit N`
  (query expansion adds a network round-trip per query, so `--limit` keeps
  experiments against a paid API cheap/fast).

**A/B result on 100 held-out test queries** (rerank mode, same subset with
and without expansion):

| variant         | ndcg@10 | mrr    | recall@10 | ms/query |
|-----------------|---------|--------|-----------|----------|
| rerank          | 0.7855  | 0.7570 | 0.9097    | 67.89    |
| rerank + expand | 0.7708  | 0.7321 | 0.9197    | 453.63   |

Query expansion **hurt** top-of-ranking precision (NDCG@10, MRR both
dropped) while modestly improving recall@10, and added ~400ms/query of
pure network latency. Read honestly rather than as a failure to hide: on
SciFact, queries are precise scientific claims, and BM25/dense retrieval
are already strong on this vocabulary; injected synonyms likely add
lexical noise that dilutes specificity more than it closes a
vocabulary-mismatch gap. This is the expected outcome on a task where the
base retrievers are already well-matched to the corpus's language — the
technique is still worth having (a different, jargon-mismatched corpus is
where it should help), but it isn't a blind win, and the eval caught that
instead of a hand-picked example papering over it.

**Not yet built:**

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

`.env` is gitignored — never commit it. `MISTRAL_API_KEY` is required for
`evaluate --expand-query` (see Phase 4 below); every other command works
without it.

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

# Train the learned reranker on a dataset's train split
uv run natsuki train-reranker --train-dataset beir/scifact/train \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz \
  --out models/reranker.txt

# Evaluate the reranker on the (held-out) test split
uv run natsuki evaluate --dataset beir/scifact/test --mode rerank \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz \
  --reranker models/reranker.txt --k 10

# Same, with LLM query expansion (Mistral API), capped to 100 queries
uv run natsuki evaluate --dataset beir/scifact/test --mode rerank \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz \
  --reranker models/reranker.txt --k 10 --expand-query --limit 100

# Ad-hoc single BM25 query
uv run natsuki search --index indexes/scifact.index.gz --query "your query here" --k 10
```

## Tests

```bash
uv run pytest -q
```

35 unit tests covering tokenizer edge cases, BM25 ranking behavior (term
frequency ordering, no-match handling, unfinalized-index guard), dense
index search/persistence, reciprocal rank fusion, candidate feature
extraction, the LambdaMART reranker (including a synthetic-data test that
checks it actually learns to weight a real signal over pure noise), query
expansion (via an injected fake chat function, including its
fail-open-to-original-query behavior), and eval metrics against
hand-computed NDCG/MRR/Recall values. Dense-index, reranker, and query-
expansion tests use hand-crafted vectors/features or fake LLM responses
rather than the real embedding/chat models, so the suite stays fast,
free, and offline — embedding correctness itself was checked manually
(see `natsuki/embeddings.py`
docstring) and validated indirectly by the SciFact NDCG numbers above
matching the published baseline.

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
