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
- `src/natsuki/cli.py` — `natsuki build-index`, `natsuki evaluate`, `natsuki search`.

**Validated on BEIR/SciFact** (5,183 docs, 300 test queries, real qrels):

```text
ndcg@10: 0.6692
mrr:     0.6432
recall@10: 0.7856
0.74 ms/query avg
```

The published BEIR paper reports a BM25 baseline of **NDCG@10 ≈ 0.665** on
SciFact — this from-scratch implementation lands at 0.6692, matching the
reference almost exactly. That's the correctness check for the whole
tokenizer -> index -> BM25 pipeline before anything else gets layered on.

**Not yet built:**

- Dense retrieval (embeddings via a local CPU model for bulk documents;
  Mistral API for query-time work) + reciprocal rank fusion.
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

# Evaluate against that dataset's real relevance judgments
uv run natsuki evaluate --dataset beir/scifact/test --index indexes/scifact.index.gz --k 10

# Ad-hoc single query
uv run natsuki search --index indexes/scifact.index.gz --query "your query here" --k 10
```

## Tests

```bash
uv run pytest -q
```

13 unit tests covering tokenizer edge cases, BM25 ranking behavior (term
frequency ordering, no-match handling, unfinalized-index guard), and eval
metrics against hand-computed NDCG/MRR/Recall values.

## Hardware notes (why the architecture looks like this)

Built on a laptop with no discrete GPU (Intel i7-12650H, 14GB RAM,
integrated graphics only) — no local model fine-tuning or high-throughput
inference. The plan going forward:

- **Document embeddings** (bulk, one-time cost, scales with corpus size):
  a small CPU-friendly open embedding model (e.g. `all-MiniLM-L6-v2` via
  ONNX Runtime), *not* the Mistral API — embedding hundreds of thousands
  of documents through a paid API is slow and expensive; this is a local,
  free, one-time indexing job.
- **Query-time LLM work** (query understanding, reranking prompts,
  LLM-as-judge eval): Mistral API — call volume is per-query, not
  per-document, so it's cheap and doesn't need local compute.
- **Learned reranker**: LambdaMART (LightGBM) trained on CPU — no GPU
  dependency, trains in minutes even on a few hundred thousand rows.

Target corpus size for the full hybrid pipeline: ~500k-1M documents (fits
comfortably in RAM for embeddings + HNSW index); a 2-3M-document run is a
stretch goal for a corpus-size-vs-latency scaling chart, run as a
dedicated batch job rather than interactively.
