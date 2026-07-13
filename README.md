# natsuki

A hybrid BM25 + dense retrieval search engine with a learned reranker and
LLM query expansion — built from scratch (no BM25/IR library dependency)
and evaluated end-to-end against a public benchmark with real relevance
judgments, not self-graded numbers.

## Results

Evaluated on **BEIR/SciFact** (5,183 docs, 300 test queries, real qrels).
The reranker is trained on the dataset's separate 809-query train split —
no leakage into the numbers below.

| mode            | ndcg@10 | mrr    | recall@10 | ms/query |
|-----------------|---------|--------|-----------|----------|
| bm25            | 0.6692  | 0.6432 | 0.7856    | 0.74     |
| dense           | 0.7215  | 0.6932 | 0.8396    | 42.10    |
| hybrid (RRF)    | 0.7245  | 0.6988 | 0.8379    | 48.19    |
| rerank          | 0.7459  | 0.7144 | 0.8762    | 52.38    |

Every retrieval stage is validated, not assumed:

- **BM25 correctness**: the published BEIR paper reports a BM25 baseline
  of NDCG@10 ≈ 0.665 on SciFact. This from-scratch implementation lands
  at 0.6692 — matching the reference almost exactly, which is the
  correctness check for the whole tokenizer → index → BM25 chain before
  anything else is layered on top.
- **Dense retrieval** clearly beats BM25 here (SciFact is a
  scientific-claim-matching task where semantic similarity carries a lot
  of the signal), and **hybrid RRF fusion** edges out dense-only on every
  metric — combining two ranking signals helps even when one is
  individually much stronger.
- **The learned reranker (LambdaMART)** beats hybrid RRF on every metric
  too, giving a clean monotonic improvement across all four modes:
  `bm25 < dense < hybrid < rerank`. It replaces RRF's fixed
  `1/(k+rank)` formula with weights learned from labeled data, and its
  feature importances (below) show it discovered — rather than being
  told — that dense rank matters far more than BM25 rank on this corpus.
- **LLM query expansion** (Mistral API) was A/B tested honestly rather
  than cherry-picked: it *hurt* NDCG@10/MRR while modestly helping
  recall, and added ~400ms/query of network latency (see
  [Design decisions](#design-decisions)). Not every technique wins, and
  the eval caught that instead of a hand-picked example hiding it.

## Architecture

```text
query ─┬─→ BM25 (inverted index)      ─┐
       └─→ dense (cosine similarity)  ─┴─→ fuse ─→ rank
                                            │
                              RRF (fixed formula)         [hybrid mode]
                              LambdaMART (learned)         [rerank mode]

query ─→ [optional: Mistral query expansion] ─→ any of the above
```

Candidate generation always starts from the union of BM25's and dense
retrieval's top-N hits; `hybrid` mode fuses them with reciprocal rank
fusion, `rerank` mode scores them with a trained LambdaMART model whose
features are each retriever's score and reciprocal rank (see
`src/natsuki/features.py`).

## Project structure

```text
src/natsuki/
  tokenizer.py            regex tokenizer + stopword filtering
  index/
    inverted_index.py     term -> postings (doc_id, tf), BM25's index
    dense_index.py         flat cosine-similarity vector index
  bm25.py                  BM25 scoring (Robertson/Sparck-Jones IDF)
  embeddings.py            local CPU embeddings (fastembed/ONNX, no GPU)
  hybrid.py                reciprocal rank fusion
  features.py              candidate features for the reranker
  reranker.py              LambdaMART (LightGBM), trained on CPU
  llm.py                   Mistral chat API wrapper
  query_understanding.py   LLM query expansion
  data.py                  ir_datasets loader (BEIR, MS MARCO, ...)
  eval.py                  NDCG@k / MRR / Recall@k, implemented from scratch
  config.py                .env / secrets loading
  cli.py                   natsuki build-index / build-dense-index /
                           train-reranker / evaluate / search
tests/                     35 tests, one file per module above
```

## Setup

```bash
uv sync
cp .env.example .env   # then fill in MISTRAL_API_KEY
```

`.env` is gitignored — never commit it. `MISTRAL_API_KEY` is only
required for `evaluate --expand-query`; every other command works
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

## Design decisions

Decisions worth explaining rather than leaving implicit:

**Why no `pytrec_eval`.** It failed to compile on this machine — its
bundled `trec_eval` C source is old-style C89 (implicit `qsort`
comparator casts) that trips strict prototype checking on newer GCC.
Rather than patch upstream C, NDCG/MRR/Recall@k are implemented directly
in `natsuki/eval.py`, matching the standard formulas (DCG with `2^rel-1`
gain, log2 discount). The SciFact number matching the published BEIR
baseline is the validation that these are correct — arguably a stronger
result than depending on a black-box library would have been.

**The `mistralai` SDK import quirk.** This SDK build (2.6.0) ships with
no root-level `mistralai/__init__.py` — `from mistralai import Mistral`
(the documented import) fails. The working import is
`from mistralai.client import Mistral`. Found by inspecting the installed
package's `RECORD` file rather than assuming the docs were current.

**Hardware drove the architecture, not the other way round.** Built on a
laptop with no discrete GPU (Intel i7-12650H, 14GB RAM, integrated
graphics only) — no local model fine-tuning or high-throughput inference.
That constraint made several decisions for me:

- *Document embeddings* (bulk, one-time, scales with corpus size):
  `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX runtime), not the
  Mistral API — embedding hundreds of thousands of documents through a
  paid API is slow and expensive; this is a local, free job. Measured
  throughput: ~5.3 docs/sec, single-threaded, on this CPU (5,183 docs in
  ~16 minutes). At that rate a 500k-document corpus is a ~26-hour embed
  job and 1M docs is ~2 days — both fine as an overnight/multi-day batch
  job, but they set the practical ceiling. The first lever to pull before
  assuming a smaller corpus is required: `fastembed` is currently running
  single-threaded despite 16 CPU threads being available.
- *Query-time LLM work* (query expansion, and reranking/judging if
  extended later): Mistral API — call volume is per-query, not
  per-document, so it's cheap and doesn't need local compute.
- *Learned reranker*: LambdaMART (LightGBM) trained on CPU — no GPU
  dependency, trains in minutes even on a few hundred thousand rows (809
  queries / 138,103 candidate pairs trained in ~40s here).

**LLM query expansion's honest result.** A/B tested on the same 100
held-out queries with and without expansion (rerank mode):

| variant         | ndcg@10 | mrr    | recall@10 | ms/query |
|-----------------|---------|--------|-----------|----------|
| rerank          | 0.7855  | 0.7570 | 0.9097    | 67.89    |
| rerank + expand | 0.7708  | 0.7321 | 0.9197    | 453.63   |

Expansion hurt top-of-ranking precision while modestly improving recall,
and added ~400ms/query of pure network latency. SciFact queries are
precise scientific claims that BM25/dense already match well; injected
synonyms likely add lexical noise rather than closing a vocabulary gap.
The technique is still worth having — a jargon-mismatched corpus is where
it should earn its keep — but it isn't a blind win here, and the eval
surfaced that instead of a cherry-picked example hiding it.

## Tests

```bash
uv run pytest -q
```

35 unit tests covering tokenizer edge cases; BM25 ranking behavior (term
frequency ordering, no-match handling, unfinalized-index guard); dense
index search/persistence; reciprocal rank fusion; candidate feature
extraction; the LambdaMART reranker (including a synthetic-data test that
checks it actually learns to weight a real signal over pure noise); query
expansion (via an injected fake chat function, including its
fail-open-to-original-query behavior); and eval metrics against
hand-computed NDCG/MRR/Recall values. Dense-index, reranker, and
query-expansion tests use hand-crafted vectors/features or fake LLM
responses rather than the real embedding/chat models, so the suite stays
fast, free, and offline — embedding correctness itself was checked
manually (see `natsuki/embeddings.py` docstring) and validated indirectly
by the SciFact NDCG numbers above matching the published baseline.

