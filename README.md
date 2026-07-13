# natsuki

Hybrid BM25 + dense retrieval search engine with a learned reranker and
LLM query expansion, built from scratch with no IR library dependency.
Evaluated on BEIR/SciFact (5,183 docs, 300 test queries).

## Results

| mode         | ndcg@10 | mrr    | recall@10 | ms/query |
|--------------|---------|--------|-----------|----------|
| bm25         | 0.6692  | 0.6432 | 0.7856    | 0.74     |
| dense        | 0.7215  | 0.6932 | 0.8396    | 42.10    |
| hybrid (RRF) | 0.7245  | 0.6988 | 0.8379    | 48.19    |
| rerank       | 0.7459  | 0.7144 | 0.8762    | 52.38    |

The published BEIR paper reports BM25 at NDCG@10 ≈ 0.665 on SciFact; this
implementation gets 0.6692, so the tokenizer/index/scoring chain is
scoring correctly.

Dense retrieval beats BM25 by a wide margin here — SciFact is a
scientific-claim-matching task where semantic similarity does most of the
work. RRF fusion of BM25+dense edges out dense-only on every metric, and
the LambdaMART reranker (trained on the dataset's separate 809-query
train split) beats RRF on every metric too — a clean progression from
bm25 → dense → hybrid → rerank. Its learned feature importances lean
heavily on dense rank over BM25 rank, which tracks the standalone
numbers above.

LLM query expansion (Mistral API) hurt NDCG@10/MRR while modestly
improving recall, and added ~400ms/query of network latency (numbers and
reasoning below). SciFact queries are precise scientific claims that
BM25/dense already handle well, so the injected synonyms mostly add
noise rather than closing a vocabulary gap.

## Nearest-neighbor search: KD-tree vs LSH

Two more retrieval backends, both built from scratch (no FAISS/hnswlib)
and benchmarked against the flat brute-force dense index above, on the
same 5,183-doc corpus:

| index        | ndcg@10 | mrr    | recall@10 | ms/query | candidates touched |
|--------------|---------|--------|-----------|----------|--------------------|
| dense (flat) | 0.7215  | 0.6932 | 0.8396    | 42.10    | 100% (5183/5183)   |
| kdtree       | 0.7215  | 0.6932 | 0.8396    | 51.00    | 100% (5183/5183)   |
| lsh (8t/8b)  | 0.6355  | 0.6237 | 0.7176    | 55.12    | 44.9% (2327/5183)  |
| lsh (16t/6b) | 0.7168  | 0.6895 | 0.8339    | 59.45    | 93.1% (4823/5183)  |

**KD-tree** does exact search with branch-and-bound pruning. At 384
dimensions it visits all 5,183 nodes per query, every time — the tree is
only ~13 levels deep for this corpus, so almost none of the 384
dimensions ever get split on, and pruning has nothing to work with. Its
NDCG/MRR/Recall match flat dense exactly (it's exact search, not
approximate), but at 51ms/query it's slower than the 42ms flat numpy
matmul. There's no upside here — that's the point of building it: KD-trees
work well for low-dimensional spatial data, not high-dimensional
embeddings, and this measures that directly instead of citing it.

**LSH** (random hyperplane hashing, tunable via table/bit count) does
show a real reduction in work: with 8 tables and 8 bits per table, each
query only gets compared against 44.9% of the corpus, at a real recall
cost. With 16 tables and 6 bits, it recovers most of that quality (NDCG@10
0.7168 vs flat's 0.7215) while still only touching 93.1% of the corpus —
a genuine, tunable recall/candidate-fraction tradeoff.

Neither structure actually wins on wall-clock time here, and that's an
honest result, not a bug: 5,183 docs is small enough that one vectorized
numpy matmul is already close to the floor, while pure-Python hashing and
tree traversal both carry their own per-query overhead. The
candidates-touched number is what would turn into a real wall-clock win
at a larger corpus size or in a compiled implementation — which is also
why production systems reach for graph-based structures (HNSW) or
vectorized libraries (FAISS) once the corpus is big enough for it to
matter, rather than either of these.

## Generalization beyond SciFact

<!-- TODO: results for NFCorpus and ArguAna go here -->

## Evidence extraction

Retrieval returns whole documents. `natsuki explain` goes one step
further: it retrieves the top-k documents for a query, then splits each
into sentences and returns the one(s) most relevant to the query by
embedding similarity, instead of the full document text
(`src/natsuki/evidence.py`). Example, on SciFact:

```text
$ natsuki explain --dataset beir/scifact/test --dense-index indexes/scifact.dense.npz \
    --index indexes/scifact.index.gz --query "0-dimensional biomaterials show inductive properties." --k 3

1. 43385013
   [0.703] We used nontumorigenic basal cell lines as models of normal stem
   cells/progenitors and demonstrate that these cell lines contain an
   epithelial subpopulation ...
2. 40212412
   [0.736] This need is met by the lever function of long bones,
   three-dimensional masterpieces of biomechanical engineering ...
```

This is a heuristic, not a benchmarked system: BEIR's version of SciFact
strips the original dataset's sentence-level evidence annotations (each
claim's real SUPPORT/CONTRADICT rationale sentences), so there's no
ground truth here to measure extraction accuracy against — the sentence
ranking is just cosine similarity between the query and each sentence,
using the same embedding model as dense retrieval. The sentence splitter
is a plain regex, not a real tokenizer, so it will mishandle common
scientific-text abbreviations ("Fig. 1", "e.g.").

## Architecture

```text
query ─┬─→ BM25 (inverted index)      ─┐
       └─→ dense (cosine similarity)  ─┴─→ fuse ─→ rank
                                            │
                              RRF (fixed formula)         [hybrid mode]
                              LambdaMART (learned)         [rerank mode]

query ─→ [optional: Mistral query expansion] ─→ any of the above
```

Candidate generation is always the union of BM25's and dense retrieval's
top-N hits. `hybrid` mode fuses them with reciprocal rank fusion;
`rerank` mode scores them with a trained LambdaMART model whose features
are each retriever's score and reciprocal rank (`src/natsuki/features.py`).

## Project structure

```text
src/natsuki/
  tokenizer.py            regex tokenizer + stopword filtering
  index/
    inverted_index.py     term -> postings (doc_id, tf), BM25's index
    dense_index.py         flat cosine-similarity vector index
    kdtree_index.py         from-scratch KD-tree ANN
    lsh_index.py             from-scratch LSH (random hyperplane) ANN
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

`.env` is gitignored. `MISTRAL_API_KEY` is only needed for
`evaluate --expand-query`; everything else works without it.

## Usage

```bash
# Build a BM25 index from any ir_datasets id
uv run natsuki build-index --dataset beir/scifact/test --out indexes/scifact.index.gz

# Build a local dense (embedding) index for the same corpus
uv run natsuki build-dense-index --dataset beir/scifact/test --out indexes/scifact.dense.npz

# Evaluate against that dataset's relevance judgments
uv run natsuki evaluate --dataset beir/scifact/test --mode bm25 --index indexes/scifact.index.gz --k 10
uv run natsuki evaluate --dataset beir/scifact/test --mode dense --dense-index indexes/scifact.dense.npz --k 10
uv run natsuki evaluate --dataset beir/scifact/test --mode hybrid \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz --k 10

# Train the learned reranker on a dataset's train split
uv run natsuki train-reranker --train-dataset beir/scifact/train \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz \
  --out models/reranker.txt

# Evaluate the reranker on the held-out test split
uv run natsuki evaluate --dataset beir/scifact/test --mode rerank \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz \
  --reranker models/reranker.txt --k 10

# Same, with LLM query expansion (Mistral API), capped to 100 queries
uv run natsuki evaluate --dataset beir/scifact/test --mode rerank \
  --index indexes/scifact.index.gz --dense-index indexes/scifact.dense.npz \
  --reranker models/reranker.txt --k 10 --expand-query --limit 100

# Build KD-tree / LSH indexes on top of an existing dense index's vectors
uv run natsuki build-kdtree-index --dense-index indexes/scifact.dense.npz --out indexes/scifact.kdtree.pkl.gz
uv run natsuki build-lsh-index --dense-index indexes/scifact.dense.npz --out indexes/scifact.lsh.pkl.gz \
  --num-tables 16 --num-bits 6

uv run natsuki evaluate --dataset beir/scifact/test --mode kdtree --kdtree-index indexes/scifact.kdtree.pkl.gz --k 10
uv run natsuki evaluate --dataset beir/scifact/test --mode lsh --lsh-index indexes/scifact.lsh.pkl.gz --k 10

# Ad-hoc single BM25 query
uv run natsuki search --index indexes/scifact.index.gz --query "your query here" --k 10
```

## Design decisions

**No `pytrec_eval`.** It wouldn't compile here — its bundled `trec_eval`
C source is old-style C89 (implicit `qsort` comparator casts) that trips
strict prototype checking on newer GCC. NDCG/MRR/Recall@k are implemented
directly in `natsuki/eval.py` instead, matching the standard formulas
(DCG with `2^rel-1` gain, log2 discount).

**`mistralai` SDK import.** This SDK build (2.6.0) ships with no
root-level `mistralai/__init__.py`, so the documented
`from mistralai import Mistral` fails. The working import is
`from mistralai.client import Mistral`.

**No GPU on this machine** (Intel i7-12650H, 14GB RAM, integrated
graphics only), which shaped a few choices:

- *Document embeddings*: `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX
  runtime), not the Mistral API — embedding hundreds of thousands of
  documents through a paid API is slow and expensive, and this is a
  one-time local job instead. Measured throughput: ~5.3 docs/sec,
  single-threaded, on this CPU (5,183 docs in ~16 minutes). At that rate
  a 500k-doc corpus is a ~26-hour embed job and 1M docs is ~2 days —
  fine as an overnight batch job, but the ceiling for now. `fastembed` is
  currently running single-threaded despite 16 CPU threads being
  available, which is the first thing to fix before assuming a smaller
  corpus is needed.
- *Query-time LLM work* (query expansion): Mistral API — call volume is
  per-query, not per-document, so it's cheap.
- *Learned reranker*: LambdaMART (LightGBM) trains fine on CPU — 809
  queries / 138,103 candidate pairs trained in ~40s.

**LLM query expansion result.** A/B tested on the same 100 held-out
queries, with and without expansion (rerank mode):

| variant         | ndcg@10 | mrr    | recall@10 | ms/query |
|-----------------|---------|--------|-----------|----------|
| rerank          | 0.7855  | 0.7570 | 0.9097    | 67.89    |
| rerank + expand | 0.7708  | 0.7321 | 0.9197    | 453.63   |

Expansion hurt top-of-ranking precision while modestly improving recall,
and added ~400ms/query of network latency. SciFact queries are precise
scientific claims that BM25/dense already match well, so injected
synonyms are more likely to add noise than to close a vocabulary gap.
Probably behaves differently on a corpus with more jargon mismatch
between how users phrase queries and how documents are written.

## Tests

```bash
uv run pytest -q
```

47 tests covering tokenizer edge cases, BM25 ranking behavior (term
frequency ordering, no-match handling, unfinalized-index guard), dense
index search/persistence, reciprocal rank fusion, candidate feature
extraction, the LambdaMART reranker (including a synthetic-data test that
checks it learns to weight a real signal over pure noise), query
expansion (via an injected fake chat function, including the
fail-open-to-original-query path), the KD-tree (including a
high-dimensional test asserting it visits >80% of nodes, matching the
result above) and LSH (recall-vs-brute-force and candidate-set-size
tests) indexes, and the eval metrics themselves against hand-computed
NDCG/MRR/Recall values. Index and query-expansion tests use hand-crafted
vectors/features or fake LLM responses instead of the real embedding/chat
models, so the suite stays fast and doesn't need network access or an
API key.
