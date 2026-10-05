# rag-eval

A document Q&A system where retrieval quality is treated as a measured
engineering metric, not a demo. The deliverable is an ablation table showing
how each retrieval change moved recall, faithfulness, latency, and cost.

Corpus: PostgreSQL 16.15 documentation, prebuilt HTML from the release source
tarball (`doc/src/sgml/html/`), 1169 pages, one file per page. Fetched by
`make fetch` per `scripts/sources.txt`; chunk ids look like `sql-createtable#c3`.

Related repos: `llmobserve-python` (SDK used to instrument queries) and
`llm-observe` (the platform that receives those traces). This repo must run
fully without either of them present.

## Current phase

Golden set + baseline. In scope: the hand-labeled eval set, the dumbest
working retrieval pipeline, and the harness that scores it. Not in scope yet:
hybrid search, reranking, the LangGraph corrective loop, or the API.

## Non-negotiable design rules

1. **No retrieval code before the golden set exists.** Everything is measured
   against `evals/golden.jsonl`. Without it there is nothing to optimise.
2. **Golden set is written by hand.** Never generate questions with an LLM and
   grade them with an LLM — that measures a circle. Each entry has a question,
   a reference answer, and the ids of chunks that should be retrieved.
3. **Baseline before cleverness.** Fixed-size chunks, dense-only retrieval,
   k=5, stuff into prompt. Measure it. Then change one variable at a time.
4. **Every variant is a config file, not a branch.** `evals/configs/*.yaml`
   selects chunking, retrieval, reranking, top_k. `evals/run.py <config>` runs
   the full golden set against it and appends a row to `docs/results.md`.
5. **Never invent numbers.** `docs/results.md` contains only rows produced by
   `evals/run.py`. If a config hasn't been run, it has no row.
6. **Own the retrieval.** LlamaIndex is used for document loading and node
   parsing only. Dense, sparse, fusion, and reranking are written here, in
   plain Python against Postgres. No `RetrievalQA`, no `query_engine` black
   boxes — the point is being able to explain every step.
7. **Chunk ids are stable.** The golden set references chunk ids, so a
   chunking strategy must produce deterministic ids for the same input.
   Changing chunking means re-labelling the affected golden entries — that is
   expected and must be done, not worked around.
8. **Instrumentation is optional.** `llmobserve` is configured from env and
   the system runs identically with it unset.

## Stack

- Python 3.11+, uv or pip
- Postgres 16 + pgvector (vectors) + tsvector (BM25-style sparse)
- LlamaIndex for loaders and node parsers only
- `text-embedding-3-small` as the baseline embedding (swappable via config)
- `bge-reranker-v2-m3` cross-encoder (later phase)
- LangGraph for the corrective retrieval loop (later phase)
- Ragas for faithfulness and answer relevance; retrieval metrics written here
- FastAPI for `/query` (later phase)
- `llmobserve` for per-query latency and cost

## Layout

```
corpus/            raw source documents, checked in or fetched by script
ingest/            parse → chunk → embed → index (chunk.py holds STRATEGIES)
retrieval/         dense.py, sparse.py, hybrid.py, rerank.py
generation/        prompts.py (versioned), answer.py (cites chunk ids)
graph/             LangGraph corrective loop (later phase)
evals/
  golden.jsonl     the hand-labelled set
  metrics/         retrieval.py (recall@k, MRR, nDCG), generation.py (Ragas)
  configs/         one yaml per variant
  run.py           runs a config over golden set, appends to results.md
docs/
  results.md       THE ablation table, machine-appended only
  tradeoffs.md     pgvector vs Qdrant, why own retrieval, chunking choices
```

## Golden set format

```jsonl
{"id": "q001", "question": "...", "answer": "...", "relevant_chunk_ids": ["doc3#c12", "doc3#c13"], "difficulty": "single-hop"}
```

Aim for 50–100 entries. Mix single-hop, multi-hop, and a few that the corpus
cannot answer (to measure refusal / hallucination).

## Metrics

- recall@k — fraction of relevant_chunk_ids present in the top k
- MRR — 1 / rank of the first relevant chunk
- nDCG@k — rank-weighted relevance
- faithfulness — Ragas, is the answer supported by the retrieved context
- latency p50/p95 and $/query — from llmobserve traces

## Commands

```bash
make dev            # postgres + pgvector via docker compose
make fetch          # download corpus per scripts/sources.txt
make chunks DOC=    # print a doc's chunks with ids (for writing golden.jsonl)
make grep Q=        # chunk ids whose text matches a phrase (for finding ids)
make golden-check   # validate golden.jsonl: shape + every chunk id exists
make index CONFIG=  # ingest corpus with a chunking strategy
make search Q=      # dense top-k for one query (sanity check, not a metric)
make eval CONFIG=   # run golden set, append row to results.md
make lint           # ruff + mypy
make test
```

## Build order (current phase)

1. Pick and fetch the corpus into `corpus/`. Decide this first.
2. `ingest/parse.py` + `ingest/chunk.py` with ONE strategy: fixed 512 chars,
   deterministic ids of the form `<doc>#c<n>`
3. Write `evals/golden.jsonl` by hand against those chunks. 50 minimum.
4. `ingest/embed.py` + `ingest/index.py` — chunks, embeddings, tsvector into
   Postgres
5. `retrieval/dense.py` — cosine similarity, top k
6. `evals/metrics/retrieval.py` — recall@k, MRR, nDCG with unit tests against
   hand-computed examples
7. `evals/run.py` — runs a config, prints metrics, appends to results.md
8. First row in `docs/results.md`: the baseline

Step 8 is the milestone. Only after it exists: hybrid → rerank → semantic
chunking → LangGraph loop, each as its own config and its own row.

## Out of scope for now

Hybrid search, reranking, the corrective loop, the API, any UI, prompt
versioning, alternative embedding models.
