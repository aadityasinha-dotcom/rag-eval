# rag-eval

Document Q&A over the PostgreSQL 16 documentation where retrieval quality is a
measured engineering metric, not a demo. The deliverable is an ablation table
(`docs/results.md`) showing how each retrieval change moved recall,
faithfulness, latency and cost against a hand-labelled golden set.

**Status:** baseline phase. Corpus, chunking, indexing, dense retrieval, the
metrics and the eval harness exist. The golden set is being written by hand;
the first results row (the baseline) lands once it has 50+ entries.

## How it works

```
corpus/*.html  ──parse──▶  ParsedDoc  ──chunk──▶  Chunk (id = <doc>#c<n>)
                                                     │
                                       embed + index ▼
                           Postgres: chunks (text, tsvector), embeddings (pgvector)
                                                     │
          evals/golden.jsonl ──▶ evals/run.py ──retrieve──▶ top-k ──generate──▶ answer
                                      │
                                      ▼
                       docs/results.md (one row per config run)
```

- **Corpus:** PostgreSQL 16.15 docs, 1169 prebuilt HTML pages from the release
  source tarball, fetched by script (not checked in).
- **Chunking:** `ingest/chunk.py` holds `STRATEGIES`. The only one so far is
  `fixed512` (512 chars, 64 overlap). Ids are deterministic and positional,
  e.g. `sql-createtable#c3`, and never cross a document boundary.
- **Index:** `ingest/index.py` upserts chunks and embeddings into Postgres
  incrementally; unchanged text is never re-embedded. Embedding model is a
  config field (`text-embedding-3-small` baseline).
- **Retrieval:** `retrieval/dense.py` is one SQL statement: exact cosine
  top-k via pgvector. Sparse, hybrid and reranking come later as configs.
- **Generation:** `generation/answer.py` stuffs the top-k chunks into a
  versioned prompt (`generation/prompts.py`) and asks Claude for a cited
  answer, abstaining when the context does not contain one.
- **Eval:** `evals/run.py` runs a config over the golden set, scores
  recall@k, MRR and nDCG@k, measures retrieval latency, and appends a row to
  `docs/results.md` plus a per-query JSON under `evals/runs/`. With
  `FAITHFULNESS=1` each real answer is scored by Ragas' Faithfulness using
  the config's judge model.

Design rules, build order and the reasoning behind each choice live in
[CLAUDE.md](CLAUDE.md) and [docs/tradeoffs.md](docs/tradeoffs.md).

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, and API keys
for embedding (OpenAI) and, optionally, answer generation (Anthropic).

```bash
uv sync --all-groups          # venv + dependencies
cp .env.example .env          # add OPENAI_API_KEY (and ANTHROPIC_API_KEY for --generate)
make fetch                    # download the corpus into corpus/ (sha256-verified)
make dev                      # postgres 16 + pgvector in docker
make test                     # 76 tests; the Postgres ones skip if the db is down
make index CONFIG=evals/configs/baseline.yaml
make eval  CONFIG=evals/configs/baseline.yaml            # retrieval metrics -> results.md
make eval  CONFIG=evals/configs/baseline.yaml GENERATE=1 # also answers each question
make eval  CONFIG=evals/configs/baseline.yaml FAITHFULNESS=1 # ...and judges them with Ragas
```

`evals/configs/smoke-hash64.yaml` runs the whole pipeline offline with a
deterministic hash embedder and no keys. Its numbers are meaningless and are
not a result.

## Writing the golden set

`evals/golden.jsonl` is written by hand, one JSON object per line. Never
generate questions with an LLM (CLAUDE.md rule 2).

```jsonl
{"id": "q001", "question": "...", "answer": "...", "relevant_chunk_ids": ["sql-lock#c0"], "difficulty": "single-hop"}
```

`difficulty` is `single-hop`, `multi-hop` or `unanswerable` (empty id list,
answer states the refusal). Tools for labelling:

```bash
make grep Q="NOWAIT is specified"       # chunk ids whose text matches, with a snippet
make chunks DOC=sql-lock                # one page's chunks with ids
make golden-check                       # shape, ids exist, cited text unchanged since freeze
make golden-freeze                      # after a batch: record cited chunks' text hashes
uv run python -m evals.golden --show q007   # question, answer and labelled chunk text
uv run python -m evals.golden --stats
```

Changing a chunking strategy, or an edited corpus, changes which text an id
points to. `golden-check` catches this against the hashes recorded by
`golden-freeze`; affected entries are re-labelled, then frozen again with
`FORCE=1` (rule 7).

## Adding a variant

1. Copy `evals/configs/baseline.yaml`, change exactly one thing
   (`chunking`, `embedding`, `retrieval`, `rerank`, `top_k`, `generation`,
   `prompt`), give it a `name`.
2. `make index CONFIG=...` if chunking or embedding changed.
3. `make eval CONFIG=...` appends the row. `docs/results.md` is never edited
   by hand; a config that has not been run has no row.

## Layout

```
corpus/            fetched source pages (gitignored)
scripts/           fetch_corpus.py + sources.txt manifest
ingest/            parse.py, chunk.py (STRATEGIES), embed.py (EMBEDDERS), index.py, db.py
retrieval/         base.py (Hit, Retriever), dense.py, factory.py
generation/        prompts.py (versioned), answer.py
evals/
  golden.jsonl     the hand-labelled set
  golden.py        validator / viewer
  config.py        RunConfig from evals/configs/*.yaml
  metrics/         retrieval.py (recall@k, MRR, nDCG@k), generation.py (Ragas faithfulness)
  run.py           the harness
  runs/            per-run JSON detail
db/                init.sql (extension), schema.sql (tables)
docs/              results.md (machine-appended), tradeoffs.md
tests/             pytest; fixtures/corpus is a 5-file mini corpus
```

## Development

```bash
make lint          # ruff check, ruff format --check, mypy --strict
make test
make search Q="lock a table without waiting"   # dense top-k sanity check, not a metric
```

Parsed corpus and embeddings are cached under `.cache/` (gitignored); the
parse cache invalidates itself when a corpus file or the parser changes.
