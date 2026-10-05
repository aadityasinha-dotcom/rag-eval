# Trade-offs

## Chunking: fixed512 and chunk identity

**Chunks never span documents.** A strategy is a pure function of one
document's normalized text. The `c<n>` counter restarts at 0 per document, and
the last chunk is whatever remains (possibly shorter than 512 chars). Adding,
removing or editing other files cannot change a document's chunks; this is
asserted in `tests/test_chunk.py`.

**`<doc>` is the file path relative to `corpus/`, extension stripped**, e.g.
`corpus/pg16/ddl.html` -> `pg16/ddl#c0`. Renaming or moving a file re-ids every
chunk in it and is a re-label event.

**Ids are positional, not content hashes.** `c<n>` is "the window starting at
offset 448 * n". Editing a document leaves chunks before the edit identical,
keeps the ids of chunks after it, but may change their text and the number of
trailing chunks. Golden entries citing those ids must be re-labelled (rule 7).
Alternatives considered:

- *Content-hash ids* survive edits elsewhere in the doc, but are unreadable
  during hand-labelling and still break whenever the cited chunk itself
  changes. Rejected.
- *Section-anchored ids* (`<doc>#<heading>#c<n>`) are more edit-tolerant but
  need a structure-aware parser; that is the semantic-chunking variant, a later
  config, not the baseline.

Each chunk carries `text_hash` (sha256 of its text) so drift can be detected
later: when indexing lands, stored hashes let a check compare the golden set's
chunks with the current corpus instead of silently scoring against moved text.

**Why a plain Python window rather than a LlamaIndex splitter.** LlamaIndex is
allowed for node parsing, but its splitters are sentence/token aware and their
output depends on tokenizer and library version. The baseline is meant to be
the dumbest reproducible thing; a 10-line character window keeps determinism
entirely in this repo. Smarter splitting arrives as its own strategy and row.

**One file = one document.** LlamaIndex's MarkdownReader splits a file into
one Document per header and PDF readers split per page. `ingest/parse.py`
re-joins those per file so `<doc>` always means a file.

## Corpus: PostgreSQL 16 HTML docs

**Source.** There is no separate docs tarball for 16.x; the release source
tarball ships prebuilt HTML under `doc/src/sgml/html/`. `make fetch` downloads
`postgresql-16.15.tar.gz`, verifies the published sha256, and extracts only
that directory, flat into `corpus/`, so `<doc>` is the page name
(`sql-createtable`, `ddl-basics`). Pinning 16.15 matters: a point release can
reword pages and would shift chunk ids.

**Parsing.** The pages are DocBook output: `<div class="sect1">`, no
`<section>` tags, so LlamaIndex's HTMLTagReader would return no text.
`ingest/parse.py` plugs a small BeautifulSoup reader into SimpleDirectoryReader
that drops the Prev/Up/Next navigation blocks and heading permalink glyphs,
starts a new line at block elements only (so `<code>` and `<em>` stay inside
their sentence), and collapses blank-line runs. Indentation inside `<pre>` is
lost; accepted for the baseline.

**Scope notes for the golden set.** `bookindex` (the alphabetical index) and
the 17 `release-*` pages are chunked like everything else. They are poor
retrieval targets and good distractors; exclude them from the corpus later only
as a measured config change, not silently.

## Index schema (step 4)

Two tables, both keyed by strategy name so several chunkings coexist and a
config picks one: `chunks(strategy, chunk_id, ...)` with a generated
`tsvector` column for sparse retrieval, and `embeddings(strategy, chunk_id,
model, ...)` with an untyped `vector` column so models of different dimension
share one table (queries always filter by model). No ANN index for the
baseline: an exact scan over ~16k rows costs a few milliseconds and keeps
recall numbers free of HNSW approximation; adding one is a config change to
measure later.

Indexing is incremental. Chunk rows are rewritten only when `text_hash`
changes, stale ids are deleted (embeddings cascade), and a vector is computed
only when its chunk has none for that model or the hash moved. Vectors are
also cached on disk by sha256(text), so a chunking change re-embeds only
genuinely new text and an unchanged corpus makes zero API calls.

`hash64` is a deterministic bag-of-words embedder for tests and the
`smoke-hash64` config. It exercises the pipeline end to end without a key or
network; its retrieval numbers are meaningless and must never land in
results.md as if they were a variant.
