-- Applied by ingest.db.ensure_schema on every index run; must stay idempotent.
-- Tables are created in the first schema on search_path, so tests can run in
-- a throwaway schema.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk per chunking strategy. tsv is the sparse (BM25-style)
-- representation used by retrieval/sparse.py later; it is derived, never set.
CREATE TABLE IF NOT EXISTS chunks (
    strategy   text NOT NULL,
    chunk_id   text NOT NULL,          -- <doc>#c<n>
    doc_id     text NOT NULL,
    n          integer NOT NULL,
    text       text NOT NULL,
    text_hash  text NOT NULL,          -- sha256(text); drift detection
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (strategy, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (strategy, doc_id);

-- One row per chunk per embedding model. The vector column is untyped so
-- models of different dimension share the table; queries always filter by
-- model, so operands match. No ANN index for the baseline: an exact scan over
-- ~16k rows is a few ms, and exactness keeps recall numbers free of index
-- approximation. text_hash records which text the vector was computed from.
CREATE TABLE IF NOT EXISTS embeddings (
    strategy   text NOT NULL,
    chunk_id   text NOT NULL,
    model      text NOT NULL,
    text_hash  text NOT NULL,
    embedding  vector NOT NULL,
    PRIMARY KEY (strategy, chunk_id, model),
    FOREIGN KEY (strategy, chunk_id) REFERENCES chunks (strategy, chunk_id) ON DELETE CASCADE
);
