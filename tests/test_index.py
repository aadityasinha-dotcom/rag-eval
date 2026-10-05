"""ingest/index.py against a real Postgres. Skipped when no database is reachable.

Run `make dev` (or set DATABASE_URL) to enable. Each test works inside a
throwaway schema so the real tables are never touched.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from evals.config import RunConfig
from ingest import db
from ingest.embed import HashEmbedder
from ingest.index import index_corpus

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
CONFIG = RunConfig(name="t", chunking="fixed512", embedding="hash64")


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    try:
        c = db.connect()
    except psycopg.OperationalError as e:
        pytest.skip(f"no database: {e}".splitlines()[0])
    schema = f"test_{uuid.uuid4().hex[:8]}"
    c.execute(f"CREATE SCHEMA {schema}")
    c.execute(f"SET search_path TO {schema}, public")
    c.commit()
    try:
        yield c
    finally:
        c.rollback()
        c.execute(f"DROP SCHEMA {schema} CASCADE")
        c.commit()
        c.close()


def test_index_is_incremental_and_tracks_edits(conn: psycopg.Connection, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES, corpus)
    emb = HashEmbedder()

    first = index_corpus(CONFIG, conn, emb, corpus)
    assert first.docs == 5 and first.chunks_written == first.chunks == first.embedded > 10
    assert first.chunks_deleted == 0

    # unchanged corpus: nothing written, nothing embedded
    second = index_corpus(CONFIG, conn, emb, corpus)
    assert (second.chunks_written, second.chunks_deleted, second.embedded) == (0, 0, 0)

    # tsvector is populated and searchable (sparse retrieval later)
    row = conn.execute(
        "SELECT chunk_id FROM chunks WHERE strategy = 'fixed512' "
        "AND tsv @@ to_tsquery('english', 'nowait')"
    ).fetchall()
    assert row == [("sql-lock#c0",)]

    # vector round-trips exactly and matches what the embedder produces
    (vec,) = conn.execute(
        "SELECT embedding FROM embeddings WHERE chunk_id = 'sql-lock#c0' AND model = 'hash64'"
    ).fetchone()
    text = conn.execute("SELECT text FROM chunks WHERE chunk_id = 'sql-lock#c0'").fetchone()[0]
    assert len(vec) == 64
    assert [round(float(x), 5) for x in vec] == [round(x, 5) for x in emb.embed([text])[0]]

    # edit the tail of doc3: only the last window(s) change text, ids stay
    doc3 = corpus / "doc3.txt"
    doc3.write_text(doc3.read_text().replace("Sentence 40", "Sentence 40 EDITED"))
    third = index_corpus(CONFIG, conn, emb, corpus)
    assert third.chunks_deleted == 0
    assert 1 <= third.chunks_written == third.embedded <= 2  # only the last window(s) changed

    # shrink a doc so it loses chunks: stale rows and their embeddings go away
    doc3.write_text("tiny")
    fourth = index_corpus(CONFIG, conn, emb, corpus)
    assert fourth.chunks_deleted >= 10
    left = conn.execute("SELECT count(*) FROM embeddings WHERE chunk_id LIKE 'doc3#%'").fetchone()[
        0
    ]
    assert left == 1
