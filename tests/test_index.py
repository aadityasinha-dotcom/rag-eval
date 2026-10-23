"""ingest/index.py against a real Postgres (fixture `pg` in conftest.py: skipped
without a database, required in CI). Each test works inside a throwaway schema."""

from __future__ import annotations

import shutil
from pathlib import Path

import psycopg
import pytest

from evals.config import RunConfig
from ingest.embed import HashEmbedder
from ingest.index import index_corpus

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
CONFIG = RunConfig(name="t", chunking="fixed512", embedding="hash64")


def test_index_is_incremental_and_tracks_edits(pg: psycopg.Connection, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES, corpus)
    emb = HashEmbedder()

    first = index_corpus(CONFIG, pg, emb, corpus)
    assert first.docs == 5 and first.chunks_written == first.chunks == first.embedded > 10
    assert first.chunks_deleted == 0

    # unchanged corpus: nothing written, nothing embedded
    second = index_corpus(CONFIG, pg, emb, corpus)
    assert (second.chunks_written, second.chunks_deleted, second.embedded) == (0, 0, 0)

    # tsvector is populated and searchable (sparse retrieval later)
    row = pg.execute(
        "SELECT chunk_id FROM chunks WHERE strategy = 'fixed512' "
        "AND tsv @@ to_tsquery('english', 'nowait')"
    ).fetchall()
    assert row == [("sql-lock#c0",)]

    # vector round-trips exactly and matches what the embedder produces
    (vec,) = pg.execute(
        "SELECT embedding FROM embeddings WHERE chunk_id = 'sql-lock#c0' AND model = 'hash64'"
    ).fetchone()
    text = pg.execute("SELECT text FROM chunks WHERE chunk_id = 'sql-lock#c0'").fetchone()[0]
    values = vec.to_list() if hasattr(vec, "to_list") else list(vec)  # pgvector returns Vector
    assert len(values) == 64
    assert [round(float(x), 5) for x in values] == [round(x, 5) for x in emb.embed([text])[0]]

    # edit the tail of doc3: only the last window(s) change text, ids stay
    doc3 = corpus / "doc3.txt"
    doc3.write_text(doc3.read_text().replace("Sentence 40", "Sentence 40 EDITED"))
    third = index_corpus(CONFIG, pg, emb, corpus)
    assert third.chunks_deleted == 0
    assert 1 <= third.chunks_written == third.embedded <= 2  # only the last window(s) changed

    # shrink a doc so it loses chunks: stale rows and their embeddings go away
    doc3.write_text("tiny")
    fourth = index_corpus(CONFIG, pg, emb, corpus)
    assert fourth.chunks_deleted >= 10
    left = pg.execute("SELECT count(*) FROM embeddings WHERE chunk_id LIKE 'doc3#%'").fetchone()[0]
    assert left == 1


def test_database_url_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest import db

    monkeypatch.setattr(db, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(db, "EMBEDDED_URL_FILE", tmp_path / "DATABASE_URL")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.database_url() == db.DEFAULT_URL  # nothing configured: docker default
    (tmp_path / "DATABASE_URL").write_text("postgresql://postgres:@/postgres?host=/tmp/sock\n")
    assert db.database_url() == "postgresql://postgres:@/postgres?host=/tmp/sock"  # embedded
    monkeypatch.setenv("DATABASE_URL", "postgresql://x@h/d")
    assert db.database_url() == "postgresql://x@h/d"  # explicit env wins
