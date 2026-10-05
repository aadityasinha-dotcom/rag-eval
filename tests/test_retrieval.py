"""retrieval/dense.py against Postgres (skips without a database) plus offline checks."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from evals.config import RunConfig
from ingest import db
from ingest.embed import HashEmbedder
from ingest.index import index_corpus
from retrieval.base import Hit
from retrieval.dense import DenseRetriever, search_vector

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
CONFIG = RunConfig(name="t", chunking="fixed512", embedding="hash64")


@pytest.fixture
def indexed(tmp_path: Path) -> Iterator[tuple[psycopg.Connection, HashEmbedder]]:
    try:
        c = db.connect()
    except psycopg.OperationalError as e:
        pytest.skip(f"no database: {e}".splitlines()[0])
    schema = f"test_{uuid.uuid4().hex[:8]}"
    c.execute(f"CREATE SCHEMA {schema}")
    c.execute(f"SET search_path TO {schema}, public")
    c.commit()
    emb = HashEmbedder()
    index_corpus(CONFIG, c, emb, FIXTURES)
    try:
        yield c, emb
    finally:
        c.rollback()
        c.execute(f"DROP SCHEMA {schema} CASCADE")
        c.commit()
        c.close()


def test_exact_chunk_text_scores_one_and_ranks_first(
    indexed: tuple[psycopg.Connection, HashEmbedder],
) -> None:
    conn, emb = indexed
    text = conn.execute("SELECT text FROM chunks WHERE chunk_id = 'sql-lock#c0'").fetchone()[0]
    r = DenseRetriever(conn, emb, "fixed512")
    hits = r.search(text, top_k=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "sql-lock#c0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert all(isinstance(h, Hit) and h.doc_id == h.chunk_id.split("#")[0] for h in hits)


def test_top_k_and_model_isolation(indexed: tuple[psycopg.Connection, HashEmbedder]) -> None:
    conn, emb = indexed
    (vec,) = emb.embed(["write-ahead logging"])
    assert len(search_vector(conn, vec, "fixed512", "hash64", 7)) == 7
    assert search_vector(conn, vec, "fixed512", "no-such-model", 5) == []
    assert search_vector(conn, vec, "no-such-strategy", "hash64", 5) == []
    assert DenseRetriever(conn, emb, "fixed512").name == "dense(fixed512/hash64)"


def test_dense_cli_without_database_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from retrieval import dense

    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nope")
    rc = dense.main(["lock a table", "--config", "evals/configs/smoke-hash64.yaml"])
    assert rc == 1 and "error:" in capsys.readouterr().err
