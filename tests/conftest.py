"""Shared fixtures. Database-backed tests skip when Postgres is unreachable,
except in CI (RAG_EVAL_REQUIRE_DB=1), where a missing database is a failure so
the schema, indexing and retrieval paths are always exercised there."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from ingest import db


@pytest.fixture
def pg() -> Iterator[psycopg.Connection]:
    """A connection whose search_path points at a throwaway schema, dropped afterwards."""
    try:
        conn = db.connect()
    except psycopg.OperationalError as e:
        reason = f"no database: {e}".splitlines()[0]
        if os.environ.get("RAG_EVAL_REQUIRE_DB"):
            pytest.fail(f"RAG_EVAL_REQUIRE_DB is set but {reason}")
        pytest.skip(reason)
    schema = f"test_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    conn.commit()
    try:
        yield conn
    finally:
        conn.rollback()
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        conn.close()
