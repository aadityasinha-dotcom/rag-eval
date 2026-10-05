"""Postgres connection and schema. DATABASE_URL from the environment or .env."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = ROOT / "db" / "schema.sql"
DEFAULT_URL = "postgresql://rag:rag@localhost:5432/rag_eval"


def database_url() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect(url: str | None = None) -> psycopg.Connection:
    """Open a connection with the vector type registered. Ensures the extension exists."""
    conn = psycopg.connect(url or database_url())
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    """Idempotent: db/schema.sql uses CREATE ... IF NOT EXISTS throughout."""
    conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.commit()
