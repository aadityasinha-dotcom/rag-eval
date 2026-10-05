"""Index the corpus for a config: chunks + tsvector + embeddings into Postgres.

    python -m ingest.index evals/configs/baseline.yaml [--corpus DIR] [--no-cache]

Idempotent and incremental. Chunk rows are upserted by (strategy, chunk_id)
and only rewritten when text_hash changed; rows for chunk ids that no longer
exist are deleted (their embeddings cascade). Embeddings are computed only for
chunks that have none for this model or whose text changed since, and the
embedding cache makes even that free when the text was seen before.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector import Vector

from evals.config import RunConfig, load_config
from ingest import db
from ingest.chunk import STRATEGIES, chunk_corpus
from ingest.embed import Embedder, get_embedder
from ingest.parse import DEFAULT_CORPUS, load_corpus

EMBED_BATCH = 512  # chunks per embed()+insert round trip


@dataclass(frozen=True)
class IndexStats:
    docs: int
    chunks: int
    chunks_written: int  # inserted or text changed
    chunks_deleted: int
    embedded: int  # vectors computed or refreshed this run
    seconds: float


def index_corpus(
    config: RunConfig,
    conn: psycopg.Connection,
    embedder: Embedder,
    corpus: Path = DEFAULT_CORPUS,
    log: bool = False,
) -> IndexStats:
    t0 = time.perf_counter()
    strategy = STRATEGIES[config.chunking]
    docs = load_corpus(corpus)
    chunks = chunk_corpus(docs, strategy)
    if log:
        print(f"{len(docs)} docs -> {len(chunks)} chunks ({strategy.name})")

    db.ensure_schema(conn)
    with conn.transaction():
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO chunks (strategy, chunk_id, doc_id, n, text, text_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (strategy, chunk_id) DO UPDATE
               SET doc_id = EXCLUDED.doc_id, n = EXCLUDED.n,
                   text = EXCLUDED.text, text_hash = EXCLUDED.text_hash
             WHERE chunks.text_hash IS DISTINCT FROM EXCLUDED.text_hash
            """,
            [(strategy.name, c.chunk_id, c.doc_id, c.n, c.text, c.text_hash) for c in chunks],
            returning=False,
        )
        written = cur.rowcount
        cur.execute(
            "DELETE FROM chunks WHERE strategy = %s AND NOT (chunk_id = ANY(%s))",
            (strategy.name, [c.chunk_id for c in chunks]),
        )
        deleted = cur.rowcount

        cur.execute(
            """
            SELECT c.chunk_id, c.text, c.text_hash
              FROM chunks c
              LEFT JOIN embeddings e
                ON e.strategy = c.strategy AND e.chunk_id = c.chunk_id AND e.model = %s
             WHERE c.strategy = %s AND (e.chunk_id IS NULL OR e.text_hash <> c.text_hash)
             ORDER BY c.chunk_id
            """,
            (embedder.name, strategy.name),
        )
        todo = cur.fetchall()
        if log:
            print(f"{written} chunk rows written, {deleted} deleted, {len(todo)} to embed")

        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i : i + EMBED_BATCH]
            vectors = embedder.embed([text for _, text, _ in batch])
            cur.executemany(
                """
                INSERT INTO embeddings (strategy, chunk_id, model, text_hash, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (strategy, chunk_id, model) DO UPDATE
                   SET text_hash = EXCLUDED.text_hash, embedding = EXCLUDED.embedding
                """,
                [
                    (strategy.name, cid, embedder.name, h, Vector(v))
                    for (cid, _, h), v in zip(batch, vectors, strict=True)
                ],
            )
            if log:
                print(f"  embedded {min(i + EMBED_BATCH, len(todo))}/{len(todo)}")

    return IndexStats(
        docs=len(docs),
        chunks=len(chunks),
        chunks_written=written,
        chunks_deleted=deleted,
        embedded=len(todo),
        seconds=time.perf_counter() - t0,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ingest.index", description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path, help="evals/configs/<variant>.yaml")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--no-cache", action="store_true", help="bypass the embedding cache")
    args = ap.parse_args(argv)
    try:
        config = load_config(args.config)
        embedder = get_embedder(config.embedding, cache=not args.no_cache)
        with db.connect() as conn:
            stats = index_corpus(config, conn, embedder, args.corpus, log=True)
    except (ValueError, RuntimeError, psycopg.OperationalError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(
        f"done: {stats.docs} docs, {stats.chunks} chunks, {stats.embedded} embedded "
        f"({config.chunking} / {embedder.name}) in {stats.seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
