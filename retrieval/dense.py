"""Dense retrieval: embed the query, cosine similarity against pgvector, top k.

Plain SQL on the tables ingest/index.py fills (CLAUDE.md rule 6). `<=>` is
pgvector's cosine distance; score = 1 - distance, so 1.0 is identical and
0.0 orthogonal. With no ANN index this is an exact scan, which is what the
baseline wants: recall numbers reflect the embedding, not an index's recall.

    python -m retrieval.dense "how do I lock a table without waiting" \
        [--config evals/configs/baseline.yaml] [--top-k 5]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg
from pgvector import Vector

from evals.config import load_config
from ingest import db
from ingest.embed import Embedder, get_embedder
from retrieval.base import Hit

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "evals" / "configs" / "baseline.yaml"

_SQL = """
SELECT c.chunk_id, c.doc_id, c.text, 1 - (e.embedding <=> %(q)s) AS score
  FROM embeddings e
  JOIN chunks c ON c.strategy = e.strategy AND c.chunk_id = e.chunk_id
 WHERE e.strategy = %(strategy)s AND e.model = %(model)s
 ORDER BY e.embedding <=> %(q)s, c.chunk_id
 LIMIT %(k)s
"""


def search_vector(
    conn: psycopg.Connection,
    query_vec: Sequence[float],
    strategy: str,
    model: str,
    top_k: int,
) -> list[Hit]:
    """Top-k by cosine similarity for an already-embedded query."""
    rows = conn.execute(
        _SQL,
        {"q": Vector(list(query_vec)), "strategy": strategy, "model": model, "k": top_k},
    ).fetchall()
    return [Hit(chunk_id=r[0], doc_id=r[1], text=r[2], score=float(r[3])) for r in rows]


class DenseRetriever:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder, strategy: str) -> None:
        self.conn = conn
        self.embedder = embedder
        self.strategy = strategy

    @property
    def name(self) -> str:
        return f"dense({self.strategy}/{self.embedder.name})"

    def search(self, query: str, top_k: int) -> list[Hit]:
        (vec,) = self.embedder.embed([query])
        return search_vector(self.conn, vec, self.strategy, self.embedder.name, top_k)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m retrieval.dense", description=__doc__.splitlines()[0]
    )
    ap.add_argument("query")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--top-k", type=int, default=None, help="override the config's top_k")
    ap.add_argument("--full", action="store_true", help="print whole chunk texts")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        embedder = get_embedder(cfg.embedding)
        with db.connect() as conn:
            hits = DenseRetriever(conn, embedder, cfg.chunking).search(
                args.query, args.top_k or cfg.top_k
            )
    except (ValueError, RuntimeError, psycopg.OperationalError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not hits:
        print(f"no hits: is the index built? (make index CONFIG={args.config})", file=sys.stderr)
        return 1
    for rank, h in enumerate(hits, start=1):
        body = h.text if args.full else " ".join(h.text.split())[:160] + "..."
        print(f"{rank:2d}. {h.score:.4f}  {h.chunk_id}\n    {body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
