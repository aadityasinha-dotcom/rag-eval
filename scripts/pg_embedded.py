"""Embedded PostgreSQL 16 + pgvector for development and demos, no Docker or root.

    uv run --extra pg python scripts/pg_embedded.py start | stop | status

Uses the `pgserver` package (bundled Postgres binaries, unix-socket only). Data
lives in .pg/data; the connection URL is written to .pg/DATABASE_URL, which
ingest.db.database_url() reads when DATABASE_URL is not set, so make index,
make eval and make test need no environment changes after `make pg-start`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PG_DIR = ROOT / ".pg"
PGDATA = PG_DIR / "data"
URL_FILE = PG_DIR / "DATABASE_URL"


def _server():  # type: ignore[no-untyped-def]
    try:
        import pgserver
    except ImportError:
        print("pgserver is not installed: run `uv sync --extra pg`", file=sys.stderr)
        raise SystemExit(1) from None
    PG_DIR.mkdir(exist_ok=True)
    # cleanup_mode=None: keep the server running after this process exits
    return pgserver.get_server(PGDATA, cleanup_mode=None)


def start() -> int:
    srv = _server()
    srv.psql("SET client_min_messages TO WARNING; CREATE EXTENSION IF NOT EXISTS vector;")
    uri = srv.get_uri()
    URL_FILE.write_text(uri + "\n", encoding="utf-8")
    version = (
        srv.psql(
            "SELECT split_part(version(), ' ', 2) || ' / pgvector ' || extversion "
            "FROM pg_extension WHERE extname = 'vector';"
        )
        .strip()
        .splitlines()
    )
    info = version[-2].strip() if len(version) > 1 else ""
    print(f"embedded postgres running (pid {srv.get_pid()}), {info}")
    print(f"DATABASE_URL={uri}")
    print(
        f"written to {URL_FILE.relative_to(ROOT)}; make index / make eval / make test will use it"
    )
    return 0


def stop() -> int:
    from pgserver._commands import pg_ctl

    if not PGDATA.exists():
        print("no embedded server data directory")
        return 0
    try:
        pg_ctl(["-w", "stop"], pgdata=PGDATA)
        print("embedded postgres stopped (data kept in .pg/data)")
    except Exception as e:  # already stopped, or never started
        print(f"not running ({str(e).strip().splitlines()[-1] if str(e).strip() else e})")
    if URL_FILE.exists():
        URL_FILE.unlink()
    return 0


def status() -> int:
    if not URL_FILE.exists():
        print("embedded postgres: not started (run `make pg-start`)")
        return 0
    import psycopg

    uri = URL_FILE.read_text(encoding="utf-8").strip()
    try:
        with psycopg.connect(uri, connect_timeout=3) as conn:
            n = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'chunks'"
            ).fetchone()
            rows = 0
            if n and n[0]:
                rows = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]  # type: ignore[index]
        print(f"embedded postgres: running, {rows} chunk rows indexed")
        print(f"DATABASE_URL={uri}")
        return 0
    except psycopg.OperationalError as e:
        print(f"embedded postgres: URL file exists but connection failed: {e}".splitlines()[0])
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pg_embedded", description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["start", "stop", "status"])
    args = ap.parse_args(argv)
    return {"start": start, "stop": stop, "status": status}[args.command]()


if __name__ == "__main__":
    sys.exit(main())
