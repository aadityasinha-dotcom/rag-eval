-- Executed by the postgres image on first start (empty data volume only).
-- Tables are created later by ingest/index.py; this only enables extensions.
CREATE EXTENSION IF NOT EXISTS vector;
