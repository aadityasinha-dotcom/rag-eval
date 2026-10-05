"""Pick the retriever a config names. Later phases register sparse/hybrid/rerank here."""

from __future__ import annotations

import psycopg

from evals.config import RunConfig
from ingest.embed import get_embedder
from retrieval.base import Retriever
from retrieval.dense import DenseRetriever


def build_retriever(config: RunConfig, conn: psycopg.Connection) -> Retriever:
    if config.rerank is not None:
        raise NotImplementedError("reranking is a later phase (CLAUDE.md)")
    if config.retrieval == "dense":
        return DenseRetriever(conn, get_embedder(config.embedding), config.chunking)
    raise NotImplementedError(f"retrieval={config.retrieval!r} is a later phase (CLAUDE.md)")
