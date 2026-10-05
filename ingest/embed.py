"""Embedding models behind one interface, with a content-addressed disk cache.

EMBEDDERS maps a config name to a factory. `text-embedding-3-small` is the
baseline (CLAUDE.md stack); `hash64` is a deterministic offline embedder for
tests and smoke runs whose numbers mean nothing.

The cache (.cache/embeddings/<model>.sqlite) is keyed by sha256 of the text,
so re-indexing after a chunking change only pays for chunks whose text is new,
and re-running an unchanged corpus makes no API calls at all.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ingest.parse import CACHE_DIR

_WORD = re.compile(r"\w+")


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One unit-length vector per input text, in input order."""
        ...


# ---------------------------------------------------------------- offline


@dataclass(frozen=True)
class HashEmbedder:
    """Hashed bag of words, L2-normalised. Deterministic, dependency-free, useless for quality."""

    name: str = "hash64"
    dim: int = 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _WORD.findall(text.lower()):
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            vec[idx] += 1.0 if h[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


# ----------------------------------------------------------------- openai


class OpenAIEmbedder:
    """text-embedding-3-* via the openai SDK. The client is created lazily so
    importing this module never needs a key; the SDK retries transient errors."""

    def __init__(
        self,
        model: str,
        dim: int,
        batch_size: int = 256,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._dim = dim
        self.batch_size = batch_size
        self._client = client

    @property
    def name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def _get_client(self) -> Any:
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; needed for embedding "
                    f"{self._model}. Use embedding: hash64 for an offline smoke run."
                )
            from openai import OpenAI

            self._client = OpenAI(max_retries=5)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        client = self._get_client()
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = client.embeddings.create(model=self._model, input=batch)
            rows = sorted(resp.data, key=lambda d: d.index)
            if len(rows) != len(batch):
                raise RuntimeError(
                    f"{self._model}: asked for {len(batch)} embeddings, got {len(rows)}"
                )
            out.extend(list(r.embedding) for r in rows)
        return out


# ------------------------------------------------------------------ cache


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CachedEmbedder:
    """Wraps any Embedder with a sqlite cache keyed by (model, sha256(text))."""

    inner: Embedder
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR / "embeddings")
    hits: int = 0
    misses: int = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    def _db(self) -> sqlite3.Connection:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.inner.name)
        conn = sqlite3.connect(self.cache_dir / f"{safe}.sqlite")
        conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        return conn

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [text_key(t) for t in texts]
        with self._db() as db:
            found: dict[str, list[float]] = {}
            unique = list(dict.fromkeys(keys))
            for i in range(0, len(unique), 900):  # sqlite parameter limit
                part = unique[i : i + 900]
                marks = ",".join("?" * len(part))
                for key, blob in db.execute(
                    f"SELECT key, vec FROM vectors WHERE key IN ({marks})", part
                ):
                    found[key] = _unpack(blob)
            missing_keys = [k for k in unique if k not in found]
            self.hits += len(unique) - len(missing_keys)
            self.misses += len(missing_keys)
            if missing_keys:
                first_text = {k: t for k, t in zip(keys, texts, strict=True)}
                vectors = self.inner.embed([first_text[k] for k in missing_keys])
                for k, v in zip(missing_keys, vectors, strict=True):
                    if len(v) != self.dim:
                        raise RuntimeError(f"{self.name}: expected dim {self.dim}, got {len(v)}")
                    found[k] = list(v)
                db.executemany(
                    "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                    [(k, _pack(found[k])) for k in missing_keys],
                )
        return [found[k] for k in keys]


def _pack(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


# --------------------------------------------------------------- registry

EMBEDDERS: dict[str, Callable[[], Embedder]] = {
    "text-embedding-3-small": lambda: OpenAIEmbedder("text-embedding-3-small", dim=1536),
    "hash64": HashEmbedder,
}


def get_embedder(name: str, cache: bool = True) -> Embedder:
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedding {name!r}; known: {sorted(EMBEDDERS)}")
    inner = EMBEDDERS[name]()
    return CachedEmbedder(inner) if cache else inner
