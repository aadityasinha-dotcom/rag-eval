"""Shared retrieval types. Every retriever (dense, sparse, hybrid, reranked)
returns the same Hit list so the eval harness and generation never care which
one produced it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    score: float  # higher is better, retriever-specific scale
    text: str


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: str, top_k: int) -> list[Hit]:
        """Top-k chunks for a natural-language query, best first."""
        ...
