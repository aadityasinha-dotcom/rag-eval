"""Embedders: offline determinism, cache behaviour, OpenAI batching (stubbed)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ingest.embed import EMBEDDERS, CachedEmbedder, HashEmbedder, OpenAIEmbedder, get_embedder


@dataclass
class CountingEmbedder:
    """Records every text it is asked to embed."""

    name: str = "counting"
    dim: int = 4
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Any) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


def test_hash_embedder_is_deterministic_and_unit_length() -> None:
    e = HashEmbedder()
    a, b = e.embed(["LOCK TABLE obtains a lock", "LOCK TABLE obtains a lock"])
    assert a == b and len(a) == e.dim == 64
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
    (c,) = e.embed(["something entirely different"])
    assert c != a
    assert e.embed([""]) == [[0.0] * 64]


def test_cache_hits_on_second_call_and_dedupes_within_batch(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, cache_dir=tmp_path)
    out = cached.embed(["aa", "bbb", "aa"])
    assert out == [[2.0, 1, 0, 0], [3.0, 1, 0, 0], [2.0, 1, 0, 0]]
    assert inner.calls == [["aa", "bbb"]]  # duplicates collapsed
    assert (cached.hits, cached.misses) == (0, 2)

    again = CachedEmbedder(CountingEmbedder(), cache_dir=tmp_path)  # fresh wrapper, same disk
    assert again.embed(["bbb", "aa", "cccc"]) == [[3.0, 1, 0, 0], [2.0, 1, 0, 0], [4.0, 1, 0, 0]]
    assert again.inner.calls == [["cccc"]]  # type: ignore[attr-defined]
    assert (again.hits, again.misses) == (2, 1)


def test_cache_is_per_model(tmp_path: Path) -> None:
    CachedEmbedder(CountingEmbedder(name="m1"), cache_dir=tmp_path).embed(["x"])
    other = CountingEmbedder(name="m2")
    CachedEmbedder(other, cache_dir=tmp_path).embed(["x"])
    assert other.calls == [["x"]]


def test_cache_rejects_wrong_dimension(tmp_path: Path) -> None:
    bad = CountingEmbedder(dim=3)
    with pytest.raises(RuntimeError, match="expected dim 3"):
        CachedEmbedder(bad, cache_dir=tmp_path).embed(["x"])


def test_openai_embedder_batches_and_restores_order() -> None:
    requests: list[list[str]] = []

    def create(*, model: str, input: list[str]) -> Any:
        requests.append(input)
        # return rows out of order to prove we sort by index
        data = [SimpleNamespace(index=i, embedding=[float(i), 0.5]) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    e = OpenAIEmbedder("text-embedding-3-small", dim=2, batch_size=2, client=client)
    out = e.embed(["a", "b", "c", "d", "e"])
    assert requests == [["a", "b"], ["c", "d"], ["e"]]
    assert out == [[0, 0.5], [1, 0.5], [0, 0.5], [1, 0.5], [0, 0.5]]


def test_openai_embedder_needs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder("text-embedding-3-small", dim=1536).embed(["x"])


def test_registry() -> None:
    assert set(EMBEDDERS) == {"text-embedding-3-small", "hash64"}
    e = get_embedder("text-embedding-3-small")
    assert (e.name, e.dim) == ("text-embedding-3-small", 1536)
    assert get_embedder("hash64", cache=False).dim == 64
    with pytest.raises(ValueError):
        get_embedder("nope")
