"""evals/instrument.py: a pass-through without a key, real llm-metrics events with one."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from evals import instrument
from evals.config import RunConfig
from evals.golden import GoldenEntry
from evals.run import run
from retrieval.base import Hit
from tests.test_run import ENTRIES, TABLE, ScriptedGenerator, ScriptedJudge, ScriptedRetriever

CFG = RunConfig(name="t", top_k=3)


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_METRICS_API_KEY", raising=False)
    monkeypatch.setattr(instrument, "load_dotenv", lambda *_: None)


def test_disabled_without_key_is_a_pure_pass_through(no_key: None) -> None:
    assert instrument.enabled() is False
    assert instrument.setup(release="abc") is False
    calls: list[tuple[Any, ...]] = []
    assert instrument.traced("n", "span", lambda *a: calls.append(a) or 42, 1, 2) == 42
    assert calls == [(1, 2)]
    sentinel = object()
    assert instrument.wrap_anthropic(sentinel) is sentinel
    assert instrument.wrap_openai(sentinel) is sentinel
    instrument.record_score("x", 1.0, source="heuristic")  # no-op, must not raise
    instrument.annotate_trace(tags=["a"])
    instrument.flush()
    assert instrument.current_trace_id() is None


def test_disabled_when_sdk_missing_even_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_METRICS_API_KEY", "k")
    monkeypatch.setattr(instrument, "load_dotenv", lambda *_: None)
    monkeypatch.setitem(sys.modules, "llm_metrics", None)  # makes `import llm_metrics` fail
    assert instrument.enabled() is False
    assert instrument.traced("n", "span", lambda: "ok") == "ok"


def test_run_emits_trace_retrieval_and_scores_per_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm_metrics = pytest.importorskip("llm_metrics")
    from llm_metrics.buffer import EventBuffer

    captured: list[dict[str, Any]] = []
    buffer = EventBuffer(flush_fn=lambda payload, deadline: captured.extend(payload), start=False)
    llm_metrics.configure(api_key="test-key", sink=buffer, enabled=True)
    monkeypatch.setenv("LLM_METRICS_API_KEY", "test-key")
    monkeypatch.setattr(instrument, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(instrument, "setup", lambda release, environment="eval": True)
    monkeypatch.setattr(instrument, "flush", lambda timeout=10.0: None)  # drain by hand below

    res = run(CFG, ENTRIES, ScriptedRetriever(TABLE), ScriptedGenerator(), judge=ScriptedJudge())
    buffer.flush_once()
    llm_metrics.configure(enabled=False)

    assert res.instrumented is True
    assert all(q.trace_id for q in res.queries)
    assert len({q.trace_id for q in res.queries}) == 3  # one trace per golden query

    by_type: dict[str, list[dict[str, Any]]] = {}
    for ev in captured:
        by_type.setdefault(ev["type"], []).append(ev)
    assert [t["name"] for t in by_type["trace"]] == ["rag-eval.query"] * 3
    assert len(by_type["retrieval"]) == 3 and by_type["retrieval"][0]["name"] == "retrieve"
    assert len(by_type["span"]) == 3  # the per-query roots
    scores = {(s["name"], s["trace_id"]) for s in by_type["score"]}
    q1 = next(q.trace_id for q in res.queries if q.id == "q1")
    q3 = next(q.trace_id for q in res.queries if q.id == "q3")
    assert {
        ("recall@3", q1),
        ("mrr", q1),
        ("ndcg@3", q1),
        ("abstained", q1),
        ("faithfulness", q1),
    } <= scores
    assert ("recall@3", q3) not in scores  # unanswerable: no retrieval metrics
    assert ("abstained", q3) in scores and ("faithfulness", q3) not in scores
    sources = {s["name"]: s["source"] for s in by_type["score"]}
    assert sources["faithfulness"] == "llm_judge" and sources["recall@3"] == "heuristic"


def test_setup_stamps_release(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_metrics = pytest.importorskip("llm_metrics")
    from llm_metrics._runtime import current_settings

    monkeypatch.setenv("LLM_METRICS_API_KEY", "test-key")
    monkeypatch.delenv("LLM_METRICS_RELEASE", raising=False)
    monkeypatch.setattr(instrument, "load_dotenv", lambda *_: None)
    assert instrument.setup(release="deadbeef") is True
    assert current_settings().release == "deadbeef"
    assert current_settings().environment == "eval"
    llm_metrics.configure(enabled=False)
    _ = Hit, GoldenEntry  # imported for typing parity with the other test modules
