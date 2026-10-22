"""Faithfulness wrapper: gating, NaN handling, judge construction (no network)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from evals.metrics.generation import (
    DEFAULT_JUDGE,
    RagasFaithfulness,
    build_ragas_llm,
    sampling_args,
)


class FakeMetric:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.samples: list[Any] = []

    def single_turn_score(self, sample: Any) -> Any:
        self.samples.append(sample)
        return self.value


def test_score_builds_ragas_sample_and_returns_float() -> None:
    m = FakeMetric(0.75)
    j = RagasFaithfulness("judge-x", metric=m)
    assert j.name == "ragas-faithfulness/judge-x"
    assert j.score("q?", "a [x#c0]", ["ctx1", "ctx2"]) == 0.75
    (s,) = m.samples
    assert (s.user_input, s.response, s.retrieved_contexts) == ("q?", "a [x#c0]", ["ctx1", "ctx2"])


def test_nan_or_none_from_ragas_becomes_none() -> None:
    assert RagasFaithfulness(metric=FakeMetric(math.nan)).score("q", "a", ["c"]) is None
    assert RagasFaithfulness(metric=FakeMetric(None)).score("q", "a", ["c"]) is None


def test_sampling_args_by_model_family() -> None:
    assert sampling_args("claude-sonnet-4-6") == {"temperature": 0.0}
    assert sampling_args("claude-haiku-4-5") == {"temperature": 0.0}
    assert sampling_args("claude-sonnet-5-5") == {}
    assert sampling_args("claude-opus-5-5") == {}


def test_build_ragas_llm_uses_json_mode_and_no_top_p() -> None:
    import anthropic
    import instructor

    llm = build_ragas_llm(DEFAULT_JUDGE, client=anthropic.Anthropic(api_key="sk-test"))
    assert llm.model == DEFAULT_JUDGE and llm.provider == "anthropic"
    assert llm.model_args == {"max_tokens": 4096, "temperature": 0.0}
    # instructor v2 normalises ANTHROPIC_JSON to its generic JSON mode; what matters is not TOOLS
    assert llm.client.mode in (instructor.Mode.JSON, instructor.Mode.ANTHROPIC_JSON)
    assert llm.client.mode not in (instructor.Mode.TOOLS, instructor.Mode.ANTHROPIC_TOOLS)
    assert build_ragas_llm(
        "claude-sonnet-5-5", client=anthropic.Anthropic(api_key="x")
    ).model_args == {"max_tokens": 4096}


def test_build_ragas_llm_needs_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_ragas_llm(DEFAULT_JUDGE)
