"""generation/answer.py with a stubbed Anthropic client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from generation.answer import ClaudeGenerator, abstained, parse_citations
from generation.prompts import ABSTAIN_PREFIX, PROMPTS, V1
from retrieval.base import Hit

HITS = [
    Hit("sql-lock#c0", "sql-lock", 0.9, "LOCK TABLE obtains a table-level lock."),
    Hit("sql-lock#c1", "sql-lock", 0.8, "NOWAIT means do not wait."),
]


def test_parse_citations_dedupes_and_ignores_noise() -> None:
    text = "Locks [sql-lock#c0] and again [sql-lock#c0], see [ddl-basics#c12]. Not [this] or [a#b]."
    assert parse_citations(text) == ("sql-lock#c0", "ddl-basics#c12")
    assert abstained(f"  {ABSTAIN_PREFIX} The chunks cover locks only.")
    assert not abstained("LOCK TABLE obtains a lock [sql-lock#c0]")


def test_prompt_wraps_chunks_with_ids() -> None:
    user = V1.user("What does NOWAIT do?", HITS)
    assert '<chunk id="sql-lock#c1">\nNOWAIT means do not wait.\n</chunk>' in user
    assert user.endswith("<question>\nWhat does NOWAIT do?\n</question>")
    assert PROMPTS["v1"] is V1 and ABSTAIN_PREFIX in V1.system


def test_claude_generator_builds_request_and_parses_response() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kw: Any) -> Any:
        calls.append(kw)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="It does not wait [sql-lock#c1]."),
            ],
            usage=SimpleNamespace(input_tokens=120, output_tokens=12),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    gen = ClaudeGenerator("claude-opus-5-5", V1, client=client)
    a = gen.answer("What does NOWAIT do?", HITS)
    assert gen.name == "claude-opus-5-5/v1"
    assert a.text == "It does not wait [sql-lock#c1]." and a.cited_chunk_ids == ("sql-lock#c1",)
    assert (a.abstained, a.refused, a.input_tokens, a.output_tokens) == (False, False, 120, 12)
    (kw,) = calls
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "medium"}
    assert kw["system"][0]["text"] == V1.system and "cache_control" in kw["system"][0]
    assert kw["messages"] == [{"role": "user", "content": V1.user("What does NOWAIT do?", HITS)}]


def test_refusal_is_recorded() -> None:
    def create(**kw: Any) -> Any:
        return SimpleNamespace(
            stop_reason="refusal",
            content=[],
            usage=SimpleNamespace(input_tokens=1, output_tokens=0),
        )

    gen = ClaudeGenerator(
        "claude-opus-5-5", V1, client=SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    a = gen.answer("q", HITS)
    assert a.refused and a.text == "" and a.cited_chunk_ids == ()


def test_generator_needs_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeGenerator("claude-opus-5-5", V1).answer("q", HITS)
