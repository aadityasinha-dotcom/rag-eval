"""golden.jsonl validation: shape, difficulty rules, and chunk ids resolving."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import golden
from evals.golden import GoldenEntry, load_golden, validate

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"
CHUNKS = {"doc1#c0", "doc1#c1", "doc3#c0", "sql-lock#c0", "sql-lock#c1"}


def entry(**kw: object) -> GoldenEntry:
    base: dict[str, object] = {
        "id": "q001",
        "question": "What does LOCK TABLE do?",
        "answer": "Obtains a table-level lock.",
        "relevant_chunk_ids": ("sql-lock#c0",),
        "difficulty": "single-hop",
    }
    base.update(kw)
    return GoldenEntry(**base)  # type: ignore[arg-type]


def write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_valid_entries_pass() -> None:
    entries = [
        entry(),
        entry(
            id="q002",
            question="Multi?",
            relevant_chunk_ids=("doc1#c0", "sql-lock#c1"),
            difficulty="multi-hop",
        ),
        entry(
            id="q003",
            question="Nope?",
            answer="Not in corpus.",
            relevant_chunk_ids=(),
            difficulty="unanswerable",
        ),
    ]
    assert validate(entries, CHUNKS) == []


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"relevant_chunk_ids": ("sql-lock#c99",)}, "unknown chunk id"),
        ({"relevant_chunk_ids": ()}, "no relevant_chunk_ids"),
        ({"difficulty": "unanswerable"}, "must have no relevant_chunk_ids"),
        ({"difficulty": "hard"}, "difficulty 'hard'"),
        ({"answer": "  "}, "empty answer"),
        ({"relevant_chunk_ids": ("sql-lock#c0", "sql-lock#c0")}, "repeated chunk id"),
    ],
)
def test_single_entry_problems(kw: dict[str, object], needle: str) -> None:
    problems = validate([entry(**kw)], CHUNKS)
    assert any(needle in p for p in problems), problems


def test_duplicates_detected() -> None:
    problems = validate([entry(), entry(question="what does lock  table do?")], CHUNKS)
    assert any("duplicate id" in p for p in problems)
    assert any("duplicate question" in p for p in problems)


def test_load_golden_reports_line_numbers(tmp_path: Path) -> None:
    p = tmp_path / "golden.jsonl"
    p.write_text(
        '{"id": "q001"}\nnot json\n\n{"id": "q2", "question": "q", "answer": "a", '
        '"relevant_chunk_ids": ["x"], "difficulty": "single-hop", "notes": "?"}\n'
    )
    with pytest.raises(ValueError) as exc:
        load_golden(p)
    msg = str(exc.value)
    assert "line 1: missing" in msg
    assert "line 2: invalid JSON" in msg
    assert "line 4" in msg and "unexpected ['notes']" in msg


def test_cli_check_against_fixture_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = write(
        tmp_path,
        [
            {
                "id": "q001",
                "question": "What does LOCK TABLE do?",
                "answer": "Takes a table lock.",
                "relevant_chunk_ids": ["sql-lock#c0"],
                "difficulty": "single-hop",
            },
        ],
    )
    rc = golden.main(["--check", "--golden", str(good), "--corpus", str(FIXTURES)])
    out = capsys.readouterr().out
    assert rc == 0 and out.startswith("ok: 1 entries (target >= 50)")

    bad = write(
        tmp_path,
        [
            {
                "id": "q001",
                "question": "?",
                "answer": "a",
                "relevant_chunk_ids": ["doc9#c0"],
                "difficulty": "single-hop",
            },
        ],
    )
    rc = golden.main(["--check", "--golden", str(bad), "--corpus", str(FIXTURES)])
    captured = capsys.readouterr()
    assert rc == 1 and "unknown chunk id 'doc9#c0'" in captured.err
    assert captured.out.startswith("FAIL:")

    rc = golden.main(
        ["--check", "--golden", str(tmp_path / "missing.jsonl"), "--corpus", str(FIXTURES)]
    )
    assert rc == 1


def test_cli_show_prints_chunk_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    g = write(
        tmp_path,
        [
            {
                "id": "q001",
                "question": "What does LOCK TABLE do?",
                "answer": "Takes a table lock.",
                "relevant_chunk_ids": ["sql-lock#c0"],
                "difficulty": "single-hop",
            },
        ],
    )
    assert golden.main(["--show", "q001", "--golden", str(g), "--corpus", str(FIXTURES)]) == 0
    out = capsys.readouterr().out
    assert "Q: What does LOCK TABLE do?" in out
    assert "== sql-lock#c0" in out and "obtains a table-level lock" in out
