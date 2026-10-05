"""evals/run.py with a scripted retriever and generator: no database, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.config import RunConfig
from evals.golden import GoldenEntry
from evals.run import RESULTS_HEADER, append_results, results_row, run, summary, write_run_file
from generation.answer import Answer
from retrieval.base import Hit

CFG = RunConfig(name="t", top_k=3)


class ScriptedRetriever:
    name = "scripted"

    def __init__(self, table: dict[str, list[str]]) -> None:
        self.table = table

    def search(self, query: str, top_k: int) -> list[Hit]:
        ids = self.table[query][:top_k]
        return [Hit(c, c.split("#")[0], 1.0 - i / 10, f"text of {c}") for i, c in enumerate(ids)]


class ScriptedGenerator:
    name = "scripted/v1"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        abstain = not hits
        text = (
            "Not answerable from the documentation. Nothing."
            if abstain
            else f"A [{hits[0].chunk_id}]"
        )
        return Answer(
            text, (hits[0].chunk_id,) if hits else (), abstain, False, "m", "v1", 10, 5, 1.0
        )


ENTRIES = [
    GoldenEntry("q1", "lock?", "a", ("sql-lock#c0",), "single-hop"),
    GoldenEntry("q2", "wal?", "a", ("wal#c1", "wal#c2"), "multi-hop"),
    GoldenEntry("q3", "moon?", "no", (), "unanswerable"),
]
TABLE = {
    "lock?": ["sql-lock#c0", "x#c0", "y#c0"],  # hit at rank 1
    "wal?": ["x#c0", "wal#c2", "z#c0", "wal#c1"],  # one of two inside k=3, first at rank 2
    "moon?": [],
}


def test_run_scores_each_query_and_aggregates() -> None:
    res = run(CFG, ENTRIES, ScriptedRetriever(TABLE))
    by_id = {q.id: q for q in res.queries}
    assert by_id["q1"].metrics.recall == 1.0 and by_id["q1"].metrics.mrr == 1.0
    assert by_id["q2"].metrics.recall == 0.5 and by_id["q2"].metrics.mrr == 0.5
    assert by_id["q2"].retrieved == ["x#c0", "wal#c2", "z#c0"]  # cut at k
    assert by_id["q3"].metrics.recall is None
    agg = res.retrieval_scores
    assert (agg.recall, agg.mrr, agg.scored, agg.skipped) == (0.75, 0.75, 2, 1)
    assert res.generator is None and res.abstain_accuracy() is None
    assert res.retriever == "scripted"


def test_generate_records_answers_and_abstain_accuracy() -> None:
    res = run(CFG, ENTRIES, ScriptedRetriever(TABLE), ScriptedGenerator())
    answers = {q.id: q.answer for q in res.queries}
    assert answers["q1"] is not None and answers["q1"].cited_chunk_ids == ("sql-lock#c0",)
    assert answers["q3"] is not None and answers["q3"].abstained
    assert res.abstain_accuracy() == 1.0  # abstained exactly on the unanswerable one
    assert "abstain accuracy 1.00" in summary(res)


def test_results_row_and_run_file(tmp_path: Path) -> None:
    res = run(CFG, ENTRIES, ScriptedRetriever(TABLE))
    run_file = write_run_file(res, tmp_path / "runs")
    payload = json.loads(run_file.read_text())
    assert payload["retrieval"]["recall"] == 0.75
    assert [q["id"] for q in payload["queries"]] == ["q1", "q2", "q3"]
    assert payload["queries"][2]["metrics"]["recall"] is None

    md = tmp_path / "results.md"
    append_results(res, run_file, md)
    append_results(res, run_file, md)
    text = md.read_text()
    assert text.startswith(RESULTS_HEADER)
    rows = [ln for ln in text.splitlines() if ln.startswith("| 20")]
    assert len(rows) == 2  # header written once, one row per run
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    assert cells[1:7] == ["t", "fixed512", "text-embedding-3-small", "dense", "3", "2/1"]
    assert cells[7:10] == ["0.750", "0.750", cells[9]] and cells[9] != "—"
    assert cells[10] == "—"  # faithfulness not measured
    assert cells[13] == "—"  # $/query not measured
    assert run_file.name in cells[15]


def test_row_has_one_cell_per_header_column() -> None:
    res = run(CFG, ENTRIES, ScriptedRetriever(TABLE))
    header_cols = RESULTS_HEADER.splitlines()[-2].count("|") - 1
    assert results_row(res, None).count("|") - 1 == header_cols


def test_empty_golden_set_is_an_error() -> None:
    res = run(CFG, [], ScriptedRetriever(TABLE))
    with pytest.raises(ValueError):
        _ = res.retrieval_scores


def test_cli_reports_missing_golden_and_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals import run as run_cli

    cfg = "evals/configs/smoke-hash64.yaml"
    assert run_cli.main([cfg, "--golden", str(tmp_path / "none.jsonl"), "--dry-run"]) == 1
    assert "does not exist" in capsys.readouterr().err

    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"id": "q1", "question": "x", "answer": "y", "relevant_chunk_ids": ["a#c0"], '
        '"difficulty": "single-hop"}\n'
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nope")
    assert run_cli.main([cfg, "--golden", str(golden), "--dry-run"]) == 1
    assert "error:" in capsys.readouterr().err
