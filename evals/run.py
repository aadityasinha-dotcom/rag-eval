"""Run one config over the golden set, print metrics, append a row to docs/results.md.

    python -m evals.run evals/configs/baseline.yaml [--generate] [--faithfulness] \
        [--dry-run] [--limit N]

Retrieval metrics (recall@k, MRR, nDCG@k) need Postgres with the config
indexed (make index CONFIG=...). --generate also answers every question with
the config's generation model and records answers, citations, abstentions and
token usage in the run file. --faithfulness additionally scores each real
answer with Ragas' Faithfulness using the config's judge model; abstentions and
refusals are not judged. Columns for anything not measured stay blank
(CLAUDE.md rule 5: no number that was not measured).

Every run writes evals/runs/<config>-<utc>.json with per-query detail; the
results.md row links to it. --dry-run skips the row and the file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

from evals import instrument
from evals.config import RunConfig, load_config
from evals.golden import DEFAULT_GOLDEN, GoldenEntry, load_golden
from evals.metrics.generation import FaithfulnessJudge, RagasFaithfulness
from evals.metrics.retrieval import AggregateScores, RetrievalScores, aggregate, score_query
from generation.answer import Answer, ClaudeGenerator, Generator
from generation.prompts import PROMPTS
from ingest import db
from retrieval.base import Retriever
from retrieval.factory import build_retriever

ROOT = Path(__file__).resolve().parent.parent
RESULTS_MD = ROOT / "docs" / "results.md"
RUNS_DIR = ROOT / "evals" / "runs"
BLANK = "—"  # a metric that was not measured in this run

RESULTS_HEADER = (
    "# Results\n\n"
    "Appended by `evals/run.py` only. Never edit rows by hand (CLAUDE.md rule 5). "
    f"`{BLANK}` means the metric was not measured in that run.\n\n"
    "| date (UTC) | config | chunking | embedding | retrieval | k | queries (scored/skipped) "
    "| recall@k | MRR | nDCG@k | faithfulness | abstain acc | retrieval p50/p95 ms | $/query "
    "| commit | run |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


@dataclass
class QueryResult:
    id: str
    difficulty: str
    relevant: list[str]
    retrieved: list[str]
    scores: list[float]
    retrieval_ms: float
    metrics: RetrievalScores
    answer: Answer | None = None
    faithfulness: float | None = None
    trace_id: str | None = None  # llm-metrics trace, when instrumentation is on


@dataclass
class RunResult:
    config: RunConfig
    started_utc: str
    commit: str | None
    retriever: str
    generator: str | None
    judge: str | None = None
    instrumented: bool = False
    queries: list[QueryResult] = field(default_factory=list)

    @property
    def retrieval_scores(self) -> AggregateScores:
        return aggregate([q.metrics for q in self.queries])

    def latency_percentiles(self) -> tuple[float, float]:
        xs = sorted(q.retrieval_ms for q in self.queries)
        if not xs:
            return (0.0, 0.0)
        p50 = statistics.median(xs)
        p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
        return (p50, p95)

    def faithfulness(self) -> tuple[float | None, int]:
        """(mean over judged answers, number judged). Abstentions/refusals are not judged."""
        xs = [q.faithfulness for q in self.queries if q.faithfulness is not None]
        return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)

    def abstain_accuracy(self) -> float | None:
        """Fraction of answered queries where abstaining matched answerability.

        Unanswerable entries should abstain; answerable ones should not. This
        is a cheap, judge-free signal about refusal behaviour, not faithfulness.
        """
        answered = [q for q in self.queries if q.answer is not None and not q.answer.refused]
        if not answered:
            return None
        correct = sum(
            1
            for q in answered
            if q.answer is not None and q.answer.abstained == (q.difficulty == "unanswerable")
        )
        return correct / len(answered)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def run(
    config: RunConfig,
    entries: list[GoldenEntry],
    retriever: Retriever,
    generator: Generator | None = None,
    log: Callable[[str], None] | None = None,
    judge: FaithfulnessJudge | None = None,
) -> RunResult:
    if judge is not None and generator is None:
        raise ValueError("a faithfulness judge needs a generator")
    result = RunResult(
        config=config,
        started_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        commit=git_commit(),
        retriever=retriever.name,
        generator=generator.name if generator else None,
        judge=judge.name if judge else None,
        instrumented=instrument.setup(release=git_commit()),
    )

    def one_query(e: GoldenEntry) -> QueryResult:
        instrument.annotate_trace(
            session_id=f"{config.name}@{result.started_utc}",
            tags=[config.name, e.difficulty],
            golden_id=e.id,
            config=asdict(config),
        )
        t0 = time.perf_counter()
        hits = instrument.traced(
            "retrieve",
            "retrieval",
            retriever.search,
            e.question,
            config.top_k,
            metadata={"retriever": retriever.name, "top_k": config.top_k},
        )
        ms = (time.perf_counter() - t0) * 1000
        retrieved = [h.chunk_id for h in hits]
        q = QueryResult(
            id=e.id,
            difficulty=e.difficulty,
            relevant=list(e.relevant_chunk_ids),
            retrieved=retrieved,
            scores=[round(h.score, 6) for h in hits],
            retrieval_ms=round(ms, 2),
            metrics=score_query(retrieved, e.relevant_chunk_ids, config.top_k),
            trace_id=instrument.current_trace_id(),
        )
        for metric, value in (
            (f"recall@{config.top_k}", q.metrics.recall),
            ("mrr", q.metrics.mrr),
            (f"ndcg@{config.top_k}", q.metrics.ndcg),
        ):
            if value is not None:
                instrument.record_score(metric, value, source="heuristic")
        if generator is not None:
            q.answer = generator.answer(e.question, hits)
            instrument.record_score("abstained", q.answer.abstained, source="heuristic")
            if judge is not None and not (q.answer.abstained or q.answer.refused) and hits:
                q.faithfulness = judge.score(e.question, q.answer.text, [h.text for h in hits])
                if q.faithfulness is not None:
                    instrument.record_score("faithfulness", q.faithfulness, source="llm_judge")
        return q

    for i, e in enumerate(entries, start=1):
        q = instrument.traced(
            "rag-eval.query",
            "span",
            one_query,
            e,
            metadata={"golden_id": e.id, "difficulty": e.difficulty, "config": config.name},
        )
        result.queries.append(q)
        if log:
            r = q.metrics.recall
            log(f"[{i}/{len(entries)}] {e.id} recall={'n/a' if r is None else f'{r:.2f}'}")
    instrument.flush()
    return result


def _fmt(x: float | None, digits: int = 3) -> str:
    return BLANK if x is None else f"{x:.{digits}f}"


def _fmt_faith(res: RunResult) -> str:
    mean, n = res.faithfulness()
    return BLANK if mean is None else f"{mean:.3f} (n={n})"


def results_row(res: RunResult, run_file: Path | None, results_md: Path = RESULTS_MD) -> str:
    agg = res.retrieval_scores
    p50, p95 = res.latency_percentiles()
    cfg = res.config
    run_link = BLANK
    if run_file:
        rel = os.path.relpath(run_file, results_md.parent)
        run_link = f"[{run_file.name}]({Path(rel).as_posix()})"
    return (
        f"| {res.started_utc[:10]} | {cfg.name} | {cfg.chunking} | {cfg.embedding} | "
        f"{cfg.retrieval}{' + ' + cfg.rerank if cfg.rerank else ''} | {cfg.top_k} | "
        f"{agg.scored}/{agg.skipped} | {_fmt(agg.recall)} | {_fmt(agg.mrr)} | {_fmt(agg.ndcg)} | "
        f"{_fmt_faith(res)} | {_fmt(res.abstain_accuracy(), 2)} | {p50:.0f}/{p95:.0f} | {BLANK} | "
        f"{res.commit or BLANK} | {run_link} |\n"
    )


def append_results(res: RunResult, run_file: Path | None, results_md: Path = RESULTS_MD) -> None:
    results_md.parent.mkdir(parents=True, exist_ok=True)
    if not results_md.exists():
        results_md.write_text(RESULTS_HEADER, encoding="utf-8")
    with results_md.open("a", encoding="utf-8") as f:
        f.write(results_row(res, run_file, results_md))


def write_run_file(res: RunResult, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = res.started_utc.replace(":", "").replace("-", "")
    path = runs_dir / f"{res.config.name}-{stamp}.json"
    payload: dict[str, Any] = {
        "config": asdict(res.config),
        "started_utc": res.started_utc,
        "commit": res.commit,
        "retriever": res.retriever,
        "generator": res.generator,
        "judge": res.judge,
        "instrumented": res.instrumented,
        "retrieval": asdict(res.retrieval_scores),
        "faithfulness_mean_n": res.faithfulness(),
        "retrieval_ms_p50_p95": res.latency_percentiles(),
        "abstain_accuracy": res.abstain_accuracy(),
        "queries": [asdict(q) for q in res.queries],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def summary(res: RunResult) -> str:
    agg = res.retrieval_scores
    p50, p95 = res.latency_percentiles()
    lines = [
        f"{res.config.name}: {agg.scored} scored, {agg.skipped} unanswerable skipped "
        f"(k={res.config.top_k}, {res.retriever})",
        f"  recall@k {agg.recall:.3f}   MRR {agg.mrr:.3f}   nDCG@k {agg.ndcg:.3f}",
        f"  retrieval latency p50 {p50:.0f} ms, p95 {p95:.0f} ms",
    ]
    if res.instrumented:
        lines.append("  traces sent to llm-metrics (trace ids in the run file)")
    if res.generator:
        answered = [q.answer for q in res.queries if q.answer]
        refused = sum(1 for a in answered if a.refused)
        toks_in = sum(a.input_tokens for a in answered)
        toks_out = sum(a.output_tokens for a in answered)
        acc = res.abstain_accuracy()
        lines.append(
            f"  generation {res.generator}: {len(answered)} answers, {refused} refusals, "
            f"abstain accuracy {_fmt(acc, 2)}, tokens in/out {toks_in}/{toks_out}"
        )
    if res.judge:
        mean, n = res.faithfulness()
        lines.append(f"  faithfulness {res.judge}: {_fmt(mean)} over {n} judged answers")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--generate", action="store_true", help="also answer with the LLM")
    ap.add_argument(
        "--faithfulness", action="store_true", help="judge answers with Ragas (implies --generate)"
    )
    ap.add_argument("--dry-run", action="store_true", help="print metrics, write nothing")
    ap.add_argument("--limit", type=int, default=None, help="first N golden entries only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        config = load_config(args.config)
        if not args.golden.is_file():
            raise FileNotFoundError(f"{args.golden} does not exist: write the golden set first")
        entries = load_golden(args.golden)
        if args.limit:
            entries = entries[: args.limit]
        if not entries:
            raise ValueError(f"{args.golden} has no entries")
        generate = args.generate or args.faithfulness
        generator = ClaudeGenerator(config.generation, PROMPTS[config.prompt]) if generate else None
        judge = RagasFaithfulness(config.judge) if args.faithfulness else None
        with db.connect() as conn:
            retriever = build_retriever(config, conn, instrumented=instrument.enabled())
            res = run(
                config,
                entries,
                retriever,
                generator,
                log=None if args.quiet else lambda s: print(s, file=sys.stderr),
                judge=judge,
            )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        NotImplementedError,
        psycopg.OperationalError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(summary(res))
    if args.dry_run or args.limit:
        print("(dry run or --limit: no results row written)")
        return 0
    run_file = write_run_file(res)
    append_results(res, run_file)
    print(f"appended to {RESULTS_MD.relative_to(ROOT)}; detail in {run_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
