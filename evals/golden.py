"""The hand-labelled golden set: loading, validation, and review.

evals/golden.jsonl is written by hand (CLAUDE.md rule 2). This module never
writes it; it checks that what was written is usable:

  python -m evals.golden --check              # validate shape + every chunk id exists
  python -m evals.golden --show q001          # question, answer, and the labelled chunks' text
  python -m evals.golden --stats              # counts by difficulty, docs covered
  python -m evals.golden --freeze [--force]   # record cited chunks' text hashes in golden.lock.json

Drift (CLAUDE.md rule 7): ids are positional, so an edited corpus or a changed
chunker can leave an id pointing at different text. --freeze records
sha256(text) for every cited chunk in evals/golden.lock.json (next to the
golden file); --check then fails on any cited chunk whose text no longer
matches. After re-labelling, --freeze --force accepts the new text.

Entry format (one JSON object per line):
  {"id": "q001", "question": "...", "answer": "...",
   "relevant_chunk_ids": ["sql-lock#c0"], "difficulty": "single-hop"}

difficulty: single-hop (one chunk answers it), multi-hop (needs several
chunks, usually from different pages), unanswerable (the corpus cannot answer
it; relevant_chunk_ids must be empty and the answer states the refusal).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ingest.chunk import STRATEGIES, Chunk, ChunkStrategy, chunk_corpus
from ingest.parse import DEFAULT_CORPUS, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "evals" / "golden.jsonl"
DIFFICULTIES = ("single-hop", "multi-hop", "unanswerable")
REQUIRED_FIELDS = ("id", "question", "answer", "relevant_chunk_ids", "difficulty")
TARGET_MIN = 50


@dataclass(frozen=True)
class GoldenEntry:
    id: str
    question: str
    answer: str
    relevant_chunk_ids: tuple[str, ...]
    difficulty: str


def load_golden(path: Path = DEFAULT_GOLDEN) -> list[GoldenEntry]:
    """Parse the file. Raises ValueError with line numbers for malformed input."""
    entries: list[GoldenEntry] = []
    errors: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: invalid JSON ({e.msg})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {lineno}: expected an object")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        extra = sorted(set(obj) - set(REQUIRED_FIELDS))
        if missing or extra:
            errors.append(f"line {lineno}: missing {missing}, unexpected {extra}")
            continue
        ids = obj["relevant_chunk_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            errors.append(f"line {lineno}: relevant_chunk_ids must be a list of strings")
            continue
        if not all(isinstance(obj[f], str) for f in ("id", "question", "answer", "difficulty")):
            errors.append(f"line {lineno}: id, question, answer, difficulty must be strings")
            continue
        entries.append(
            GoldenEntry(obj["id"], obj["question"], obj["answer"], tuple(ids), obj["difficulty"])
        )
    if errors:
        raise ValueError("\n".join(errors))
    return entries


def validate(entries: list[GoldenEntry], chunk_ids: set[str]) -> list[str]:
    """Semantic checks against the current chunking. Returns problems (empty = ok)."""
    problems: list[str] = []
    seen_ids: Counter[str] = Counter(e.id for e in entries)
    seen_q: Counter[str] = Counter(" ".join(e.question.lower().split()) for e in entries)
    for e in entries:
        where = f"{e.id}:"
        if seen_ids[e.id] > 1:
            problems.append(f"{where} duplicate id")
        if seen_q[" ".join(e.question.lower().split())] > 1:
            problems.append(f"{where} duplicate question")
        if not e.question.strip():
            problems.append(f"{where} empty question")
        if not e.answer.strip():
            problems.append(f"{where} empty answer")
        if e.difficulty not in DIFFICULTIES:
            problems.append(f"{where} difficulty {e.difficulty!r} not in {DIFFICULTIES}")
        if e.difficulty == "unanswerable":
            if e.relevant_chunk_ids:
                problems.append(f"{where} unanswerable entries must have no relevant_chunk_ids")
        elif not e.relevant_chunk_ids:
            problems.append(f"{where} no relevant_chunk_ids")
        if len(set(e.relevant_chunk_ids)) != len(e.relevant_chunk_ids):
            problems.append(f"{where} repeated chunk id")
        for cid in e.relevant_chunk_ids:
            if cid not in chunk_ids:
                problems.append(f"{where} unknown chunk id {cid!r}")
    return problems


def _all_chunks(strategy: ChunkStrategy, corpus: Path) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunk_corpus(load_corpus(corpus), strategy)}


@dataclass(frozen=True)
class Lock:
    """Text hashes of every cited chunk at the time the labels were last accepted."""

    strategy: str
    chunks: dict[str, str]  # chunk_id -> sha256(text)


def lock_path_for(golden: Path) -> Path:
    return golden.with_suffix(".lock.json")  # golden.jsonl -> golden.lock.json


def load_lock(path: Path) -> Lock | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Lock(strategy=data["strategy"], chunks=dict(data["chunks"]))


def save_lock(path: Path, lock: Lock) -> None:
    payload = {"strategy": lock.strategy, "chunks": dict(sorted(lock.chunks.items()))}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def cited_ids(entries: list[GoldenEntry]) -> list[str]:
    return sorted({cid for e in entries for cid in e.relevant_chunk_ids})


def drift(
    entries: list[GoldenEntry], chunks: dict[str, Chunk], lock: Lock, strategy: str
) -> tuple[list[str], list[str]]:
    """(problems, unfrozen ids). A problem is a cited chunk whose text changed
    since --freeze, or a lock recorded under a different strategy."""
    if lock.strategy != strategy:
        return (
            [f"lock was frozen under strategy {lock.strategy!r}, checking {strategy!r}: re-label"],
            [],
        )
    problems: list[str] = []
    unfrozen: list[str] = []
    for e in entries:
        for cid in e.relevant_chunk_ids:
            c = chunks.get(cid)
            if c is None:
                continue  # reported by validate()
            frozen = lock.chunks.get(cid)
            if frozen is None:
                unfrozen.append(cid)
            elif frozen != c.text_hash:
                problems.append(
                    f"{e.id}: {cid} text changed since it was labelled "
                    f"(frozen {frozen[:12]}, now {c.text_hash[:12]}); "
                    "re-label, then --freeze --force"
                )
    return problems, sorted(set(unfrozen))


def cmd_check(golden: Path, strategy: ChunkStrategy, corpus: Path) -> int:
    if not golden.is_file():
        print(f"{golden} does not exist yet", file=sys.stderr)
        return 1
    try:
        entries = load_golden(golden)
    except ValueError as e:
        print(f"{golden}: malformed\n{e}", file=sys.stderr)
        return 1
    chunks = _all_chunks(strategy, corpus)
    problems = validate(entries, set(chunks))
    lock = load_lock(lock_path_for(golden))
    if lock is None:
        if cited_ids(entries):
            print("notice: no lock file; run --freeze to record chunk text hashes", file=sys.stderr)
    else:
        drifted, unfrozen = drift(entries, chunks, lock, strategy.name)
        problems.extend(drifted)
        if unfrozen:
            print(
                f"notice: {len(unfrozen)} cited chunk(s) not in the lock file; "
                "run --freeze after labelling",
                file=sys.stderr,
            )
    for p in problems:
        print(p, file=sys.stderr)
    n = len(entries)
    status = "FAIL" if problems else "ok"
    target = "" if n >= TARGET_MIN else f" (target >= {TARGET_MIN})"
    print(f"{status}: {n} entries{target}, {len(problems)} problems, strategy={strategy.name}")
    return 1 if problems else 0


def cmd_freeze(golden: Path, strategy: ChunkStrategy, corpus: Path, force: bool) -> int:
    """Record current text hashes for every cited chunk. Refuses to overwrite a
    changed hash unless --force, so drift is accepted explicitly, never by habit."""
    entries = load_golden(golden)
    chunks = _all_chunks(strategy, corpus)
    problems = validate(entries, set(chunks))
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print("refusing to freeze a golden set with problems", file=sys.stderr)
        return 1
    path = lock_path_for(golden)
    old = load_lock(path)
    new = Lock(strategy.name, {cid: chunks[cid].text_hash for cid in cited_ids(entries)})
    if old is None:
        added, updated, removed = set(new.chunks), set(), set()
    else:
        added = set(new.chunks) - set(old.chunks)
        removed = set(old.chunks) - set(new.chunks)
        updated = {
            cid for cid in set(new.chunks) & set(old.chunks) if old.chunks[cid] != new.chunks[cid]
        }
        if old.strategy != strategy.name:
            updated |= set(new.chunks) & set(old.chunks)
    if updated and not force:
        for cid in sorted(updated):
            print(f"{cid}: text changed since last freeze", file=sys.stderr)
        print(
            f"refusing to overwrite {len(updated)} changed hash(es); re-label those entries, "
            "then run --freeze --force",
            file=sys.stderr,
        )
        return 1
    save_lock(path, new)
    print(
        f"froze {len(new.chunks)} chunk hashes to {path.name} "
        f"({len(added)} added, {len(updated)} updated, {len(removed)} removed, "
        f"strategy={strategy.name})"
    )
    return 0


def cmd_show(golden: Path, qid: str, strategy: ChunkStrategy, corpus: Path) -> int:
    entries = {e.id: e for e in load_golden(golden)}
    if qid not in entries:
        print(f"no entry {qid!r} in {golden}", file=sys.stderr)
        return 1
    e = entries[qid]
    chunks = _all_chunks(strategy, corpus)
    lock = load_lock(lock_path_for(golden))
    print(f"# {e.id} [{e.difficulty}]\nQ: {e.question}\nA: {e.answer}\n")
    for cid in e.relevant_chunk_ids:
        c = chunks.get(cid)
        if c is None:
            print(f"== {cid}  (MISSING under {strategy.name})\n")
            continue
        mark = ""
        if lock is not None and lock.strategy == strategy.name:
            frozen = lock.chunks.get(cid)
            if frozen is None:
                mark = "  (not frozen)"
            elif frozen != c.text_hash:
                mark = "  (DRIFTED since freeze)"
        print(f"== {cid}  [{c.start}:{c.end})  sha256:{c.text_hash[:12]}{mark}\n{c.text}\n")
    return 0


def cmd_stats(golden: Path) -> int:
    entries = load_golden(golden)
    by_diff = Counter(e.difficulty for e in entries)
    docs = Counter(cid.split("#", 1)[0] for e in entries for cid in e.relevant_chunk_ids)
    n_chunks = sum(len(e.relevant_chunk_ids) for e in entries)
    print(f"{len(entries)} entries, {n_chunks} chunk refs, {len(docs)} docs covered")
    for d in DIFFICULTIES:
        print(f"  {d:13s} {by_diff.get(d, 0)}")
    print("top docs:")
    for doc, n in docs.most_common(10):
        print(f"  {n:3d}  {doc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals.golden", description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="fixed512")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--show", metavar="QID")
    mode.add_argument("--stats", action="store_true")
    mode.add_argument("--freeze", action="store_true", help="record cited chunk hashes")
    ap.add_argument("--force", action="store_true", help="with --freeze: accept changed text")
    args = ap.parse_args(argv)
    strategy = STRATEGIES[args.strategy]
    try:
        if args.check:
            return cmd_check(args.golden, strategy, args.corpus)
        if args.freeze:
            return cmd_freeze(args.golden, strategy, args.corpus, args.force)
        if args.stats:
            return cmd_stats(args.golden)
        return cmd_show(args.golden, args.show, strategy, args.corpus)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
