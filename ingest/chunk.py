"""Chunking strategies with deterministic ids.

Each strategy maps one ParsedDoc to a list of Chunks whose ids have the form
`<doc>#c<n>`. Chunking never crosses a document boundary: the counter `n`
restarts at 0 for every document, and a chunk's id and text depend only on
that document's text and the strategy parameters. Adding, removing or editing
*other* files cannot change a document's chunks.

Ids are positional. If a document is edited, chunks before the edit keep both
id and text; chunks at or after the edit keep their id but may hold different
text, and the document may gain or lose trailing chunks. Golden entries that
reference those ids must then be re-labelled (CLAUDE.md rule 7). `text_hash`
exists so that drift can be detected rather than silently accepted.

CLI:
    python -m ingest.chunk --strategy fixed512 --show doc3
    python -m ingest.chunk --list
"""

from __future__ import annotations

import argparse
import hashlib
import signal
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ingest.parse import DEFAULT_CORPUS, ParsedDoc, load_corpus, load_doc


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    n: int
    start: int  # char offset into the normalized document text, inclusive
    end: int  # exclusive
    text: str

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, n: int) -> str:
    return f"{doc_id}#c{n}"


class ChunkStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...


@dataclass(frozen=True)
class FixedSizeStrategy:
    """Fixed character windows. Window n covers [n*(size-overlap), that + size)."""

    name: str
    size: int
    overlap: int

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")

    @property
    def stride(self) -> int:
        return self.size - self.overlap

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        text = doc.text
        length = len(text)
        chunks: list[Chunk] = []
        for n, start in enumerate(range(0, length, self.stride)):
            end = min(start + self.size, length)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, n),
                    doc_id=doc.doc_id,
                    n=n,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
            if end == length:
                break  # the remaining tail is already covered; no extra overlap-only chunk
        return chunks


STRATEGIES: dict[str, ChunkStrategy] = {
    "fixed512": FixedSizeStrategy(name="fixed512", size=512, overlap=64),
}


def chunk_corpus(docs: Iterable[ParsedDoc], strategy: ChunkStrategy) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(strategy.chunk(doc))
    return out


# ---------------------------------------------------------------- CLI


def format_chunk(chunk: Chunk, ids_only: bool = False) -> str:
    header = f"== {chunk.chunk_id}  [{chunk.start}:{chunk.end})  sha256:{chunk.text_hash[:12]}"
    if ids_only:
        return header
    return f"{header}\n{chunk.text}\n"


def show_doc(doc_id: str, strategy: ChunkStrategy, corpus: Path, ids_only: bool) -> int:
    doc = load_doc(doc_id, corpus)
    chunks = strategy.chunk(doc)
    print(
        f"# {doc.doc_id}: {len(doc.text)} chars -> {len(chunks)} chunks "
        f"(strategy={strategy.name}, file={doc.path.relative_to(corpus.resolve())})"
    )
    for c in chunks:
        print(format_chunk(c, ids_only))
    return 0


def list_docs(strategy: ChunkStrategy, corpus: Path) -> int:
    docs = load_corpus(corpus)
    if not docs:
        print(f"no documents under {corpus}", file=sys.stderr)
        return 1
    for doc in docs:
        print(f"{doc.doc_id}\t{len(doc.text)} chars\t{len(strategy.chunk(doc))} chunks")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ingest.chunk", description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), default="fixed512")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus root directory")
    ap.add_argument("--ids-only", action="store_true", help="print chunk headers without text")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--show", metavar="DOC", help="print one document's chunks with ids")
    mode.add_argument("--list", action="store_true", help="list doc ids with chunk counts")
    args = ap.parse_args(argv)

    strategy = STRATEGIES[args.strategy]
    try:
        if args.list:
            return list_docs(strategy, args.corpus)
        return show_doc(args.show, strategy, args.corpus, args.ids_only)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Let `python -m ingest.chunk --show x | head` exit quietly instead of tracing BrokenPipeError.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
