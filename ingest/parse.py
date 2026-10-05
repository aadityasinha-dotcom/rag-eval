"""Load raw corpus files into one normalized text per file.

LlamaIndex is used here for *loading only* (CLAUDE.md rule 6). Some LlamaIndex
readers split a single file into several Documents (MarkdownReader splits on
headers, PDF readers split on pages). We re-join those so that one file always
becomes exactly one ParsedDoc, because chunk ids are `<doc>#c<n>` and `<doc>`
identifies a file.

The doc id is the file path relative to the corpus root with its extension
removed, as a POSIX path: corpus/sql-createtable.html -> "sql-createtable".

HTML goes through DocBookHtmlReader below rather than LlamaIndex's HTMLTagReader:
the PostgreSQL pages are DocBook output (`<div class="sect1">`), contain no
`<section>` tags, and HTMLTagReader would therefore return no text at all.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# llama-index-core 0.13 triggers a harmless pydantic warning on import; keep CLI output clean.
warnings.filterwarnings("ignore", message=".*validate_default.*")
from bs4 import BeautifulSoup  # noqa: E402
from llama_index.core import SimpleDirectoryReader  # noqa: E402
from llama_index.core.readers.base import BaseReader  # noqa: E402
from llama_index.core.schema import Document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "corpus"
CACHE_DIR = ROOT / ".cache"
SUPPORTED_EXTS: tuple[str, ...] = (".txt", ".md", ".html", ".htm", ".pdf")

# Chrome around every DocBook page: "Prev | Up | Next" tables. Not content.
_DOCBOOK_NAV_CLASSES = ("navheader", "navfooter")
# Elements that start a new line in the extracted text. Inline markup such as
# <code>, <em>, <a> is flattened so that sentences stay intact.
_BLOCK_TAGS = [
    *("h1", "h2", "h3", "h4", "h5", "h6"),
    *("p", "div", "pre", "blockquote", "br", "hr"),
    *("ul", "ol", "li", "dl", "dt", "dd"),
    *("table", "thead", "tbody", "tr", "th", "td", "caption"),
]
_BLANK_RUN = re.compile(r"\n{3,}")


class DocBookHtmlReader(BaseReader):
    """One Document per HTML file: body text with DocBook navigation removed.

    Text extraction is deterministic given the file and the pinned bs4 version:
    block elements start a new line, inline elements are flattened into the
    surrounding sentence, each line is stripped of surrounding whitespace (this
    drops indentation inside <pre> blocks, accepted for the baseline), and runs
    of blank lines collapse to one.
    """

    def load_data(self, file: Path, extra_info: dict[str, Any] | None = None) -> list[Document]:
        html = Path(file).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for cls in _DOCBOOK_NAV_CLASSES:
            for node in soup.find_all(class_=cls):
                node.decompose()
        for node in soup.find_all("a", class_="id_link"):  # the "#" permalink glyph on headings
            node.decompose()
        for node in soup.find_all(["script", "style"]):
            node.decompose()
        root = soup.body or soup
        for node in root.find_all(_BLOCK_TAGS):
            node.insert_before("\n")
            node.insert_after("\n")
        text = root.get_text("").replace("\xa0", " ")
        lines = [ln.strip() for ln in text.splitlines()]
        text = _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()
        metadata: dict[str, Any] = {"file_path": str(file)}
        metadata.update(extra_info or {})
        return [Document(text=text, metadata=metadata)]


FILE_EXTRACTOR: dict[str, BaseReader] = {
    ".html": DocBookHtmlReader(),
    ".htm": DocBookHtmlReader(),
}


@dataclass(frozen=True)
class ParsedDoc:
    doc_id: str
    path: Path
    text: str


def doc_id_for(path: Path, corpus_root: Path) -> str:
    """Stable id for a corpus file, derived only from its path under the root."""
    rel = path.resolve().relative_to(corpus_root.resolve())
    doc_id = rel.with_suffix("").as_posix()
    if "#" in doc_id:
        raise ValueError(f"corpus file name must not contain '#': {rel}")
    return doc_id


def path_for(doc_id: str, corpus_root: Path) -> Path:
    """Inverse of doc_id_for. Raises FileNotFoundError if no supported file matches."""
    candidates = [corpus_root / f"{doc_id}{ext}" for ext in SUPPORTED_EXTS]
    hits = [p for p in candidates if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"no corpus file for doc id {doc_id!r} under {corpus_root}")
    if len(hits) > 1:
        raise ValueError(f"ambiguous doc id {doc_id!r}: {[p.name for p in hits]}")
    return hits[0]


def normalize_text(text: str) -> str:
    """Make loader output byte-for-byte reproducible across platforms."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _group_by_file(documents: list) -> dict[str, list[str]]:  # type: ignore[type-arg]
    by_path: dict[str, list[str]] = {}
    for d in documents:
        by_path.setdefault(d.metadata["file_path"], []).append(d.text)
    return by_path


def _to_parsed(by_path: dict[str, list[str]], corpus_root: Path) -> list[ParsedDoc]:
    out = [
        ParsedDoc(
            doc_id=doc_id_for(Path(fp), corpus_root),
            path=Path(fp).resolve(),
            text=normalize_text("\n\n".join(parts)),
        )
        for fp, parts in by_path.items()
    ]
    out.sort(key=lambda d: d.doc_id)
    return out


def _corpus_files(corpus_root: Path) -> list[Path]:
    return sorted(
        p
        for p in corpus_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and not any(part.startswith(".") for part in p.relative_to(corpus_root).parts)
    )


def _cache_key(corpus_root: Path, files: list[Path]) -> str:
    """Changes when any corpus file, or this parser module, changes."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(str(corpus_root).encode())
    for p in files:
        st = p.stat()
        h.update(
            f"{p.relative_to(corpus_root).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode()
        )
    return h.hexdigest()[:16]


def _parse_corpus(corpus_root: Path) -> list[ParsedDoc]:
    reader = SimpleDirectoryReader(
        input_dir=str(corpus_root),
        recursive=True,
        required_exts=list(SUPPORTED_EXTS),
        exclude_hidden=True,
        file_extractor=FILE_EXTRACTOR,
    )
    return _to_parsed(_group_by_file(reader.load_data()), corpus_root)


def load_corpus(corpus_root: Path = DEFAULT_CORPUS, cache: bool = True) -> list[ParsedDoc]:
    """Every supported file under corpus_root, one ParsedDoc each, sorted by doc id.

    Parsing 1k+ HTML pages takes ~15s, so the result is cached under .cache/
    keyed by file sizes/mtimes and the parser source; it never serves stale text.
    """
    corpus_root = corpus_root.resolve()
    files = _corpus_files(corpus_root)
    if not files:
        return []
    cache_file = CACHE_DIR / f"parsed-{_cache_key(corpus_root, files)}.json"
    if cache and cache_file.is_file():
        rows = json.loads(cache_file.read_text(encoding="utf-8"))
        return [ParsedDoc(r["doc_id"], Path(r["path"]), r["text"]) for r in rows]
    docs = _parse_corpus(corpus_root)
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        rows = [{**asdict(d), "path": str(d.path)} for d in docs]
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(cache_file)
    return docs


def load_doc(doc_id: str, corpus_root: Path = DEFAULT_CORPUS) -> ParsedDoc:
    """Load a single document by id without reading the rest of the corpus."""
    corpus_root = corpus_root.resolve()
    path = path_for(doc_id, corpus_root)
    reader = SimpleDirectoryReader(input_files=[str(path)], file_extractor=FILE_EXTRACTOR)
    docs = _to_parsed(_group_by_file(reader.load_data()), corpus_root)
    assert len(docs) == 1, docs
    return docs[0]
