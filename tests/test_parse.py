"""parse.py: one ParsedDoc per file, ids from paths, reproducible text. Also the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest import chunk as chunk_cli
from ingest.parse import doc_id_for, load_corpus, load_doc, path_for

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


def test_one_doc_per_file_with_path_derived_ids() -> None:
    docs = load_corpus(FIXTURES)
    assert [d.doc_id for d in docs] == ["doc1", "doc2", "doc3", "guides/nested", "sql-lock"]
    assert all(d.text == d.text.strip() and "\r" not in d.text for d in docs)


def test_markdown_with_several_headers_is_still_one_doc() -> None:
    doc2 = [d for d in load_corpus(FIXTURES) if d.doc_id == "doc2"][0]
    assert "Creating a schema" in doc2.text
    assert "public schema" in doc2.text


def test_docbook_html_keeps_body_text_and_drops_navigation() -> None:
    doc = load_doc("sql-lock", FIXTURES)
    assert doc.text.startswith("LOCK\n")
    assert "LOCK — lock a table" in doc.text
    assert "obtains a table-level lock" in doc.text
    assert "LOCK [ TABLE ] [ ONLY ] name [ * ]" in doc.text  # inline markup flattened
    for nav in ("Prev", "Next", "Home", "Navigation header", "LISTEN", "MERGE"):
        assert nav not in doc.text, nav
    assert "\n\n\n" not in doc.text and "<" not in doc.text


def test_docbook_heading_permalink_and_nbsp_are_cleaned(tmp_path: Path) -> None:
    page = tmp_path / "ddl-basics.html"
    page.write_text(
        '<html><body><div class="sect1"><h2>5.1.&nbsp;Table Basics'
        '<a class="id_link" href="#DDL-BASICS">#</a></h2><p>A table.</p></div></body></html>'
    )
    assert load_doc("ddl-basics", tmp_path).text == "5.1. Table Basics\n\nA table."


def test_load_doc_matches_load_corpus() -> None:
    from_corpus = {d.doc_id: d for d in load_corpus(FIXTURES)}
    for doc_id in ("doc1", "doc3", "guides/nested", "sql-lock"):
        single = load_doc(doc_id, FIXTURES)
        assert single.text == from_corpus[doc_id].text
        assert single.doc_id == doc_id


def test_load_corpus_is_reproducible() -> None:
    assert load_corpus(FIXTURES) == load_corpus(FIXTURES)


def test_doc_id_roundtrip_and_hash_rejected(tmp_path: Path) -> None:
    assert doc_id_for(FIXTURES / "guides" / "nested.txt", FIXTURES) == "guides/nested"
    assert path_for("guides/nested", FIXTURES) == (FIXTURES / "guides" / "nested.txt").resolve()
    bad = tmp_path / "a#b.txt"
    bad.write_text("x")
    with pytest.raises(ValueError):
        doc_id_for(bad, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_doc("missing", FIXTURES)


def test_empty_corpus(tmp_path: Path) -> None:
    assert load_corpus(tmp_path) == []


# ------------------------------------------------------------------- CLI


def test_cli_show_prints_every_chunk_with_id(capsys: pytest.CaptureFixture[str]) -> None:
    rc = chunk_cli.main(["--strategy", "fixed512", "--show", "doc3", "--corpus", str(FIXTURES)])
    out = capsys.readouterr().out
    assert rc == 0
    expected = chunk_cli.STRATEGIES["fixed512"].chunk(load_doc("doc3", FIXTURES))
    headers = [line for line in out.splitlines() if line.startswith("== ")]
    assert len(headers) == len(expected) > 1
    for c, header in zip(expected, headers, strict=True):
        assert header.startswith(f"== {c.chunk_id}  [{c.start}:{c.end})")
        assert c.text in out


def test_cli_list_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert chunk_cli.main(["--list", "--corpus", str(FIXTURES)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("doc1\t")
    assert chunk_cli.main(["--show", "nope", "--corpus", str(FIXTURES)]) == 1
    assert "no corpus file" in capsys.readouterr().err
