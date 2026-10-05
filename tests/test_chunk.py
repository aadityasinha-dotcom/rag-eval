"""Chunk ids are deterministic, positional, and never cross document boundaries."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from ingest.chunk import STRATEGIES, Chunk, FixedSizeStrategy, chunk_corpus
from ingest.parse import ParsedDoc

FIXED = STRATEGIES["fixed512"]
ID_RE = re.compile(r"^(?P<doc>[^#]+)#c(?P<n>\d+)$")


def pd(doc_id: str, text: str) -> ParsedDoc:
    return ParsedDoc(doc_id=doc_id, path=Path(f"{doc_id}.txt"), text=text)


def synthetic(n_tokens: int = 300) -> str:
    """'0000 0001 0002 ...' - 5 chars per token, 1500 chars for n_tokens=300."""
    return "".join(f"{i:04d} " for i in range(n_tokens))


# ------------------------------------------------------------- determinism


def test_fixed512_is_the_only_strategy_and_has_expected_params() -> None:
    assert list(STRATEGIES) == ["fixed512"]
    assert isinstance(FIXED, FixedSizeStrategy)
    assert (FIXED.size, FIXED.overlap, FIXED.stride) == (512, 64, 448)


def test_pinned_expectation_for_known_input() -> None:
    """Hard-coded ids, offsets and hashes. If this fails, chunking changed and
    every golden entry built on fixed512 must be re-labelled (CLAUDE.md rule 7)."""
    chunks = FIXED.chunk(pd("doc3", synthetic()))
    expected = [
        ("doc3#c0", 0, 512, "34987559ba2b"),
        ("doc3#c1", 448, 960, "9d794ef68b3c"),
        ("doc3#c2", 896, 1408, "4abca5ead9be"),
        ("doc3#c3", 1344, 1500, "91a4125ee5a6"),
    ]
    got = [(c.chunk_id, c.start, c.end, c.text_hash[:12]) for c in chunks]
    assert got == expected


def test_rerun_on_identical_input_gives_identical_chunks() -> None:
    doc = pd("doc3", synthetic())
    first = FIXED.chunk(doc)
    second = FIXED.chunk(pd("doc3", synthetic()))  # fresh objects, same content
    assert first == second
    assert [c.text_hash for c in first] == [c.text_hash for c in second]


def test_ids_have_doc_hash_c_n_form_and_are_sequential_from_zero() -> None:
    chunks = FIXED.chunk(pd("pg16/ddl", synthetic(500)))
    for i, c in enumerate(chunks):
        m = ID_RE.match(c.chunk_id)
        assert m, c.chunk_id
        assert m["doc"] == "pg16/ddl" == c.doc_id
        assert int(m["n"]) == i == c.n


def test_text_hash_is_sha256_of_text() -> None:
    c = Chunk(chunk_id="d#c0", doc_id="d", n=0, start=0, end=3, text="abc")
    assert c.text_hash == hashlib.sha256(b"abc").hexdigest()


# -------------------------------------------------------------- geometry


def test_window_geometry_and_overlap() -> None:
    text = synthetic(200)  # 1000 chars
    chunks = FIXED.chunk(pd("d", text))
    assert [(c.start, c.end) for c in chunks] == [(0, 512), (448, 960), (896, 1000)]
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.text[-64:] == nxt.text[:64]
        assert text[nxt.start : nxt.end] == nxt.text


@pytest.mark.parametrize(
    ("length", "n_chunks"),
    [(0, 0), (1, 1), (511, 1), (512, 1), (513, 2), (960, 2), (961, 3)],
)
def test_chunk_count_edges(length: int, n_chunks: int) -> None:
    chunks = FIXED.chunk(pd("d", "a" * length))
    assert len(chunks) == n_chunks
    if chunks:
        assert chunks[-1].end == length  # whole document covered
        assert all(c.text for c in chunks)  # no empty chunk


def test_strategy_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        FixedSizeStrategy(name="bad", size=100, overlap=100)
    with pytest.raises(ValueError):
        FixedSizeStrategy(name="bad", size=100, overlap=-1)


# ------------------------------------------------------ document boundaries


def test_chunks_never_span_documents() -> None:
    a, b = pd("a", synthetic(150)), pd("b", synthetic(130))
    chunks = chunk_corpus([a, b], FIXED)
    by_doc = {"a": a, "b": b}
    for c in chunks:
        doc = by_doc[c.doc_id]
        assert 0 <= c.start < c.end <= len(doc.text)
        assert c.text == doc.text[c.start : c.end]
    # the final chunk of `a` ends exactly at a's end; nothing of `b` leaks in
    last_a = [c for c in chunks if c.doc_id == "a"][-1]
    assert last_a.end == len(a.text)
    # a and b contain the same leading text, so their windows have the same
    # content but *different* ids - ids are per document
    assert chunks[0].text == [c for c in chunks if c.doc_id == "b"][0].text
    assert chunks[0].chunk_id == "a#c0"
    assert [c.chunk_id for c in chunks if c.doc_id == "b"][0] == "b#c0"


def test_doc_chunks_do_not_depend_on_other_docs_in_corpus() -> None:
    doc3 = pd("doc3", synthetic())
    alone = chunk_corpus([doc3], FIXED)
    with_neighbours = chunk_corpus([pd("doc1", "short"), doc3, pd("doc9", synthetic(999))], FIXED)
    assert [c for c in with_neighbours if c.doc_id == "doc3"] == alone


def test_editing_a_document_keeps_ids_and_preserves_untouched_prefix() -> None:
    original = synthetic()
    edited = original[:1300] + "EDITED" + original[1306:]  # same length, change in window 2
    before = FIXED.chunk(pd("doc3", original))
    after = FIXED.chunk(pd("doc3", edited))

    # ids are positional, so the id sequence is unchanged ...
    assert [c.chunk_id for c in before] == [c.chunk_id for c in after]
    # ... chunks before the edit are byte-identical ...
    assert before[0] == after[0]
    assert before[1] == after[1]
    # ... and the chunk holding the edit keeps its id but changes its hash,
    # which is the signal that golden entries citing doc3#c2 need re-labelling.
    assert before[2].chunk_id == after[2].chunk_id == "doc3#c2"
    assert before[2].text_hash != after[2].text_hash


def test_growing_a_document_appends_chunks_without_renumbering() -> None:
    before = FIXED.chunk(pd("doc3", synthetic(300)))
    after = FIXED.chunk(pd("doc3", synthetic(400)))
    assert len(after) > len(before)
    # every full window before the old tail is unchanged
    assert after[: len(before) - 1] == before[:-1]
