"""Versioned prompts. A change to the wording is a new version, never an edit in
place, so a results.md row can always be traced to the exact prompt it ran."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.base import Hit

ABSTAIN_PREFIX = "Not answerable from the documentation."


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str

    def user(self, question: str, hits: list[Hit]) -> str:
        blocks = "\n\n".join(f'<chunk id="{h.chunk_id}">\n{h.text}\n</chunk>' for h in hits)
        return f"<context>\n{blocks}\n</context>\n\n<question>\n{question}\n</question>"


V1 = Prompt(
    version="v1",
    system=(
        "You answer questions about the PostgreSQL 16 documentation using only the "
        "chunks provided in <context>. Rules:\n"
        "- Answer concisely (one to four sentences), stating facts exactly as the chunks do.\n"
        "- After every claim, cite the chunk id(s) it comes from in square brackets, "
        "for example [sql-lock#c0].\n"
        "- Use nothing outside the context. If the chunks do not contain the answer, "
        f"reply with exactly: {ABSTAIN_PREFIX} followed by one sentence saying what is "
        "missing, and cite nothing."
    ),
)

PROMPTS: dict[str, Prompt] = {"v1": V1}
