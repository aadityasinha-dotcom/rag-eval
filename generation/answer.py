"""Generate an answer from retrieved chunks, citing chunk ids ("stuff into prompt").

The baseline generator is Claude via the Anthropic SDK. The model is a config
field (generation:) so a variant can swap it; everything else about the call
is fixed so rows stay comparable. Refusals and abstentions are recorded, not
hidden: a `refusal` stop reason is a safety decline by the API, an abstention
is the model following the prompt's "not answerable" rule.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from generation.prompts import ABSTAIN_PREFIX, Prompt
from retrieval.base import Hit

CITATION = re.compile(r"\[([^\[\]\s]+#c\d+)\]")
MAX_TOKENS = 2048  # answers are a few sentences; this is headroom, not a target
EFFORT = "medium"


@dataclass(frozen=True)
class Answer:
    text: str
    cited_chunk_ids: tuple[str, ...]
    abstained: bool
    refused: bool
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def parse_citations(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(CITATION.findall(text)))


def abstained(text: str) -> bool:
    return text.strip().startswith(ABSTAIN_PREFIX)


class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def answer(self, question: str, hits: list[Hit]) -> Answer: ...


class ClaudeGenerator:
    def __init__(self, model: str, prompt: Prompt, client: Any | None = None) -> None:
        self.model = model
        self.prompt = prompt
        self._client = client

    @property
    def name(self) -> str:
        return f"{self.model}/{self.prompt.version}"

    def _get_client(self) -> Any:
        if self._client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set; needed to generate answers with "
                    f"{self.model}. Run without --generate for retrieval-only metrics."
                )
            import anthropic

            self._client = anthropic.Anthropic(max_retries=5)
        return self._client

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        client = self._get_client()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": self.prompt.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={"effort": EFFORT},
            messages=[{"role": "user", "content": self.prompt.user(question, hits)}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        refused = resp.stop_reason == "refusal"
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return Answer(
            text=text,
            cited_chunk_ids=parse_citations(text),
            abstained=abstained(text),
            refused=refused,
            model=self.model,
            prompt_version=self.prompt.version,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
        )
