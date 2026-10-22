"""Faithfulness via Ragas: is the generated answer supported by the retrieved chunks?

Ragas' Faithfulness metric splits the answer into statements and asks a judge
LLM whether each one is entailed by the retrieved contexts; the score is the
supported fraction in [0, 1]. The judge is a config field (judge:) and should
not be the answer model, to limit self-preference.

Only real answers are judged: an abstention ("Not answerable...") or an API
refusal has no claims to verify and scores None, which aggregation skips. The
results row shows the mean and how many answers it covers.

Live smoke test once a key is in .env:
    python -m evals.metrics.generation "What does NOWAIT do?" \
        "LOCK TABLE does not wait [sql-lock#c0]." "If NOWAIT is specified, LOCK TABLE does not wait"
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from typing import Any, Protocol

DEFAULT_JUDGE = "claude-sonnet-4-6"
JUDGE_MAX_TOKENS = 4096


class FaithfulnessJudge(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, question: str, answer: str, contexts: Sequence[str]) -> float | None:
        """Supported fraction of the answer's statements, or None if it could not be judged."""
        ...


def sampling_args(model: str) -> dict[str, Any]:
    """Temperature 0 for repeatable judgements where the API still accepts it.

    Claude 4.7+ / 5.x reject sampling parameters, so they get none."""
    if model.startswith(("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5")):
        return {"temperature": 0.0}
    return {}


def build_ragas_llm(model: str, client: Any | None = None) -> Any:
    """A ragas InstructorLLM over the Anthropic SDK.

    Uses instructor's JSON mode rather than its default forced tool call, which
    Claude 5.5 models reject, and drops ragas' default top_p so only one
    sampling parameter is ever sent."""
    import anthropic
    import instructor
    from ragas.llms import InstructorLLM

    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                f"ANTHROPIC_API_KEY is not set; needed for the faithfulness judge {model}. "
                "Run without --faithfulness to skip it."
            )
        client = anthropic.Anthropic(max_retries=5)
    wrapped = instructor.from_anthropic(client, mode=instructor.Mode.JSON)
    llm = InstructorLLM(client=wrapped, model=model, provider="anthropic")
    llm.model_args = {"max_tokens": JUDGE_MAX_TOKENS, **sampling_args(model)}
    return llm


class RagasFaithfulness:
    def __init__(self, model: str = DEFAULT_JUDGE, metric: Any | None = None) -> None:
        self.model = model
        self._metric = metric  # injectable for tests; built lazily otherwise

    @property
    def name(self) -> str:
        return f"ragas-faithfulness/{self.model}"

    def _get_metric(self) -> Any:
        if self._metric is None:
            from ragas.metrics import Faithfulness

            self._metric = Faithfulness(llm=build_ragas_llm(self.model))
        return self._metric

    def score(self, question: str, answer: str, contexts: Sequence[str]) -> float | None:
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=list(contexts)
        )
        value = self._get_metric().single_turn_score(sample)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m evals.metrics.generation", description="live faithfulness smoke test"
    )
    ap.add_argument("question")
    ap.add_argument("answer")
    ap.add_argument("contexts", nargs="+")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    args = ap.parse_args(argv)
    try:
        value = RagasFaithfulness(args.judge).score(args.question, args.answer, args.contexts)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"faithfulness={value} (judge={args.judge})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
