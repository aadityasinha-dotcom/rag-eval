"""Optional tracing through the llm-metrics SDK (the `llmobserve` of CLAUDE.md rule 8).

Everything here is a pass-through unless LLM_METRICS_API_KEY is set *and* the
SDK is installed (`uv sync --extra observe`). With neither, no module is
imported and no behaviour changes; with both, each golden query becomes a
trace: a `retrieval` observation for the search, a `generation` observation
from the wrapped Anthropic client with real token counts, and the per-query
metrics attached as scores. Cost is priced server-side on the platform; the
run file records each query's trace id so $/query can be read back later.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
T = TypeVar("T")


def _sdk() -> Any | None:
    try:
        import llm_metrics
    except ImportError:
        return None
    return llm_metrics


def enabled() -> bool:
    """True only when a key is configured, the SDK is importable, and it is not disabled."""
    load_dotenv(ROOT / ".env")
    if not os.environ.get("LLM_METRICS_API_KEY"):
        return False
    if os.environ.get("LLM_METRICS_ENABLED", "1") == "0":
        return False
    return _sdk() is not None


def setup(release: str | None, environment: str = "eval") -> bool:
    """Stamp every trace with the git commit and an environment. Returns enabled()."""
    if not enabled():
        return False
    sdk = _sdk()
    assert sdk is not None
    kwargs: dict[str, Any] = {"environment": os.environ.get("LLM_METRICS_ENVIRONMENT", environment)}
    if release and not os.environ.get("LLM_METRICS_RELEASE"):
        kwargs["release"] = release
    sdk.configure(**kwargs)
    return True


def traced(
    name: str,
    as_type: str,
    fn: Callable[..., T],
    *args: Any,
    metadata: Mapping[str, Any] | None = None,
    capture_output: bool = False,
    **kwargs: Any,
) -> T:
    """Call fn(*args, **kwargs) inside an observation of the given type.

    A call with no enclosing trace starts one, so wrapping the per-query body
    with as_type="span" makes it the root and nested calls join it."""
    if not enabled():
        return fn(*args, **kwargs)
    sdk = _sdk()
    assert sdk is not None
    wrapped = sdk.observe(
        name=name, as_type=as_type, metadata=dict(metadata or {}), capture_output=capture_output
    )(fn)
    return wrapped(*args, **kwargs)  # type: ignore[no-any-return]


def annotate_trace(
    *, session_id: str | None = None, tags: Iterable[str] | None = None, **metadata: Any
) -> None:
    if not enabled():
        return
    sdk = _sdk()
    assert sdk is not None
    kw: dict[str, Any] = {}
    if session_id is not None:
        kw["session_id"] = session_id
    if tags is not None:
        kw["tags"] = list(tags)
    if metadata:
        kw["metadata"] = metadata
    sdk.update_trace(**kw)


def record_score(
    name: str, value: float | int | bool, *, source: str, comment: str | None = None
) -> None:
    """Attach a metric to the ambient trace. Sources: heuristic | llm_judge | human."""
    if not enabled():
        return
    sdk = _sdk()
    assert sdk is not None
    sdk.score(name, value, source=source, comment=comment)


def current_trace_id() -> str | None:
    if not enabled():
        return None
    from llm_metrics import context

    return context.current_trace_id()


def wrap_anthropic(client: T) -> T:
    if not enabled():
        return client
    from llm_metrics.integrations.anthropic import wrap_anthropic as _wrap

    return _wrap(client)  # type: ignore[no-any-return]


def wrap_openai(client: T) -> T:
    if not enabled():
        return client
    from llm_metrics.integrations.openai import wrap_openai as _wrap

    return _wrap(client)  # type: ignore[no-any-return]


def flush(timeout: float = 10.0) -> None:
    """Block briefly so the last batch leaves before the process exits."""
    if not enabled():
        return
    sdk = _sdk()
    assert sdk is not None
    sdk.flush(timeout)
