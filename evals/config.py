"""Run configs: one YAML per variant (CLAUDE.md rule 4).

    name: baseline
    chunking: fixed512              # key in ingest.chunk.STRATEGIES
    embedding: text-embedding-3-small   # key in ingest.embed.EMBEDDERS
    retrieval: dense                # dense | sparse | hybrid (later)
    top_k: 5
    rerank: null                    # reranker name (later phase)
    generation: claude-opus-5-5     # answer model, used only with run.py --generate
    prompt: v1                      # key in generation.prompts.PROMPTS

Unknown keys are an error so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from ingest.chunk import STRATEGIES
from ingest.embed import EMBEDDERS

RETRIEVALS = ("dense", "sparse", "hybrid")


@dataclass(frozen=True)
class RunConfig:
    name: str
    chunking: str = "fixed512"
    embedding: str = "text-embedding-3-small"
    retrieval: str = "dense"
    top_k: int = 5
    rerank: str | None = None
    generation: str = "claude-opus-5-5"
    prompt: str = "v1"


def load_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    known = {f.name for f in fields(RunConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; allowed: {sorted(known)}")
    if "name" not in data:
        raise ValueError(f"{path}: 'name' is required")
    cfg = RunConfig(**data)
    if cfg.chunking not in STRATEGIES:
        raise ValueError(f"{path}: chunking {cfg.chunking!r} not in {sorted(STRATEGIES)}")
    if cfg.embedding not in EMBEDDERS:
        raise ValueError(f"{path}: embedding {cfg.embedding!r} not in {sorted(EMBEDDERS)}")
    if cfg.retrieval not in RETRIEVALS:
        raise ValueError(f"{path}: retrieval {cfg.retrieval!r} not in {RETRIEVALS}")
    if not isinstance(cfg.top_k, int) or cfg.top_k < 1:
        raise ValueError(f"{path}: top_k must be a positive integer")
    from generation.prompts import PROMPTS  # local import: generation depends on retrieval

    if cfg.prompt not in PROMPTS:
        raise ValueError(f"{path}: prompt {cfg.prompt!r} not in {sorted(PROMPTS)}")
    return cfg
