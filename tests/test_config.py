from __future__ import annotations

from pathlib import Path

import pytest

from evals.config import RunConfig, load_config

CONFIGS = Path(__file__).parent.parent / "evals" / "configs"


def test_checked_in_configs_load() -> None:
    base = load_config(CONFIGS / "baseline.yaml")
    assert base == RunConfig(
        name="baseline",
        chunking="fixed512",
        embedding="text-embedding-3-small",
        retrieval="dense",
        top_k=5,
        rerank=None,
    )
    assert load_config(CONFIGS / "smoke-hash64.yaml").embedding == "hash64"


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        ("name: x\nchunk: fixed512\n", "unknown keys"),
        ("chunking: fixed512\n", "'name' is required"),
        ("name: x\nchunking: semantic\n", "chunking 'semantic'"),
        ("name: x\nembedding: ada\n", "embedding 'ada'"),
        ("name: x\nretrieval: magic\n", "retrieval 'magic'"),
        ("name: x\ntop_k: 0\n", "top_k"),
        ("name: x\nprompt: v9\n", "prompt 'v9'"),
        ("- a\n", "mapping"),
    ],
)
def test_invalid_configs(tmp_path: Path, yaml_text: str, match: str) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match=match):
        load_config(p)
