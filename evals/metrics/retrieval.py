"""Retrieval metrics against the golden set's relevant_chunk_ids.

All three are binary-relevance, computed per query over the ranked list a
retriever returned, then averaged across queries. A query with no relevant
ids (an "unanswerable" golden entry) has no defined retrieval score and
returns None; aggregation skips it rather than counting it as 0 or 1.

  recall@k  fraction of relevant ids that appear in the top k
  MRR       1 / rank of the first relevant id in the top k, 0 if none
  nDCG@k    sum(1/log2(rank+1) for relevant ranks <= k) / ideal DCG, where
            the ideal places min(|relevant|, k) relevant ids at ranks 1..
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def _top(retrieved: Sequence[str], k: int) -> list[str]:
    """First k unique ids in rank order; a repeated id keeps its first rank."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return list(dict.fromkeys(retrieved))[:k]


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    return len(rel.intersection(_top(retrieved, k))) / len(rel)


def mrr(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    for rank, cid in enumerate(_top(retrieved, k), start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(_top(retrieved, k), start=1)
        if cid in rel
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
    return dcg / ideal


@dataclass(frozen=True)
class RetrievalScores:
    recall: float | None
    mrr: float | None
    ndcg: float | None


def score_query(retrieved: Sequence[str], relevant: Collection[str], k: int) -> RetrievalScores:
    return RetrievalScores(
        recall=recall_at_k(retrieved, relevant, k),
        mrr=mrr(retrieved, relevant, k),
        ndcg=ndcg_at_k(retrieved, relevant, k),
    )


@dataclass(frozen=True)
class AggregateScores:
    recall: float
    mrr: float
    ndcg: float
    scored: int  # queries with relevant ids
    skipped: int  # queries without (unanswerable)


def aggregate(scores: Sequence[RetrievalScores]) -> AggregateScores:
    """Mean of each metric over queries that have one. Raises if none do."""
    scored = [s for s in scores if s.recall is not None]
    if not scored:
        raise ValueError("no scorable queries: every entry has empty relevant_chunk_ids")

    def mean(values: list[float | None]) -> float:
        xs = [v for v in values if v is not None]
        return sum(xs) / len(xs)

    return AggregateScores(
        recall=mean([s.recall for s in scored]),
        mrr=mean([s.mrr for s in scored]),
        ndcg=mean([s.ndcg for s in scored]),
        scored=len(scored),
        skipped=len(scores) - len(scored),
    )
