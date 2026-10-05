"""Hand-computed examples for recall@k, MRR and nDCG@k (CLAUDE.md build order step 6).

Worked numbers, retrieved = [a, b, c, d, e], relevant = {b, d}:
  rank of b = 2, rank of d = 4
  DCG@5  = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
  IDCG@5 = 1/log2(2) + 1/log2(3) = 1 + 0.63093     = 1.63093
  nDCG@5 = 0.65093
  DCG@3  = 0.63093 (only b inside the top 3), IDCG@3 = 1.63093, nDCG@3 = 0.38685
"""

from __future__ import annotations

import math

import pytest

from evals.metrics.retrieval import (
    RetrievalScores,
    aggregate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    score_query,
)

RETRIEVED = ["a", "b", "c", "d", "e"]
LOG2_3 = math.log2(3)
LOG2_5 = math.log2(5)


def test_worked_example_at_k5() -> None:
    assert recall_at_k(RETRIEVED, {"b", "d"}, 5) == 1.0
    assert mrr(RETRIEVED, {"b", "d"}, 5) == 0.5
    expected = (1 / LOG2_3 + 1 / LOG2_5) / (1 + 1 / LOG2_3)
    assert ndcg_at_k(RETRIEVED, {"b", "d"}, 5) == pytest.approx(expected)
    assert expected == pytest.approx(0.65093, abs=1e-5)


def test_worked_example_at_k3_cuts_off_d() -> None:
    assert recall_at_k(RETRIEVED, {"b", "d"}, 3) == 0.5
    assert mrr(RETRIEVED, {"b", "d"}, 3) == 0.5
    assert ndcg_at_k(RETRIEVED, {"b", "d"}, 3) == pytest.approx((1 / LOG2_3) / (1 + 1 / LOG2_3))
    assert ndcg_at_k(RETRIEVED, {"b", "d"}, 3) == pytest.approx(0.38685, abs=1e-5)


def test_perfect_and_zero() -> None:
    assert score_query(["c", "a", "b"], {"a", "b", "c"}, 3) == RetrievalScores(1.0, 1.0, 1.0)
    assert score_query(RETRIEVED, {"a"}, 5) == RetrievalScores(1.0, 1.0, 1.0)
    assert score_query(RETRIEVED, {"x"}, 5) == RetrievalScores(0.0, 0.0, 0.0)
    assert score_query([], {"a"}, 5) == RetrievalScores(0.0, 0.0, 0.0)


def test_first_relevant_at_rank_three() -> None:
    # retrieved = [x, y, a, b]; relevant = {a, b}; k = 4
    # MRR = 1/3; DCG = 1/log2(4) + 1/log2(5) = 0.5 + 0.43068; IDCG = 1.63093
    r = ["x", "y", "a", "b"]
    assert mrr(r, {"a", "b"}, 4) == pytest.approx(1 / 3)
    assert ndcg_at_k(r, {"a", "b"}, 4) == pytest.approx((0.5 + 1 / LOG2_5) / (1 + 1 / LOG2_3))
    assert ndcg_at_k(r, {"a", "b"}, 4) == pytest.approx(0.57064, abs=1e-5)


def test_ideal_dcg_is_capped_at_k() -> None:
    # 3 relevant ids but k = 2: the best achievable is 2 hits, so a list with
    # both top slots relevant scores nDCG 1.0 while recall is 2/3.
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == 1.0
    assert recall_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(2 / 3)


def test_unanswerable_entries_have_no_score() -> None:
    assert score_query(RETRIEVED, set(), 5) == RetrievalScores(None, None, None)


def test_repeated_ids_keep_first_rank_only() -> None:
    assert mrr(["a", "a", "b"], {"b"}, 5) == 0.5  # b is at rank 2, not 3
    assert recall_at_k(["b", "b"], {"b", "c"}, 2) == 0.5


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, {"a"}, 0)


def test_aggregate_means_scored_and_skips_unanswerable() -> None:
    agg = aggregate(
        [
            RetrievalScores(1.0, 1.0, 1.0),
            RetrievalScores(0.5, 0.5, 0.4),
            RetrievalScores(None, None, None),
        ]
    )
    assert (agg.recall, agg.mrr, agg.ndcg) == (0.75, 0.75, 0.7)
    assert (agg.scored, agg.skipped) == (2, 1)
    with pytest.raises(ValueError):
        aggregate([RetrievalScores(None, None, None)])
