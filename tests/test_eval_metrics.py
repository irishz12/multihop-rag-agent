"""Document-level retrieval metric tests: dedup, Recall@K, Hit@K, MRR@10,
NDCG@10, Complete-Evidence@K, aggregation.

Expected values for MRR/NDCG are computed independently in each test (via
the textbook formula, not by calling into the implementation) so these
tests actually catch a wrong formula, not just a self-consistent one.
"""

from __future__ import annotations

import math

import pytest

from mhrag.eval.metrics import (
    collapse_to_unique_documents,
    complete_evidence_at_k,
    compute_all_metrics,
    hit_at_k,
    mean_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from mhrag.retrieval.schema import RetrievalResult


def _result(rank: int, doc_id: str, chunk_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=1.0 / rank,
        method="dense",
        chunk_id=chunk_id or f"chunk-{rank}",
        doc_id=doc_id,
        title=f"Title {doc_id}",
        url=f"https://example.com/{doc_id}",
        source="Source",
        category="technology",
        published_at="2024-01-01T00:00:00+00:00",
        text="chunk text",
        position=0,
    )


# --- chunk-to-document deduplication ------------------------------------------------


def test_collapse_dedupes_preserving_first_occurrence_rank():
    results = [
        _result(1, "doc1"),
        _result(2, "doc2"),
        _result(3, "doc1"),  # repeat — should not appear again
        _result(4, "doc3"),
        _result(5, "doc2"),  # repeat
    ]
    assert collapse_to_unique_documents(results) == ["doc1", "doc2", "doc3"]


def test_collapse_of_all_same_document_yields_one_entry():
    results = [_result(i, "doc1") for i in range(1, 6)]
    assert collapse_to_unique_documents(results) == ["doc1"]


def test_collapse_of_empty_results_is_empty():
    assert collapse_to_unique_documents([]) == []


# --- Recall@K ------------------------------------------------------------------------


def test_recall_at_k_partial_and_full_match():
    ranking = ["a", "x", "b", "y", "z", "c"]
    gold = frozenset({"a", "b", "c"})
    assert recall_at_k(ranking, gold, k=2) == pytest.approx(1 / 3)  # only "a" in top-2
    assert recall_at_k(ranking, gold, k=3) == pytest.approx(2 / 3)  # "a","b" in top-3
    assert recall_at_k(ranking, gold, k=6) == pytest.approx(1.0)  # all three by rank 6


def test_recall_at_k_zero_when_nothing_found():
    assert recall_at_k(["x", "y"], frozenset({"a"}), k=2) == 0.0


def test_recall_at_k_raises_on_empty_gold():
    with pytest.raises(ValueError):
        recall_at_k(["a"], frozenset(), k=4)


# --- Hit@K -----------------------------------------------------------------------------


def test_hit_at_k_true_when_any_gold_present():
    assert hit_at_k(["x", "a", "z"], frozenset({"a", "b"}), k=3) == 1


def test_hit_at_k_false_when_none_present():
    assert hit_at_k(["x", "y", "z"], frozenset({"a", "b"}), k=3) == 0


def test_hit_at_k_respects_cutoff():
    assert hit_at_k(["x", "y", "a"], frozenset({"a"}), k=2) == 0  # "a" is rank 3, outside k=2
    assert hit_at_k(["x", "y", "a"], frozenset({"a"}), k=3) == 1


# --- MRR@10 ------------------------------------------------------------------------------


def test_mrr_at_k_reciprocal_of_first_relevant_rank():
    ranking = ["x", "y", "a", "z"]
    assert mrr_at_k(ranking, frozenset({"a"}), k=10) == pytest.approx(1 / 3)


def test_mrr_at_k_zero_when_no_relevant_in_top_k():
    assert mrr_at_k(["x", "y", "z"], frozenset({"a"}), k=10) == 0.0


def test_mrr_at_k_uses_earliest_relevant_when_multiple_gold_present():
    ranking = ["x", "a", "b", "y"]
    assert mrr_at_k(ranking, frozenset({"a", "b"}), k=10) == pytest.approx(1 / 2)


# --- NDCG@10 -----------------------------------------------------------------------------


def test_ndcg_at_k_perfect_ranking_is_one():
    ranking = ["a", "b", "x", "y"]
    gold = frozenset({"a", "b"})
    assert ndcg_at_k(ranking, gold, k=10) == pytest.approx(1.0)


def test_ndcg_at_k_zero_when_no_relevant_present():
    assert ndcg_at_k(["x", "y", "z"], frozenset({"a"}), k=10) == 0.0


def test_ndcg_at_k_matches_hand_computed_value():
    gold = frozenset({"a", "c"})
    ranking = ["x", "a", "y", "c", "z"]  # relevant at ranks 2 and 4
    result = ndcg_at_k(ranking, gold, k=10)

    dcg = 1 / math.log2(2 + 1) + 1 / math.log2(4 + 1)
    idcg = 1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)  # 2 gold docs, ideal: ranks 1 and 2
    expected = dcg / idcg

    assert result == pytest.approx(expected)


def test_ndcg_at_k_raises_on_empty_gold():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], frozenset(), k=10)


# --- Complete-Evidence@K ---------------------------------------------------------------


def test_complete_evidence_requires_all_gold_present():
    gold = frozenset({"a", "b"})
    assert complete_evidence_at_k(["a", "x", "y"], gold, k=3) == 0  # missing "b"
    assert complete_evidence_at_k(["a", "b", "x"], gold, k=3) == 1  # both present


def test_complete_evidence_respects_cutoff():
    gold = frozenset({"a", "b"})
    ranking = ["a", "x", "y", "b"]  # "b" is rank 4
    assert complete_evidence_at_k(ranking, gold, k=3) == 0
    assert complete_evidence_at_k(ranking, gold, k=4) == 1


def test_complete_evidence_is_stricter_than_hit():
    """For a partial match, Hit@K is 1 but Complete-Evidence@K must be 0."""
    gold = frozenset({"a", "b"})
    ranking = ["a", "x", "y"]
    assert hit_at_k(ranking, gold, k=3) == 1
    assert complete_evidence_at_k(ranking, gold, k=3) == 0


# --- compute_all_metrics / mean_metrics -------------------------------------------------


def test_compute_all_metrics_returns_all_expected_keys():
    ranking = ["a", "x", "b", "y", "z", "c", "d", "e", "f", "g", "h"]
    gold = frozenset({"a", "b", "c"})
    metrics = compute_all_metrics(ranking, gold)
    expected_keys = {
        "recall@4",
        "recall@5",
        "recall@10",
        "hit@4",
        "hit@10",
        "complete_evidence@4",
        "complete_evidence@10",
        "mrr@10",
        "ndcg@10",
    }
    assert set(metrics.keys()) == expected_keys


def test_compute_all_metrics_raises_on_empty_gold():
    with pytest.raises(ValueError):
        compute_all_metrics(["a"], frozenset())


def test_mean_metrics_averages_across_queries():
    per_query = [{"recall@4": 1.0, "hit@4": 1.0}, {"recall@4": 0.0, "hit@4": 0.0}]
    result = mean_metrics(per_query)
    assert result == {"recall@4": 0.5, "hit@4": 0.5}


def test_mean_metrics_raises_on_empty_list():
    with pytest.raises(ValueError):
        mean_metrics([])
