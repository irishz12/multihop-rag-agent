"""Oracle route label tests — offline, synthetic per_query records (same
shape as results/retrieval_eval_development.json's per_query entries).
Proves the exact CE@5 rule and that null_query (empty gold) is rejected,
not silently labeled.
"""

from __future__ import annotations

import pytest

from mhrag.routing.oracle import (
    ORACLE_K,
    ROUTE_LABELS,
    compute_oracle_label,
    compute_oracle_labels,
    label_distribution,
)


def _pq(qa_id, gold_doc_ids, hybrid_top10, reranker_top10, question_type="inference_query", hop_count=2):
    return {
        "qa_id": qa_id,
        "question_type": question_type,
        "hop_count": hop_count,
        "gold_doc_ids": gold_doc_ids,
        "methods": {
            "hybrid": {"unique_doc_ids_top10": hybrid_top10},
            "hybrid_reranker": {"unique_doc_ids_top10": reranker_top10},
        },
    }


def test_oracle_k_is_5():
    assert ORACLE_K == 5


def test_simple_when_hybrid_alone_covers_gold_within_top5():
    pq = _pq("q1", ["a", "b"], hybrid_top10=["a", "x", "b", "y", "z"], reranker_top10=["a", "b", "x", "y", "z"])
    label = compute_oracle_label(pq)
    assert label.route == "SIMPLE"
    assert label.hybrid_complete_evidence_at_5 is True
    assert label.hybrid_reranker_complete_evidence_at_5 is True  # also true, doesn't matter for SIMPLE


def test_medium_when_hybrid_fails_but_reranker_covers_gold_within_top5():
    # gold doc "b" is at position 6 in hybrid (outside top-5) but position 3 after reranking.
    pq = _pq(
        "q2", ["a", "b"],
        hybrid_top10=["a", "x", "y", "z", "w", "b"],
        reranker_top10=["a", "x", "b", "y", "z"],
    )
    label = compute_oracle_label(pq)
    assert label.route == "MEDIUM"
    assert label.hybrid_complete_evidence_at_5 is False
    assert label.hybrid_reranker_complete_evidence_at_5 is True


def test_complex_when_neither_covers_gold_within_top5():
    pq = _pq(
        "q3", ["a", "b"],
        hybrid_top10=["x", "y", "z", "w", "v", "a", "b"],
        reranker_top10=["x", "y", "z", "w", "v", "a", "b"],
    )
    label = compute_oracle_label(pq)
    assert label.route == "COMPLEX"
    assert label.hybrid_complete_evidence_at_5 is False
    assert label.hybrid_reranker_complete_evidence_at_5 is False


def test_boundary_gold_doc_exactly_at_rank_5_counts_as_covered():
    # gold "b" at index 4 (0-based) = rank 5 (1-based) -> WITHIN top-5.
    pq = _pq("q4", ["a", "b"], hybrid_top10=["x", "y", "z", "a", "b"], reranker_top10=["x", "y", "z", "a", "b"])
    label = compute_oracle_label(pq)
    assert label.route == "SIMPLE"


def test_boundary_gold_doc_exactly_at_rank_6_does_not_count():
    # gold "b" at index 5 (0-based) = rank 6 (1-based) -> OUTSIDE top-5.
    pq = _pq("q5", ["a", "b"], hybrid_top10=["x", "y", "z", "w", "a", "b"], reranker_top10=["x", "y", "z", "w", "a", "b"])
    label = compute_oracle_label(pq)
    assert label.route == "COMPLEX"


def test_single_gold_doc_question():
    pq = _pq("q6", ["a"], hybrid_top10=["a", "x", "y"], reranker_top10=["a", "x", "y"])
    label = compute_oracle_label(pq)
    assert label.route == "SIMPLE"


def test_empty_gold_doc_ids_raises_not_silently_labeled():
    """Defensive: an empty gold set (null_query shape) must never reach
    oracle labeling — reject loudly rather than assign a meaningless route."""
    pq = _pq("q_null", [], hybrid_top10=["x"], reranker_top10=["x"])
    with pytest.raises(ValueError, match="empty gold_doc_ids"):
        compute_oracle_label(pq)


def test_compute_oracle_labels_processes_whole_artifact_in_order():
    artifact = {
        "per_query": [
            _pq("q1", ["a"], ["a"], ["a"]),
            _pq("q2", ["a", "b"], ["x", "y", "z", "w", "v", "a", "b"], ["a", "b"]),
        ]
    }
    labels = compute_oracle_labels(artifact)
    assert [label.qa_id for label in labels] == ["q1", "q2"]
    assert [label.route for label in labels] == ["SIMPLE", "MEDIUM"]


def test_label_distribution_counts_all_three_routes():
    artifact = {
        "per_query": [
            _pq("q1", ["a"], ["a"], ["a"]),  # SIMPLE
            _pq("q2", ["a", "b"], ["x", "y", "z", "w", "v", "a", "b"], ["a", "b"]),  # MEDIUM
            _pq("q3", ["a", "b"], ["x"] * 7, ["x"] * 7),  # COMPLEX
        ]
    }
    dist = label_distribution(compute_oracle_labels(artifact))
    assert dist == {"SIMPLE": 1, "MEDIUM": 1, "COMPLEX": 1}
    assert set(dist) == set(ROUTE_LABELS)


def test_real_frozen_artifact_produces_265_labels_with_no_crash():
    """Integration sanity check against the actual frozen Phase 4/4.1/5
    artifact — not a live call, just reading the already-committed JSON
    file this evaluator is designed to consume."""
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "results" / "retrieval_eval_development.json"
    if not path.exists():
        pytest.skip("results/retrieval_eval_development.json not present in this checkout")
    artifact = json.loads(path.read_text())
    labels = compute_oracle_labels(artifact)
    assert len(labels) == 265
    dist = label_distribution(labels)
    assert sum(dist.values()) == 265
    assert all(v > 0 for v in dist.values())  # all three routes actually occur in real data
