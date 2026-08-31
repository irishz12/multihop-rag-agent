"""Router feature extraction tests — offline, no Qdrant/embedding/BM25
model required. `extract_query_features` and `extract_retrieval_signals`
are pure functions; only `compute_router_features` (the live wrapper) is
untested here (exercised only by scripts/build_router_dataset.py)."""

from __future__ import annotations

import inspect

from mhrag.routing.features import (
    compute_router_features,
    extract_query_features,
    extract_retrieval_signals,
)
from mhrag.retrieval.schema import RetrievalResult


def _result(chunk_id: str, doc_id: str, rank: int, score: float, method: str = "dense") -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=score, method=method, chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text="chunk text", position=0,
    )


# --- extract_query_features ---------------------------------------------------------------


def test_query_length_counts_words_and_chars():
    f = extract_query_features("What year was it founded?")
    assert f.query_length_words == 5
    assert f.query_length_chars == len("What year was it founded?")


def test_comparison_marker_detected():
    f = extract_query_features("Did article A and article B both report the same outcome, or did they differ?")
    assert f.has_comparison_marker is True
    assert f.comparison_marker_count >= 1


def test_no_comparison_marker_when_absent():
    f = extract_query_features("What is the capital of France?")
    assert f.has_comparison_marker is False
    assert f.comparison_marker_count == 0


def test_temporal_marker_detected():
    f = extract_query_features("Was there a change in the narrative between October and December?")
    assert f.has_temporal_marker is True


def test_conjunction_count_detects_and_or():
    f = extract_query_features("Did X report on Sam Bankman-Fried and did Y report on him too?")
    assert f.has_conjunction_marker is True
    assert f.conjunction_count >= 1


def test_quoted_span_count_detects_quoted_titles():
    f = extract_query_features("Did the 'Sport Grill' article agree with \"The Roar\" on the outcome?")
    assert f.quoted_span_count == 2


def test_numeric_date_indicator_counts_years_and_months():
    f = extract_query_features("Compare the TechCrunch report from October 7, 2023 with the one from 2024.")
    assert f.numeric_date_indicator_count >= 2  # at least "October", "2023", "2024", "7"


def test_query_features_is_pure_function_of_text_only():
    """Structural: no parameter for gold answer/evidence_list/question_type/
    oracle label — the only input is the question string."""
    params = list(inspect.signature(extract_query_features).parameters)
    assert params == ["question"]


# --- extract_retrieval_signals -------------------------------------------------------------


def test_retrieval_signals_pure_function_takes_only_result_lists():
    params = list(inspect.signature(extract_retrieval_signals).parameters)
    assert params == ["dense_results", "bm25_results", "hybrid_results"]


def test_high_agreement_when_dense_and_bm25_share_all_top_docs():
    dense = [_result(f"d{i}", f"doc{i}", i + 1, 1.0 - i * 0.05, "dense") for i in range(5)]
    bm25 = [_result(f"b{i}", f"doc{i}", i + 1, 10.0 - i, "bm25") for i in range(5)]
    hybrid = [_result(f"h{i}", f"doc{i}", i + 1, 0.03 - i * 0.001, "hybrid") for i in range(5)]
    signals = extract_retrieval_signals(dense, bm25, hybrid)
    assert signals.dense_bm25_jaccard_top10 == 1.0
    assert signals.consensus_fraction_top5 == 1.0
    assert signals.mean_abs_rank_diff_common_docs == 0.0


def test_low_agreement_when_dense_and_bm25_share_nothing():
    dense = [_result(f"d{i}", f"doc_dense_{i}", i + 1, 1.0 - i * 0.05, "dense") for i in range(5)]
    bm25 = [_result(f"b{i}", f"doc_bm25_{i}", i + 1, 10.0 - i, "bm25") for i in range(5)]
    hybrid = [_result(f"h{i}", f"doc_dense_{i}", i + 1, 0.03 - i * 0.001, "hybrid") for i in range(5)]
    signals = extract_retrieval_signals(dense, bm25, hybrid)
    assert signals.dense_bm25_jaccard_top10 == 0.0
    assert signals.consensus_fraction_top5 == 0.0


def test_num_unique_docs_collapses_duplicate_chunks_from_same_doc():
    hybrid = [
        _result("c1", "docA", 1, 0.05, "hybrid"),
        _result("c2", "docA", 2, 0.04, "hybrid"),  # same doc as c1
        _result("c3", "docB", 3, 0.03, "hybrid"),
    ]
    signals = extract_retrieval_signals([], [], hybrid)
    assert signals.num_unique_docs_top5 == 2  # docA, docB — not 3 chunks


def test_score_gaps_computed_from_hybrid_top_scores():
    hybrid = [_result(f"h{i}", f"doc{i}", i + 1, 0.10 - i * 0.02, "hybrid") for i in range(5)]
    signals = extract_retrieval_signals([], [], hybrid)
    assert signals.hybrid_top1_score == 0.10
    assert abs(signals.score_gap_top1_top2 - 0.02) < 1e-9
    assert abs(signals.score_gap_top1_top5 - 0.08) < 1e-9


def test_empty_results_do_not_crash_and_return_zeros():
    signals = extract_retrieval_signals([], [], [])
    assert signals.hybrid_top1_score == 0.0
    assert signals.num_unique_docs_top5 == 0
    assert signals.dense_bm25_jaccard_top10 == 0.0
    assert signals.consensus_fraction_top5 == 0.0


# --- structural: no gold anywhere in this module -------------------------------------------


def test_compute_router_features_signature_has_no_gold_parameter():
    params = list(inspect.signature(compute_router_features).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label"}
    assert not (forbidden & set(params))
    assert "question" in params
