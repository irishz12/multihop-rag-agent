"""Learned-router feature-vector assembly tests — offline, pure dataclass
plumbing. Verifies the FROZEN CONTRACT that `STAGE1_FEATURE_NAMES`/
`STAGE2_FEATURE_NAMES` order matches `stage1_feature_vector`/
`stage2_feature_vector`'s actual output order, 1:1, plus basic assembly
correctness and length invariants.
"""

from __future__ import annotations

from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.learned_features import (
    STAGE1_FEATURE_NAMES,
    STAGE2_ADDITIONAL_FEATURE_NAMES,
    STAGE2_FEATURE_NAMES,
    stage1_feature_vector,
    stage2_feature_vector,
)
from mhrag.routing.rerank_features import RerankSignals


def _router_features() -> RouterFeatures:
    query = QueryFeatures(
        query_length_words=6, query_length_chars=40, comparison_marker_count=1, has_comparison_marker=True,
        temporal_marker_count=2, has_temporal_marker=True, conjunction_count=3, has_conjunction_marker=False,
        quoted_span_count=4, numeric_date_indicator_count=5,
    )
    retrieval = RetrievalSignals(
        hybrid_top1_score=0.11, hybrid_top5_mean_score=0.12, score_gap_top1_top2=0.13, score_gap_top1_top5=0.14,
        dense_bm25_jaccard_top10=0.15, consensus_fraction_top5=0.16, num_unique_docs_top5=7, num_unique_docs_top10=8,
        mean_abs_rank_diff_common_docs=1.7,
    )
    return RouterFeatures(query=query, retrieval=retrieval)


def _rerank_signals() -> RerankSignals:
    return RerankSignals(
        rerank_top1_score=0.21, rerank_top5_mean_score=0.22, rerank_score_gap_top1_top2=0.23,
        rerank_score_gap_top1_top5=0.24, rank_change_mean_abs=2.5, top5_overlap_with_hybrid=0.6,
        num_docs_new_in_rerank_top5=2,
    )


def test_stage1_feature_names_and_vector_have_matching_length():
    features = _router_features()
    vector = stage1_feature_vector(features)
    assert len(vector) == len(STAGE1_FEATURE_NAMES) == 19


def test_stage2_feature_names_and_vector_have_matching_length():
    features = _router_features()
    vector = stage2_feature_vector(features, _rerank_signals())
    assert len(vector) == len(STAGE2_FEATURE_NAMES) == 26
    assert len(STAGE2_ADDITIONAL_FEATURE_NAMES) == 7


def test_stage2_feature_names_is_stage1_names_plus_additional_names_in_order():
    assert STAGE2_FEATURE_NAMES == STAGE1_FEATURE_NAMES + STAGE2_ADDITIONAL_FEATURE_NAMES


def test_feature_vector_order_matches_declared_names():
    """The FROZEN CONTRACT: build a RouterFeatures/RerankSignals pair where
    every field has a distinct, recognizable value, then assert each
    position in the produced vector equals the field named at that same
    position in STAGE1_FEATURE_NAMES / STAGE2_FEATURE_NAMES — i.e. name[i]
    genuinely describes vector[i], not just "same length"."""
    features = _router_features()
    rerank = _rerank_signals()

    expected_by_name = {
        "query_length_words": 6.0, "query_length_chars": 40.0, "comparison_marker_count": 1.0,
        "has_comparison_marker": 1.0, "temporal_marker_count": 2.0, "has_temporal_marker": 1.0,
        "conjunction_count": 3.0, "has_conjunction_marker": 0.0, "quoted_span_count": 4.0,
        "numeric_date_indicator_count": 5.0,
        "hybrid_top1_score": 0.11, "hybrid_top5_mean_score": 0.12, "score_gap_top1_top2": 0.13,
        "score_gap_top1_top5": 0.14, "dense_bm25_jaccard_top10": 0.15, "consensus_fraction_top5": 0.16,
        "mean_abs_rank_diff_common_docs": 1.7, "num_unique_docs_top5": 7.0, "num_unique_docs_top10": 8.0,
        "rerank_top1_score": 0.21, "rerank_top5_mean_score": 0.22, "rerank_score_gap_top1_top2": 0.23,
        "rerank_score_gap_top1_top5": 0.24, "rank_change_mean_abs": 2.5, "top5_overlap_with_hybrid": 0.6,
        "num_docs_new_in_rerank_top5": 2.0,
    }

    stage1_vector = stage1_feature_vector(features)
    for name, value in zip(STAGE1_FEATURE_NAMES, stage1_vector):
        assert value == expected_by_name[name], f"{name}: expected {expected_by_name[name]}, got {value}"

    stage2_vector = stage2_feature_vector(features, rerank)
    for name, value in zip(STAGE2_FEATURE_NAMES, stage2_vector):
        assert value == expected_by_name[name], f"{name}: expected {expected_by_name[name]}, got {value}"


def test_stage2_vector_prefix_equals_stage1_vector():
    features = _router_features()
    rerank = _rerank_signals()
    stage1_vector = stage1_feature_vector(features)
    stage2_vector = stage2_feature_vector(features, rerank)
    assert stage2_vector[: len(stage1_vector)] == stage1_vector


def test_all_vector_entries_are_floats():
    features = _router_features()
    rerank = _rerank_signals()
    assert all(isinstance(x, float) for x in stage1_feature_vector(features))
    assert all(isinstance(x, float) for x in stage2_feature_vector(features, rerank))
