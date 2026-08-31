"""Stage A heuristic router tests — offline, pure-function, no models."""

from __future__ import annotations

from pathlib import Path

from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.heuristic import DEFAULT_THRESHOLDS, HeuristicThresholds, classify_heuristic

THRESHOLDS = HeuristicThresholds(
    simple_min_top1_score=0.03,
    simple_min_agreement=0.5,
    complex_max_top1_score=0.015,
    complex_max_agreement=0.2,
    medium_min_top1_score=0.02,
)


def _features(
    top1_score=0.025, agreement=0.35, has_comparison=False, has_temporal=False,
) -> RouterFeatures:
    query = QueryFeatures(
        query_length_words=6, query_length_chars=40,
        comparison_marker_count=1 if has_comparison else 0, has_comparison_marker=has_comparison,
        temporal_marker_count=1 if has_temporal else 0, has_temporal_marker=has_temporal,
        conjunction_count=0, has_conjunction_marker=False,
        quoted_span_count=0, numeric_date_indicator_count=0,
    )
    retrieval = RetrievalSignals(
        hybrid_top1_score=top1_score, hybrid_top5_mean_score=top1_score * 0.8,
        score_gap_top1_top2=0.005, score_gap_top1_top5=0.01,
        dense_bm25_jaccard_top10=agreement, consensus_fraction_top5=agreement,
        num_unique_docs_top5=4, num_unique_docs_top10=8, mean_abs_rank_diff_common_docs=1.0,
    )
    return RouterFeatures(query=query, retrieval=retrieval)


def test_simple_when_high_score_high_agreement():
    f = _features(top1_score=0.05, agreement=0.8)
    v = classify_heuristic(f, THRESHOLDS)
    assert v.route == "SIMPLE"
    assert v.confident is True


def test_simple_regardless_of_comparison_or_temporal_marker():
    """Stage A deliberately does NOT gate SIMPLE on lexical markers — see
    heuristic.py's "DESIGN NOTE" (the marker-implies-harder assumption was
    measured to be backwards on the real oracle distribution)."""
    f_marked = _features(top1_score=0.05, agreement=0.8, has_comparison=True, has_temporal=True)
    v = classify_heuristic(f_marked, THRESHOLDS)
    assert v.route == "SIMPLE"
    assert v.confident is True


def test_complex_when_low_score_and_low_agreement():
    f = _features(top1_score=0.01, agreement=0.1)
    v = classify_heuristic(f, THRESHOLDS)
    assert v.route == "COMPLEX"
    assert v.confident is True


def test_complex_regardless_of_temporal_marker_when_score_and_agreement_are_weak():
    f = _features(top1_score=0.01, agreement=0.1, has_temporal=True)
    v = classify_heuristic(f, THRESHOLDS)
    assert v.route == "COMPLEX"
    assert v.confident is True


def test_medium_when_moderate_score():
    f = _features(top1_score=0.022, agreement=0.3)
    v = classify_heuristic(f, THRESHOLDS)
    assert v.route == "MEDIUM"
    assert v.confident is True


def test_medium_regardless_of_comparison_marker():
    f = _features(top1_score=0.022, agreement=0.3, has_comparison=True)
    v = classify_heuristic(f, THRESHOLDS)
    assert v.route == "MEDIUM"
    assert v.confident is True


def test_ambiguous_falls_through_to_none_not_confident():
    # top1 score below medium bar, agreement above complex bar -> no rule matches.
    f = _features(top1_score=0.005, agreement=0.3)
    v = classify_heuristic(f, THRESHOLDS)
    assert v.route is None
    assert v.confident is False


def test_heuristic_is_deterministic_repeated_calls():
    f = _features(top1_score=0.05, agreement=0.8)
    results = [classify_heuristic(f, THRESHOLDS) for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_default_thresholds_never_confident_fail_safe():
    """An uncalibrated default must never make a confident call — every
    query defers to Stage B (GLM) until real thresholds are tuned and
    frozen."""
    for f in (
        _features(top1_score=1.0, agreement=1.0),
        _features(top1_score=-1.0, agreement=0.0),
        _features(top1_score=0.02, agreement=0.5),
    ):
        v = classify_heuristic(f, DEFAULT_THRESHOLDS)
        assert v.confident is False
        assert v.route is None


def test_heuristic_module_never_imports_oracle():
    """Structural: Stage A must have no access to gold route labels (prose
    mentioning "oracle" in comments/docstrings is fine — no actual import
    statement referencing mhrag.routing.oracle is allowed)."""
    import re

    path = Path(__file__).parent.parent / "src" / "mhrag" / "routing" / "heuristic.py"
    source = path.read_text()
    assert not re.search(r"^\s*(import|from)\s+.*\boracle\b", source, re.MULTILINE)
