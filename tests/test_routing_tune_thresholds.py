"""Threshold-fitting tests — offline, synthetic tune examples with a
perfectly separable signal so the fitted thresholds are checkable exactly."""

from __future__ import annotations

import pytest

from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.heuristic import classify_heuristic
from mhrag.routing.tune_thresholds import TuneExample, fit_thresholds


def _example(qa_id, route, top1_score, agreement, has_comparison=False, has_temporal=False):
    query = QueryFeatures(
        query_length_words=5, query_length_chars=30,
        comparison_marker_count=1 if has_comparison else 0, has_comparison_marker=has_comparison,
        temporal_marker_count=1 if has_temporal else 0, has_temporal_marker=has_temporal,
        conjunction_count=0, has_conjunction_marker=False, quoted_span_count=0,
        numeric_date_indicator_count=0,
    )
    retrieval = RetrievalSignals(
        hybrid_top1_score=top1_score, hybrid_top5_mean_score=top1_score,
        score_gap_top1_top2=0.0, score_gap_top1_top5=0.0,
        dense_bm25_jaccard_top10=agreement, consensus_fraction_top5=agreement,
        num_unique_docs_top5=5, num_unique_docs_top10=10, mean_abs_rank_diff_common_docs=0.0,
    )
    return TuneExample(qa_id=qa_id, features=RouterFeatures(query=query, retrieval=retrieval), oracle_route=route)


def _separable_examples() -> list[TuneExample]:
    """20 examples, cleanly separable: high score+agreement -> SIMPLE, low
    score+agreement -> COMPLEX, in between -> MEDIUM."""
    examples = []
    for i in range(7):
        examples.append(_example(f"simple_{i}", "SIMPLE", top1_score=0.08 + i * 0.001, agreement=0.9))
    for i in range(7):
        examples.append(_example(f"medium_{i}", "MEDIUM", top1_score=0.04 + i * 0.001, agreement=0.5))
    for i in range(6):
        examples.append(_example(f"complex_{i}", "COMPLEX", top1_score=0.005 + i * 0.0005, agreement=0.05))
    return examples


def test_fit_thresholds_raises_on_empty_examples():
    with pytest.raises(ValueError):
        fit_thresholds([])


def test_fitted_thresholds_achieve_perfect_accuracy_on_separable_tune_set():
    examples = _separable_examples()
    thresholds = fit_thresholds(examples)

    correct = 0
    confident = 0
    for ex in examples:
        verdict = classify_heuristic(ex.features, thresholds)
        if verdict.confident:
            confident += 1
            if verdict.route == ex.oracle_route:
                correct += 1
    assert confident > 0
    assert correct == confident  # 100% accuracy among confident calls on this clean-separated set


def test_fit_thresholds_is_deterministic():
    examples = _separable_examples()
    t1 = fit_thresholds(examples)
    t2 = fit_thresholds(examples)
    assert t1 == t2


def test_fit_thresholds_respects_min_coverage_floor():
    """A very high min_coverage on a set with genuine ambiguity should
    still find something (since some threshold reaches every example as
    'confident' at the extremes) or raise — never silently return a
    near-zero-coverage 'best'."""
    examples = _separable_examples()
    thresholds = fit_thresholds(examples, min_coverage=0.5)
    confident = sum(1 for ex in examples if classify_heuristic(ex.features, thresholds).confident)
    assert confident / len(examples) >= 0.5


def test_fit_thresholds_tolerates_contradictory_labels_by_picking_best_achievable_accuracy():
    """Two examples with identical signals but opposite oracle labels can't
    both be classified correctly — the fitter should still return SOME
    deterministic threshold set (never raise on this), just with <100%
    accuracy among confident calls."""
    examples = [
        _example("a", "SIMPLE", top1_score=0.01, agreement=0.9),
        _example("b", "COMPLEX", top1_score=0.01, agreement=0.9),  # identical signals, different label
    ]
    thresholds = fit_thresholds(examples, min_coverage=1.0)
    verdicts = [classify_heuristic(ex.features, thresholds) for ex in examples]
    assert all(v.confident for v in verdicts)  # coverage floor satisfied
    assert not all(v.route == ex.oracle_route for v, ex in zip(verdicts, examples))  # can't get both right
