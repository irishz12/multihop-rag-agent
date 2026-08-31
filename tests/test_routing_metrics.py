"""Router metrics tests — offline, hand-built confusion cases."""

from __future__ import annotations

import pytest

from mhrag.routing.metrics import (
    OVER_ROUTING_PAIRS,
    UNDER_ROUTING_PAIRS,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    under_over_routing_breakdown,
    under_over_routing_rate,
)


def test_confusion_matrix_all_correct():
    true = ["SIMPLE", "MEDIUM", "COMPLEX"]
    pred = ["SIMPLE", "MEDIUM", "COMPLEX"]
    m = confusion_matrix(true, pred)
    assert m["SIMPLE"]["SIMPLE"] == 1
    assert m["MEDIUM"]["MEDIUM"] == 1
    assert m["COMPLEX"]["COMPLEX"] == 1
    assert m["SIMPLE"]["MEDIUM"] == 0


def test_confusion_matrix_has_all_9_cells_even_when_unobserved():
    m = confusion_matrix(["SIMPLE"], ["SIMPLE"])
    assert set(m.keys()) == {"SIMPLE", "MEDIUM", "COMPLEX"}
    for t in m:
        assert set(m[t].keys()) == {"SIMPLE", "MEDIUM", "COMPLEX"}


def test_confusion_matrix_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix(["SIMPLE"], ["SIMPLE", "MEDIUM"])


def test_accuracy_perfect():
    assert accuracy(["SIMPLE", "MEDIUM"], ["SIMPLE", "MEDIUM"]) == 1.0


def test_accuracy_half():
    assert accuracy(["SIMPLE", "MEDIUM"], ["SIMPLE", "COMPLEX"]) == 0.5


def test_accuracy_requires_nonempty():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_per_class_metrics_known_confusion():
    # 2 SIMPLE (both correct), 2 MEDIUM (1 correct, 1 predicted SIMPLE), 2 COMPLEX (both correct)
    true = ["SIMPLE", "SIMPLE", "MEDIUM", "MEDIUM", "COMPLEX", "COMPLEX"]
    pred = ["SIMPLE", "SIMPLE", "MEDIUM", "SIMPLE", "COMPLEX", "COMPLEX"]
    per_class = per_class_metrics(true, pred)
    # SIMPLE: tp=2 (true SIMPLE correctly predicted), fp=1 (the misrouted MEDIUM->SIMPLE), fn=0
    assert per_class["SIMPLE"].precision == pytest.approx(2 / 3)
    assert per_class["SIMPLE"].recall == 1.0
    assert per_class["SIMPLE"].support == 2
    # MEDIUM: tp=1, fp=0, fn=1
    assert per_class["MEDIUM"].precision == 1.0
    assert per_class["MEDIUM"].recall == pytest.approx(0.5)
    assert per_class["MEDIUM"].support == 2
    # COMPLEX: perfect
    assert per_class["COMPLEX"].precision == 1.0
    assert per_class["COMPLEX"].recall == 1.0


def test_macro_f1_perfect_is_1():
    true = ["SIMPLE", "MEDIUM", "COMPLEX"]
    assert macro_f1(true, true) == 1.0


def test_macro_f1_averages_across_classes_unweighted():
    # All predictions are SIMPLE: SIMPLE has perfect recall but 1/3 precision;
    # MEDIUM/COMPLEX have 0 recall, 0 f1. macro_f1 averages these 3 f1 scores equally,
    # regardless of each class's support.
    true = ["SIMPLE", "MEDIUM", "COMPLEX"]
    pred = ["SIMPLE", "SIMPLE", "SIMPLE"]
    per_class = per_class_metrics(true, pred)
    expected_macro = sum(m.f1 for m in per_class.values()) / 3
    assert macro_f1(true, pred) == pytest.approx(expected_macro)
    assert per_class["MEDIUM"].f1 == 0.0
    assert per_class["COMPLEX"].f1 == 0.0


def test_under_routing_pairs_are_exactly_the_three_named_in_spec():
    assert UNDER_ROUTING_PAIRS == {
        ("COMPLEX", "SIMPLE"), ("COMPLEX", "MEDIUM"), ("MEDIUM", "SIMPLE"),
    }


def test_over_routing_pairs_are_exactly_the_three_named_in_spec():
    assert OVER_ROUTING_PAIRS == {
        ("SIMPLE", "MEDIUM"), ("SIMPLE", "COMPLEX"), ("MEDIUM", "COMPLEX"),
    }


def test_under_over_routing_rate_correct():
    # 1 under-route (COMPLEX->SIMPLE), 1 over-route (SIMPLE->MEDIUM), 2 correct.
    true = ["COMPLEX", "SIMPLE", "MEDIUM", "SIMPLE"]
    pred = ["SIMPLE", "MEDIUM", "MEDIUM", "SIMPLE"]
    under, over = under_over_routing_rate(true, pred)
    assert under == pytest.approx(0.25)
    assert over == pytest.approx(0.25)


def test_under_over_routing_rate_all_correct_is_zero():
    true = ["SIMPLE", "MEDIUM", "COMPLEX"]
    under, over = under_over_routing_rate(true, true)
    assert under == 0.0
    assert over == 0.0


def test_under_over_routing_breakdown_sums_to_total_misroute_rate():
    true = ["COMPLEX", "SIMPLE", "MEDIUM", "SIMPLE"]
    pred = ["SIMPLE", "MEDIUM", "MEDIUM", "SIMPLE"]
    breakdown = under_over_routing_breakdown(true, pred)
    assert breakdown["COMPLEX->SIMPLE"] == pytest.approx(0.25)
    assert breakdown["SIMPLE->MEDIUM"] == pytest.approx(0.25)
    under, over = under_over_routing_rate(true, pred)
    assert sum(breakdown.values()) == pytest.approx(under + over)


def test_under_over_routing_rate_requires_nonempty():
    with pytest.raises(ValueError):
        under_over_routing_rate([], [])
