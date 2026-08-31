"""Tests for the objective budget-selection rule (`select_budget`) — pure
function of synthetic `BudgetMetrics`, no live calls, no dataset needed."""

from __future__ import annotations

import pytest

from mhrag.calibration.decision import (
    MAX_COST_LATENCY_GROWTH_RATIO,
    MAX_QUALITY_REGRESSION,
    MIN_HOP2_PLUS_NEW_DOCS,
    TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD,
    BudgetMetrics,
    select_budget,
)


def _metrics(
    token_budget,
    token_budget_stop_rate=0.7,
    mean_new_unique_docs_after_hop1=0.5,
    mean_recall=0.8,
    mean_complete_evidence_rate=0.6,
    mean_cost_usd=0.01,
    mean_latency_ms=2000.0,
) -> BudgetMetrics:
    return BudgetMetrics(
        token_budget=token_budget,
        token_budget_stop_rate=token_budget_stop_rate,
        mean_new_unique_docs_after_hop1=mean_new_unique_docs_after_hop1,
        mean_recall=mean_recall,
        mean_complete_evidence_rate=mean_complete_evidence_rate,
        mean_cost_usd=mean_cost_usd,
        mean_latency_ms=mean_latency_ms,
    )


def test_single_candidate_keeps_baseline():
    decision = select_budget([_metrics(3000)])
    assert decision.selected_token_budget == 3000
    assert decision.baseline_token_budget == 3000
    assert decision.criteria_by_candidate == {}


def test_picks_smallest_qualifying_candidate_not_the_highest():
    baseline = _metrics(3000, token_budget_stop_rate=0.75)
    # 4500 qualifies on every criterion.
    good_4500 = _metrics(
        4500,
        token_budget_stop_rate=0.75 - TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD - 0.05,
        mean_new_unique_docs_after_hop1=MIN_HOP2_PLUS_NEW_DOCS + 0.1,
        mean_recall=0.8,
        mean_complete_evidence_rate=0.6,
        mean_cost_usd=0.011,
        mean_latency_ms=2100.0,
    )
    # 6000 would also qualify but must NOT be picked since 4500 already does.
    also_good_6000 = _metrics(
        6000,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
        mean_recall=0.85,
        mean_complete_evidence_rate=0.65,
        mean_cost_usd=0.012,
        mean_latency_ms=2200.0,
    )
    decision = select_budget([baseline, good_4500, also_good_6000])
    assert decision.selected_token_budget == 4500
    assert decision.baseline_token_budget == 3000
    assert decision.criteria_by_candidate[4500] == {
        "reduces_budget_stops": True,
        "meaningful_recovery": True,
        "cost_ok": True,
        "latency_ok": True,
        "recall_ok": True,
        "complete_evidence_ok": True,
    }
    # 6000 should not even need to be evaluated as "selected", but the rule
    # still records criteria only up to (and stops at) the first qualifier.
    assert 6000 not in decision.criteria_by_candidate


def test_falls_back_to_baseline_when_no_candidate_reduces_stops_enough():
    baseline = _metrics(3000, token_budget_stop_rate=0.75)
    barely_lower = _metrics(4500, token_budget_stop_rate=0.75 - TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD + 0.01)
    decision = select_budget([baseline, barely_lower])
    assert decision.selected_token_budget == 3000
    assert decision.criteria_by_candidate[4500]["reduces_budget_stops"] is False


def test_falls_back_to_baseline_when_recovery_is_negligible():
    baseline = _metrics(3000, token_budget_stop_rate=0.9)
    negligible_recovery = _metrics(
        4500,
        token_budget_stop_rate=0.1,  # plenty of stop-rate reduction
        mean_new_unique_docs_after_hop1=MIN_HOP2_PLUS_NEW_DOCS - 0.01,  # just under threshold
    )
    decision = select_budget([baseline, negligible_recovery])
    assert decision.selected_token_budget == 3000
    assert decision.criteria_by_candidate[4500]["meaningful_recovery"] is False


def test_falls_back_to_baseline_when_cost_grows_too_much():
    baseline = _metrics(3000, token_budget_stop_rate=0.9, mean_cost_usd=0.01)
    expensive = _metrics(
        4500,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
        mean_cost_usd=0.01 * MAX_COST_LATENCY_GROWTH_RATIO + 0.001,  # just over 1.5x
    )
    decision = select_budget([baseline, expensive])
    assert decision.selected_token_budget == 3000
    assert decision.criteria_by_candidate[4500]["cost_ok"] is False


def test_falls_back_to_baseline_when_latency_grows_too_much():
    baseline = _metrics(3000, token_budget_stop_rate=0.9, mean_latency_ms=2000.0)
    slow = _metrics(
        4500,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
        mean_latency_ms=2000.0 * MAX_COST_LATENCY_GROWTH_RATIO + 1.0,
    )
    decision = select_budget([baseline, slow])
    assert decision.selected_token_budget == 3000
    assert decision.criteria_by_candidate[4500]["latency_ok"] is False


def test_falls_back_to_baseline_when_recall_regresses_too_much():
    baseline = _metrics(3000, token_budget_stop_rate=0.9, mean_recall=0.8)
    worse_recall = _metrics(
        4500,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
        mean_recall=0.8 - MAX_QUALITY_REGRESSION - 0.01,
    )
    decision = select_budget([baseline, worse_recall])
    assert decision.selected_token_budget == 3000
    assert decision.criteria_by_candidate[4500]["recall_ok"] is False


def test_falls_back_to_baseline_when_complete_evidence_regresses_too_much():
    baseline = _metrics(3000, token_budget_stop_rate=0.9, mean_complete_evidence_rate=0.6)
    worse_complete = _metrics(
        4500,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
        mean_complete_evidence_rate=0.6 - MAX_QUALITY_REGRESSION - 0.01,
    )
    decision = select_budget([baseline, worse_complete])
    assert decision.selected_token_budget == 3000
    assert decision.criteria_by_candidate[4500]["complete_evidence_ok"] is False


def test_small_quality_regression_within_tolerance_still_qualifies():
    baseline = _metrics(3000, token_budget_stop_rate=0.9, mean_recall=0.8, mean_complete_evidence_rate=0.6)
    tiny_regression = _metrics(
        4500,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
        mean_recall=0.8 - MAX_QUALITY_REGRESSION,  # exactly at the tolerance boundary
        mean_complete_evidence_rate=0.6 - MAX_QUALITY_REGRESSION,
    )
    decision = select_budget([baseline, tiny_regression])
    assert decision.selected_token_budget == 4500


def test_empty_candidates_raises():
    with pytest.raises(ValueError):
        select_budget([])


def test_candidates_need_not_be_pre_sorted():
    baseline = _metrics(3000, token_budget_stop_rate=0.9)
    good_6000 = _metrics(
        6000,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
    )
    good_4500 = _metrics(
        4500,
        token_budget_stop_rate=0.1,
        mean_new_unique_docs_after_hop1=1.0,
    )
    # Passed out of order — baseline must still be identified as the smallest.
    decision = select_budget([good_6000, baseline, good_4500])
    assert decision.baseline_token_budget == 3000
    assert decision.selected_token_budget == 4500  # smallest qualifier, not 6000
