"""Learned-router training/calibration tests — offline, synthetic data.
Covers: proper out-of-fold coverage (no row predicted by a model that saw
its own label), the Stage 2 dual-population OOF composition, final-model
weight extraction, strongest-coefficient ranking, and threshold selection
under the under-routing constraint with the cost-minimizing / prefer-
escalation-when-uncertain tie-break.
"""

from __future__ import annotations

import numpy as np
import pytest

from mhrag.routing.cost_projection import UnitCost
from mhrag.routing.learned_router import predict_proba
from mhrag.routing.learned_router_training import (
    fit_final_model,
    run_stratified_cv,
    select_thresholds,
    stage2_oof_for_all,
    strongest_coefficients,
)


def _separable_dataset(n=100, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(loc=[0, 0], scale=[1, 1], size=(n, 2))
    z = 2.0 * X[:, 0] - 1.0 * X[:, 1]
    y = (z + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return X, y


# --- run_stratified_cv ---------------------------------------------------------------------


def test_oof_covers_every_row_with_valid_probabilities():
    X, y = _separable_dataset()
    oof = run_stratified_cv(X, y)
    assert oof.shape == (len(y),)
    assert not np.isnan(oof).any()
    assert ((oof >= 0) & (oof <= 1)).all()


def test_oof_is_deterministic_across_calls():
    X, y = _separable_dataset()
    oof1 = run_stratified_cv(X, y)
    oof2 = run_stratified_cv(X, y)
    np.testing.assert_array_equal(oof1, oof2)


def test_oof_has_real_discriminative_signal_on_separable_data():
    X, y = _separable_dataset(n=200)
    oof = run_stratified_cv(X, y)
    predicted = (oof >= 0.5).astype(int)
    accuracy = (predicted == y).mean()
    assert accuracy > 0.8  # clearly-separable synthetic data should CV well


# --- stage2_oof_for_all ----------------------------------------------------------------------


def test_stage2_oof_covers_all_rows_including_undefined_population():
    X, y = _separable_dataset(n=150)
    mask = np.zeros(150, dtype=bool)
    mask[:100] = True  # first 100 rows are the "defined" (non-SIMPLE) population
    y_defined = y[:100]
    result = stage2_oof_for_all(X, mask, y_defined)
    assert result.shape == (150,)
    assert not np.isnan(result).any()
    assert ((result >= 0) & (result <= 1)).all()


def test_stage2_oof_defined_subset_differs_from_naive_full_fit_predict():
    """Sanity: the defined subset's OOF values come from proper CV (not
    simply predict_proba on data the model was fit on) — spot check they
    aren't suspiciously identical to a full in-sample fit's probabilities
    for every single row (which would indicate the CV loop is a no-op)."""
    X, y = _separable_dataset(n=150, seed=3)
    mask = np.ones(150, dtype=bool)
    result_oof = stage2_oof_for_all(X, mask, y)

    from mhrag.routing.learned_router_training import _make_pipeline

    in_sample_pipeline = _make_pipeline()
    in_sample_pipeline.fit(X, y)
    in_sample_proba = in_sample_pipeline.predict_proba(X)[:, 1]

    # Not every value should match exactly — OOF predictions come from folds
    # that excluded each row, so at least some should differ from the full fit.
    assert not np.allclose(result_oof, in_sample_proba)


# --- fit_final_model / strongest_coefficients ------------------------------------------------


def test_fit_final_model_matches_manual_predict_proba():
    X, y = _separable_dataset(n=100)
    model = fit_final_model(X, y, feature_names=("x1", "x2"), threshold=0.5)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42, max_iter=1000))])
    pipeline.fit(X, y)

    for point in X[:10]:
        sklearn_p = float(pipeline.predict_proba(point.reshape(1, -1))[0][1])
        our_p = predict_proba(model, list(point))
        assert abs(sklearn_p - our_p) < 1e-9


def test_strongest_coefficients_sorted_by_absolute_value_descending():
    X, y = _separable_dataset(n=100)  # x1 has coefficient magnitude 2x that of x2 by construction
    model = fit_final_model(X, y, feature_names=("x1", "x2"), threshold=0.5)
    ranked = strongest_coefficients(model, top_n=2)
    assert ranked[0][0] == "x1"  # the stronger true signal
    assert abs(ranked[0][1]) >= abs(ranked[1][1])


def test_strongest_coefficients_respects_top_n():
    X, y = _separable_dataset(n=100)
    model = fit_final_model(X, y, feature_names=("x1", "x2"), threshold=0.5)
    assert len(strongest_coefficients(model, top_n=1)) == 1


# --- select_thresholds -----------------------------------------------------------------------


_UNIT_COSTS = {
    "hybrid_only": UnitCost(cost_usd=0.0, latency_ms=60.0),
    "hybrid_reranker": UnitCost(cost_usd=0.0, latency_ms=3000.0),
    "agentic": UnitCost(cost_usd=0.001, latency_ms=8500.0),
}


def test_select_thresholds_respects_under_routing_constraint():
    # 10 examples: perfectly predictable via probabilities matching the oracle exactly.
    oracle = ["SIMPLE"] * 4 + ["MEDIUM"] * 3 + ["COMPLEX"] * 3
    oof_p1 = np.array([0.95, 0.95, 0.95, 0.95, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    oof_p2 = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    result = select_thresholds(oof_p1, oof_p2, oracle, _UNIT_COSTS, max_under_routing=0.10)
    assert result.under_routing_rate <= 0.10


def test_select_thresholds_perfect_signal_achieves_zero_under_routing():
    oracle = ["SIMPLE"] * 4 + ["MEDIUM"] * 3 + ["COMPLEX"] * 3
    oof_p1 = np.array([0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02])
    oof_p2 = np.array([0.5, 0.5, 0.5, 0.5, 0.95, 0.95, 0.95, 0.02, 0.02, 0.02])
    result = select_thresholds(oof_p1, oof_p2, oracle, _UNIT_COSTS, max_under_routing=0.10)
    assert result.under_routing_rate == 0.0
    assert result.accuracy == 1.0


def test_select_thresholds_raises_when_no_pair_satisfies_constraint():
    # Every example is COMPLEX but every probability signals SIMPLE -> impossible to keep
    # under-routing at 0% no matter the thresholds (COMPLEX->SIMPLE is unavoidable at high p1).
    oracle = ["COMPLEX"] * 5
    oof_p1 = np.array([0.99, 0.99, 0.99, 0.99, 0.99])
    oof_p2 = np.array([0.99, 0.99, 0.99, 0.99, 0.99])
    with pytest.raises(ValueError, match="no \\(tau1, tau2\\)"):
        select_thresholds(
            oof_p1, oof_p2, oracle, _UNIT_COSTS, max_under_routing=0.0,
            candidate_thresholds=np.array([0.5]),  # tau1=0.5 always escalates SIMPLE incorrectly here -> 100% under-routing
        )


def test_select_thresholds_prefers_lower_cost_among_valid_pairs():
    # All examples SIMPLE-labeled with a clear high p1 -> the cheapest valid solution
    # should predict SIMPLE for (nearly) everyone, keeping cost near zero.
    oracle = ["SIMPLE"] * 10
    oof_p1 = np.full(10, 0.9)
    oof_p2 = np.full(10, 0.9)
    result = select_thresholds(oof_p1, oof_p2, oracle, _UNIT_COSTS, max_under_routing=0.10)
    assert result.projected_cost_usd == 0.0  # everyone routed SIMPLE -> agentic backend cost is 0


def test_select_thresholds_tie_break_prefers_more_conservative_pair():
    """When multiple (tau1, tau2) combinations achieve identical cost and
    under-routing, the more conservative (higher tau1+tau2) one wins."""
    oracle = ["SIMPLE"] * 5 + ["COMPLEX"] * 5
    oof_p1 = np.array([0.99] * 5 + [0.01] * 5)
    oof_p2 = np.array([0.5] * 10)
    candidates = np.array([0.3, 0.6, 0.9])
    result = select_thresholds(
        oof_p1, oof_p2, oracle, _UNIT_COSTS, max_under_routing=0.10, candidate_thresholds=candidates,
    )
    # Every tau1 in {0.3,0.6,0.9} correctly separates 0.99 from 0.01 with identical routing/cost outcomes
    # here (since p1 values are far from all three candidate thresholds) -> must pick the highest (0.9).
    assert result.stage1_threshold == 0.9
