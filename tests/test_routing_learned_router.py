"""Learned-router pure-inference tests — offline, no live call. Verifies
the hand-rolled standardize+dot+sigmoid arithmetic exactly reproduces a
real fitted sklearn Pipeline(StandardScaler, LogisticRegression)'s
predict_proba, plus threshold/shape validation and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mhrag.routing.learned_router import LinearModel, predict_proba, predict_sufficient


def _toy_dataset():
    rng = np.random.RandomState(42)
    X = rng.normal(loc=[5, -2, 0.3], scale=[2, 1, 0.05], size=(200, 3))
    # Centered so roughly half the points fall on each side of 0 -> both classes present.
    z = 0.8 * (X[:, 0] - 5) - 1.5 * (X[:, 1] + 2) + 20 * (X[:, 2] - 0.3)
    y = (z + rng.normal(scale=1.0, size=200) > 0).astype(int)
    return X, y


def _fit_sklearn_pipeline():
    X, y = _toy_dataset()
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipeline.fit(X, y)
    return pipeline


def _linear_model_from_pipeline(pipeline, threshold=0.5) -> LinearModel:
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    return LinearModel(
        feature_names=("a", "b", "c"),
        scaler_mean=tuple(float(m) for m in scaler.mean_),
        scaler_scale=tuple(float(s) for s in scaler.scale_),
        coef=tuple(float(c) for c in clf.coef_[0]),
        intercept=float(clf.intercept_[0]),
        threshold=threshold,
    )


def test_predict_proba_matches_real_fitted_sklearn_pipeline():
    pipeline = _fit_sklearn_pipeline()
    model = _linear_model_from_pipeline(pipeline)

    test_points = [[5.0, -2.0, 0.3], [8.0, 0.0, 0.4], [1.0, -4.0, 0.1], [3.5, -1.5, 0.35]]
    for point in test_points:
        sklearn_proba = pipeline.predict_proba(np.array([point]))[0][1]
        our_proba = predict_proba(model, point)
        assert abs(sklearn_proba - our_proba) < 1e-9, f"mismatch at {point}: sklearn={sklearn_proba} ours={our_proba}"


def test_predict_sufficient_uses_frozen_threshold_not_hardcoded_half():
    pipeline = _fit_sklearn_pipeline()
    point = [5.0, -2.0, 0.3]
    sklearn_proba = float(pipeline.predict_proba(np.array([point]))[0][1])

    lenient_model = _linear_model_from_pipeline(pipeline, threshold=max(0.0, sklearn_proba - 0.1))
    strict_model = _linear_model_from_pipeline(pipeline, threshold=min(1.0, sklearn_proba + 0.1))

    lenient_sufficient, p1 = predict_sufficient(lenient_model, point)
    strict_sufficient, p2 = predict_sufficient(strict_model, point)
    assert bool(lenient_sufficient) is True
    assert bool(strict_sufficient) is False
    assert p1 == p2  # same probability, different threshold


def test_wrong_length_feature_vector_raises():
    model = LinearModel(
        feature_names=("a", "b"), scaler_mean=(0.0, 0.0), scaler_scale=(1.0, 1.0),
        coef=(1.0, 1.0), intercept=0.0, threshold=0.5,
    )
    with pytest.raises(ValueError):
        predict_proba(model, [1.0])


def test_mismatched_construction_lengths_raise():
    with pytest.raises(ValueError):
        LinearModel(
            feature_names=("a", "b"), scaler_mean=(0.0,), scaler_scale=(1.0, 1.0),
            coef=(1.0, 1.0), intercept=0.0, threshold=0.5,
        )


def test_zero_scale_treated_as_no_op_not_divide_by_zero():
    model = LinearModel(
        feature_names=("a",), scaler_mean=(5.0,), scaler_scale=(0.0,), coef=(2.0,), intercept=0.0, threshold=0.5,
    )
    # Should not raise ZeroDivisionError; standardized contribution treated as 0.
    proba = predict_proba(model, [5.0])
    assert proba == pytest.approx(0.5)  # sigmoid(0) == 0.5


def test_deterministic_repeated_calls():
    model = LinearModel(
        feature_names=("a", "b"), scaler_mean=(1.0, 2.0), scaler_scale=(1.0, 1.0),
        coef=(0.5, -0.3), intercept=0.1, threshold=0.6,
    )
    results = [predict_sufficient(model, [3.0, 1.0]) for _ in range(5)]
    assert all(r == results[0] for r in results)
