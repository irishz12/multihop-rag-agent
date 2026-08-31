"""RUNTIME learned-router inference — pure arithmetic, NO scikit-learn
dependency at inference time and NO live call of any kind (no Mantle, no
LLM). `scikit-learn` (`LogisticRegression` + `StandardScaler`) is used only
to FIT the model, in the evaluator-only
`mhrag.routing.learned_router_training` module; this module just applies
the fitted weights — standardize, dot product, sigmoid — exactly
reproducing what a fitted `sklearn.pipeline.Pipeline([("scaler",
StandardScaler()), ("clf", LogisticRegression())])` would compute for
`predict_proba`, verified in tests/test_routing_learned_router.py against
a real fitted sklearn pipeline.

A `LinearModel` is a small, fully-transparent, JSON-serializable
artifact (`results/learned_router_model.json`) — every number in it is a
scaler mean/scale, a logistic-regression coefficient, the intercept, or
the frozen decision threshold. `feature_names` pairs 1:1, in order, with
`mhrag.routing.learned_features.STAGE1_FEATURE_NAMES`/
`STAGE2_FEATURE_NAMES` — see that module's "frozen contract" docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearModel:
    feature_names: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    threshold: float  # frozen decision threshold for "sufficient" (P(sufficient) >= threshold)

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        for name, values in (
            ("scaler_mean", self.scaler_mean), ("scaler_scale", self.scaler_scale), ("coef", self.coef),
        ):
            if len(values) != n:
                raise ValueError(f"{name} has length {len(values)}, expected {n} (len(feature_names))")


def predict_proba(model: LinearModel, x: list[float]) -> float:
    if len(x) != len(model.feature_names):
        raise ValueError(f"feature vector has length {len(x)}, model expects {len(model.feature_names)}")
    z = model.intercept
    for xi, mean, scale, coef in zip(x, model.scaler_mean, model.scaler_scale, model.coef):
        standardized = (xi - mean) / scale if scale != 0 else 0.0
        z += coef * standardized
    return 1.0 / (1.0 + math.exp(-z))


def predict_sufficient(model: LinearModel, x: list[float]) -> tuple[bool, float]:
    """Returns (sufficient, probability). `sufficient=True` iff
    `probability >= model.threshold` — the frozen threshold IS the
    "prefer escalation when uncertain" policy baked in: it is chosen (see
    `mhrag.routing.learned_router_training.select_thresholds`) to require
    genuine confidence before declaring sufficiency, not just `>= 0.5`."""
    probability = predict_proba(model, x)
    return probability >= model.threshold, probability
