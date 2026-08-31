"""EVALUATOR-ONLY: fit and calibrate the learned two-stage router (Phase
8A.2) — 5-fold stratified CV, out-of-fold (OOF) probabilities, threshold
selection under an under-routing constraint. Uses oracle route labels
(evaluator-only ground truth) — must NEVER be imported by any runtime
router module (`mhrag.routing.learned_router`,
`mhrag.routing.learned_sequential_router`, `mhrag.routing.
learned_features`, `mhrag.routing.rerank_features`); see
tests/test_routing_no_gold_leakage.py.

`scikit-learn` (`LogisticRegression` + `StandardScaler`, composed via a
`Pipeline`) is used ONLY here, to FIT models — the runtime path
(`mhrag.routing.learned_router.predict_proba`) never imports scikit-learn,
it only applies the already-fitted weights `fit_final_model` extracts into
a plain, JSON-serializable `LinearModel`.

STAGE 2 POPULATION NOTE: Stage 2's natural population is the
oracle-non-SIMPLE subset (MEDIUM/COMPLEX questions — Stage 2 never runs in
the real pipeline for a question Stage 1 already resolved). Proper 5-fold
stratified CV is run on exactly that subset to produce honest OOF
probabilities for it. For the REMAINING (oracle-SIMPLE) questions —
included because `select_thresholds` needs to simulate what a candidate
Stage 1 threshold that WRONGLY escalates a SIMPLE question would do at
Stage 2 — `stage2_oof_for_all` predicts them from a model fit on the ENTIRE
non-SIMPLE subset; those rows were never part of the non-SIMPLE training
population at all, so this is a different (out-of-population) but equally
valid sense of "the model never saw this row's label" — not the same
mechanism as the non-SIMPLE subset's fold-based OOF, and documented
distinctly rather than conflated with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mhrag.routing.cost_projection import UnitCost, project_workload
from mhrag.routing.learned_router import LinearModel
from mhrag.routing.metrics import accuracy as route_accuracy
from mhrag.routing.metrics import under_over_routing_rate

CV_SEED = 42
N_SPLITS = 5


def _make_pipeline(seed: int = CV_SEED) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
    ])


def run_stratified_cv(X: np.ndarray, y: np.ndarray, seed: int = CV_SEED, n_splits: int = N_SPLITS) -> np.ndarray:
    """Returns OOF predicted P(y=1) for every row of `X`. StandardScaler +
    LogisticRegression are fit fresh on each fold's TRAIN portion only
    (inside a `Pipeline`, so scaling never sees the held-out fold), then
    applied to that fold's held-out rows. Every row receives exactly one
    OOF prediction, from a model that never saw that row's label."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        pipeline = _make_pipeline(seed)
        pipeline.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
    if np.isnan(oof).any():
        raise AssertionError("not every row received an OOF prediction — StratifiedKFold coverage bug")
    return oof


def stage2_oof_for_all(
    X_all: np.ndarray,
    stage2_defined_mask: np.ndarray,
    y_stage2_defined: np.ndarray,
    seed: int = CV_SEED,
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """See module docstring's "STAGE 2 POPULATION NOTE". `y_stage2_defined`
    is already aligned to (same order/length as) `X_all[stage2_defined_mask]`.
    Returns one probability per row of `X_all`, in `X_all`'s original order.
    """
    n = X_all.shape[0]
    result = np.full(n, np.nan)
    idx_defined = np.where(stage2_defined_mask)[0]
    idx_undefined = np.where(~stage2_defined_mask)[0]

    X_defined = X_all[idx_defined]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_local, test_local in skf.split(X_defined, y_stage2_defined):
        pipeline = _make_pipeline(seed)
        pipeline.fit(X_defined[train_local], y_stage2_defined[train_local])
        result[idx_defined[test_local]] = pipeline.predict_proba(X_defined[test_local])[:, 1]

    if len(idx_undefined) > 0:
        full_fit = _make_pipeline(seed)
        full_fit.fit(X_defined, y_stage2_defined)
        result[idx_undefined] = full_fit.predict_proba(X_all[idx_undefined])[:, 1]

    if np.isnan(result).any():
        raise AssertionError("not every row received a stage-2 probability")
    return result


def fit_final_model(
    X: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...], threshold: float, seed: int = CV_SEED,
) -> LinearModel:
    """Refit ONE final Pipeline on ALL of `X`/`y` (standard practice: CV is
    for unbiased performance estimation and threshold selection; the
    deployed model is refit on the full dataset afterward). Extracts the
    fitted `StandardScaler`/`LogisticRegression` weights into a plain,
    JSON-serializable `LinearModel` — scikit-learn is not needed to USE
    this artifact, only to have produced it."""
    pipeline = _make_pipeline(seed)
    pipeline.fit(X, y)
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    return LinearModel(
        feature_names=tuple(feature_names),
        scaler_mean=tuple(float(m) for m in scaler.mean_),
        scaler_scale=tuple(float(s) for s in scaler.scale_),
        coef=tuple(float(c) for c in clf.coef_[0]),
        intercept=float(clf.intercept_[0]),
        threshold=float(threshold),
    )


def strongest_coefficients(model: LinearModel, top_n: int = 5) -> list[tuple[str, float]]:
    """Top-`top_n` features by |standardized coefficient| — comparable in
    magnitude across features precisely because `StandardScaler` puts
    every feature on the same (mean-0, unit-variance) scale before the
    logistic regression sees it."""
    pairs = sorted(zip(model.feature_names, model.coef), key=lambda p: -abs(p[1]))
    return pairs[:top_n]


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    stage1_threshold: float
    stage2_threshold: float
    under_routing_rate: float
    over_routing_rate: float
    accuracy: float
    projected_cost_usd: float
    projected_latency_ms: float


def simulate_routes(oof_p1: np.ndarray, oof_p2: np.ndarray, tau1: float, tau2: float) -> list[str]:
    routes = []
    for p1, p2 in zip(oof_p1, oof_p2):
        if p1 >= tau1:
            routes.append("SIMPLE")
        elif p2 >= tau2:
            routes.append("MEDIUM")
        else:
            routes.append("COMPLEX")
    return routes


def select_thresholds(
    oof_p1: np.ndarray,
    oof_p2: np.ndarray,
    oracle_routes: list[str],
    unit_costs: dict[str, UnitCost],
    max_under_routing: float = 0.10,
    candidate_thresholds: np.ndarray | None = None,
) -> ThresholdSelection:
    """Grid search over (tau1, tau2) pairs, using OOF probabilities only —
    never final-model in-sample predictions. Objective: among every pair
    with `under_routing_rate <= max_under_routing`, pick the one with the
    LOWEST projected cost (`mhrag.routing.cost_projection.project_workload`
    on the simulated route distribution). Ties broken, in order: (1) lower
    under-routing rate, (2) higher `tau1 + tau2` — i.e. the MORE
    conservative pair, which escalates more readily under genuine
    uncertainty ("prefer escalation when uncertain")."""
    if candidate_thresholds is None:
        candidate_thresholds = np.round(np.arange(0.05, 1.00, 0.01), 4)

    best_key: tuple[float, float, float] | None = None
    best: ThresholdSelection | None = None

    for tau1 in candidate_thresholds:
        for tau2 in candidate_thresholds:
            predicted = simulate_routes(oof_p1, oof_p2, tau1, tau2)
            under, over = under_over_routing_rate(oracle_routes, predicted)
            if under > max_under_routing:
                continue

            route_counts = {r: predicted.count(r) for r in ("SIMPLE", "MEDIUM", "COMPLEX")}
            projection = project_workload(route_counts, unit_costs)
            key = (projection.total_cost_usd, under, -(tau1 + tau2))
            if best_key is None or key < best_key:
                best_key = key
                best = ThresholdSelection(
                    stage1_threshold=float(tau1), stage2_threshold=float(tau2),
                    under_routing_rate=under, over_routing_rate=over,
                    accuracy=route_accuracy(oracle_routes, predicted),
                    projected_cost_usd=projection.total_cost_usd,
                    projected_latency_ms=projection.mean_latency_ms,
                )

    if best is None:
        raise ValueError(
            f"no (tau1, tau2) combination reaches under_routing_rate <= {max_under_routing:.0%} — "
            "widen candidate_thresholds or relax max_under_routing"
        )
    return best
