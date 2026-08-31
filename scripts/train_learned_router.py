#!/usr/bin/env python
"""Phase 8A.2: OFFLINE training/calibration of the learned two-stage
router — no live calls, no retrieval, no Mantle. Reads
results/learned_router_dataset.json (all 265 non-null DEVELOPMENT
questions' query/Hybrid/rerank features + oracle route label, built by
scripts/build_learned_router_dataset.py).

Pipeline:
  1. Build Stage 1's feature matrix (265x19, all questions) and target
     (1{oracle_route == "SIMPLE"}).
  2. Build Stage 2's feature matrix (265x26, all questions — the dataset's
     rerank signals were computed UNCONDITIONALLY) and its natural target,
     defined only for the oracle-non-SIMPLE (MEDIUM/COMPLEX, 152-question)
     population: 1{oracle_route == "MEDIUM"}.
  3. `run_stratified_cv`/`stage2_oof_for_all` (5-fold stratified CV, seed
     42, preprocessing+fitting inside each fold) produce out-of-fold P(y=1)
     for every one of the 265 questions on BOTH stages.
  4. `select_thresholds` grid-searches (tau1, tau2) using ONLY the OOF
     probabilities, under the under_routing_rate<=10% constraint,
     minimizing projected cost with a prefer-escalation-when-uncertain tie
     break (unit costs sourced from results/retrieval_eval_development.json
     + results/agentic_budget_calibration_decision.json — same sourcing
     pattern as scripts/analyze_sequential_router_eval.py).
  5. `fit_final_model` refits ONE Stage 1 model on all 265 rows and ONE
     Stage 2 model on the natural 152-row population, at the chosen
     thresholds, and extracts each into a plain, JSON-serializable
     `LinearModel` — the artifact `mhrag.routing.learned_sequential_router`
     actually loads at inference time.
  6. Performance is reported from the OOF-simulated routes (never from the
     final in-sample models applied to their own training data, which
     would look artificially good) — an honest cross-validated estimate.

Does NOT write to, read from, or otherwise reference any prior Phase
8A/8A.1 output file — see tests/test_train_learned_router_guard.py.

Usage:
    python scripts/train_learned_router.py

Writes results/learned_router_model.json and results/learned_router_report.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from mhrag.config import PROJECT_ROOT
from mhrag.routing.cost_projection import UnitCost, agentic_multi_hop_projection, project_workload
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.learned_features import STAGE1_FEATURE_NAMES, STAGE2_FEATURE_NAMES, stage1_feature_vector, stage2_feature_vector
from mhrag.routing.learned_router_training import (
    fit_final_model,
    run_stratified_cv,
    select_thresholds,
    simulate_routes,
    stage2_oof_for_all,
    strongest_coefficients,
)
from mhrag.routing.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    under_over_routing_breakdown,
    under_over_routing_rate,
)
from mhrag.routing.rerank_features import RerankSignals

DATASET_PATH = "results/learned_router_dataset.json"
MODEL_OUTPUT_PATH = "results/learned_router_model.json"
REPORT_OUTPUT_PATH = "results/learned_router_report.json"


def _router_features_from_record(record: dict) -> RouterFeatures:
    return RouterFeatures(
        query=QueryFeatures(**record["query_features"]),
        retrieval=RetrievalSignals(**record["retrieval_signals"]),
    )


def _rerank_signals_from_record(record: dict) -> RerankSignals:
    return RerankSignals(**record["rerank_signals"])


def _metrics_block(true_routes: list[str], predicted_routes: list[str]) -> dict:
    per_class = per_class_metrics(true_routes, predicted_routes)
    under, over = under_over_routing_rate(true_routes, predicted_routes)
    return {
        "accuracy": accuracy(true_routes, predicted_routes),
        "macro_f1": macro_f1(true_routes, predicted_routes),
        "per_class_metrics": {
            k: {"precision": v.precision, "recall": v.recall, "f1": v.f1, "support": v.support}
            for k, v in per_class.items()
        },
        "confusion_matrix": confusion_matrix(true_routes, predicted_routes),
        "under_routing_rate": under,
        "over_routing_rate": over,
        "under_over_routing_breakdown": under_over_routing_breakdown(true_routes, predicted_routes),
    }


def main() -> None:
    dataset_path = PROJECT_ROOT / DATASET_PATH
    dataset = json.loads(dataset_path.read_text())
    records = dataset["records"]
    print(f"Loaded {len(records)} questions from {dataset_path}")
    if len(records) != 265:
        print(f"  WARNING: expected 265, got {len(records)}")

    oracle_routes = [r["oracle_route"] for r in records]

    # --- Stage 1: all 265 questions -----------------------------------------------------------
    X1 = np.array([stage1_feature_vector(_router_features_from_record(r)) for r in records])
    y1 = np.array([1 if r["oracle_route"] == "SIMPLE" else 0 for r in records])
    print(f"\nStage 1 feature matrix: {X1.shape}, positive (SIMPLE) rate: {y1.mean():.1%}")

    oof_p1 = run_stratified_cv(X1, y1)

    # --- Stage 2: all 265 questions get a feature row (rerank signals were computed
    # unconditionally); the natural (oracle-non-SIMPLE) population defines the target. ---------
    X2_all = np.array([
        stage2_feature_vector(_router_features_from_record(r), _rerank_signals_from_record(r))
        for r in records
    ])
    stage2_defined_mask = np.array([r["oracle_route"] != "SIMPLE" for r in records])
    y2_defined = np.array([1 if r["oracle_route"] == "MEDIUM" else 0 for r in records])[stage2_defined_mask]
    print(f"Stage 2 feature matrix: {X2_all.shape}, natural (non-SIMPLE) population: "
          f"{stage2_defined_mask.sum()}, positive (MEDIUM) rate within it: {y2_defined.mean():.1%}")

    oof_p2 = stage2_oof_for_all(X2_all, stage2_defined_mask, y2_defined)

    # --- threshold selection (OOF probabilities only) -----------------------------------------
    retrieval_eval = json.loads((PROJECT_ROOT / "results" / "retrieval_eval_development.json").read_text())
    hybrid_latency = retrieval_eval["latency_ms"]["hybrid_retrieval"]["mean"]
    hybrid_reranker_latency = retrieval_eval["latency_ms"]["hybrid_reranker_total"]["mean"]
    agentic_calib = json.loads((PROJECT_ROOT / "results" / "agentic_budget_calibration_decision.json").read_text())
    agentic_metrics = agentic_calib["metrics_by_budget"]["4500"]
    unit_costs = {
        "hybrid_only": UnitCost(cost_usd=0.0, latency_ms=hybrid_latency),
        "hybrid_reranker": UnitCost(cost_usd=0.0, latency_ms=hybrid_reranker_latency),
        "agentic": UnitCost(cost_usd=agentic_metrics["mean_cost_usd"], latency_ms=agentic_metrics["mean_latency_ms"]),
    }

    selection = select_thresholds(oof_p1, oof_p2, oracle_routes, unit_costs, max_under_routing=0.10)
    print(f"\nSelected thresholds: tau1={selection.stage1_threshold:.2f}, tau2={selection.stage2_threshold:.2f}")
    print(f"  under_routing_rate={selection.under_routing_rate:.1%}, "
          f"over_routing_rate={selection.over_routing_rate:.1%}, accuracy={selection.accuracy:.1%}")
    print(f"  projected_cost_usd={selection.projected_cost_usd:.4f}, "
          f"projected_latency_ms={selection.projected_latency_ms:.1f}")

    # --- honest, out-of-sample performance from the OOF-simulated routes at the chosen
    # thresholds (never from the final in-sample models applied to their own training data) ----
    predicted_routes = simulate_routes(oof_p1, oof_p2, selection.stage1_threshold, selection.stage2_threshold)
    learned_router_metrics = _metrics_block(oracle_routes, predicted_routes)

    predicted_route_counts = {r: predicted_routes.count(r) for r in ("SIMPLE", "MEDIUM", "COMPLEX")}
    routed_projection = project_workload(predicted_route_counts, unit_costs)
    agentic_multi_hop = agentic_multi_hop_projection(len(records), unit_costs["agentic"])

    print(f"\nLearned router (OOF): accuracy={learned_router_metrics['accuracy']:.1%} "
          f"macro_f1={learned_router_metrics['macro_f1']:.3f}")
    print(f"Predicted distribution: {predicted_route_counts}")
    print(f"Routed cost=${routed_projection.total_cost_usd:.4f} vs Agentic Multi-Hop RAG=${agentic_multi_hop.total_cost_usd:.4f}")

    # --- fit the FINAL deployable models on the full dataset ----------------------------------
    stage1_model = fit_final_model(X1, y1, STAGE1_FEATURE_NAMES, threshold=selection.stage1_threshold)
    X2_defined = X2_all[stage2_defined_mask]
    stage2_model = fit_final_model(X2_defined, y2_defined, STAGE2_FEATURE_NAMES, threshold=selection.stage2_threshold)

    stage1_top_coef = strongest_coefficients(stage1_model, top_n=5)
    stage2_top_coef = strongest_coefficients(stage2_model, top_n=5)
    print(f"\nStage 1 strongest coefficients: {stage1_top_coef}")
    print(f"Stage 2 strongest coefficients: {stage2_top_coef}")

    # --- router runtime cost/latency: pure arithmetic, $0 marginal cost, sub-millisecond -------
    import time

    from mhrag.routing.learned_router import predict_sufficient

    n_timing_trials = 1000
    t0 = time.monotonic()
    for x in X1[: min(n_timing_trials, len(X1))]:
        predict_sufficient(stage1_model, list(x))
    stage1_decision_latency_ms = (time.monotonic() - t0) * 1000 / min(n_timing_trials, len(X1))

    t0 = time.monotonic()
    for x in X2_defined[: min(n_timing_trials, len(X2_defined))]:
        predict_sufficient(stage2_model, list(x))
    stage2_decision_latency_ms = (time.monotonic() - t0) * 1000 / min(n_timing_trials, len(X2_defined))

    model_artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A.2 learned two-stage router — frozen deployable LinearModels + thresholds",
        "cv_seed": 42,
        "n_splits": 5,
        "stage1": {
            "feature_names": list(stage1_model.feature_names),
            "scaler_mean": list(stage1_model.scaler_mean),
            "scaler_scale": list(stage1_model.scaler_scale),
            "coef": list(stage1_model.coef),
            "intercept": stage1_model.intercept,
            "threshold": stage1_model.threshold,
            "trained_on_n_questions": len(records),
        },
        "stage2": {
            "feature_names": list(stage2_model.feature_names),
            "scaler_mean": list(stage2_model.scaler_mean),
            "scaler_scale": list(stage2_model.scaler_scale),
            "coef": list(stage2_model.coef),
            "intercept": stage2_model.intercept,
            "threshold": stage2_model.threshold,
            "trained_on_n_questions": int(stage2_defined_mask.sum()),
        },
    }
    model_out_path = PROJECT_ROOT / MODEL_OUTPUT_PATH
    model_out_path.write_text(json.dumps(model_artifact, indent=2))
    print(f"\nWrote {model_out_path}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A.2 learned two-stage router — full report (OOF-based performance) over all "
                   "265 non-null DEVELOPMENT questions",
        "n_questions": len(records),
        "cv_seed": 42,
        "n_splits": 5,
        "thresholds": {
            "stage1_threshold": selection.stage1_threshold,
            "stage2_threshold": selection.stage2_threshold,
            "max_under_routing_constraint": 0.10,
        },
        "phase_8a2_learned_router": learned_router_metrics,
        "complex_recall": learned_router_metrics["per_class_metrics"]["COMPLEX"]["recall"],
        "route_distribution_pct": {k: v / len(records) for k, v in predicted_route_counts.items()},
        "stage1_strongest_coefficients": [{"feature": f, "coef": c} for f, c in stage1_top_coef],
        "stage2_strongest_coefficients": [{"feature": f, "coef": c} for f, c in stage2_top_coef],
        "cost_projection": {
            "predicted_route_distribution": predicted_route_counts,
            "routed": {
                "backend_counts": routed_projection.backend_counts,
                "total_cost_usd": routed_projection.total_cost_usd,
                "mean_cost_usd": routed_projection.mean_cost_usd,
                "total_latency_ms": routed_projection.total_latency_ms,
                "mean_latency_ms": routed_projection.mean_latency_ms,
            },
            "agentic_multi_hop": {
                "total_cost_usd": agentic_multi_hop.total_cost_usd,
                "mean_cost_usd": agentic_multi_hop.mean_cost_usd,
                "total_latency_ms": agentic_multi_hop.total_latency_ms,
                "mean_latency_ms": agentic_multi_hop.mean_latency_ms,
            },
        },
        "router_runtime_cost_and_latency": {
            "marginal_cost_usd_per_query": 0.0,
            "note": "pure arithmetic (standardize + dot-product + sigmoid), no LLM call, no network call",
            "stage1_decision_latency_ms_mean": stage1_decision_latency_ms,
            "stage2_decision_latency_ms_mean": stage2_decision_latency_ms,
        },
    }
    report_out_path = PROJECT_ROOT / REPORT_OUTPUT_PATH
    report_out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {report_out_path}")


if __name__ == "__main__":
    main()
