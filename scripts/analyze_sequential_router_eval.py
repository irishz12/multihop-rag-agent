#!/usr/bin/env python
"""Phase 8A.1: offline analysis — merges the LIVE sequential-router eval
(results/sequential_router_eval_raw.json) with Phase 8A's classifier
predictions extended to the same 265 questions
(results/router_full_dev_eval.json, itself Phase-8A-outputs-read-only),
oracle labels, and gold evidence (evaluator-only, from dev_subset.json),
to produce the full Phase 8A vs. Phase 8A.1 comparison report.

Offline — no live calls, no retrieval, no Mantle. Reads Phase 8A's output
files but never writes to them (see
tests/test_analyze_sequential_router_eval_guard.py).

Usage:
    python scripts/analyze_sequential_router_eval.py

Writes results/sequential_router_report.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.cost_projection import UnitCost, agentic_multi_hop_projection, project_workload
from mhrag.routing.gate_analysis import analyze_gate_verdict
from mhrag.routing.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    under_over_routing_breakdown,
    under_over_routing_rate,
)

SEQUENTIAL_EVAL_PATH = "results/sequential_router_eval_raw.json"
PHASE_8A_FULL_EVAL_PATH = "results/router_full_dev_eval.json"  # READ-ONLY (derived from Phase 8A outputs)
DEV_SPLIT_FILE = "dev_subset.json"
OUTPUT_PATH = "results/sequential_router_report.json"


def _to_results(chunk_dicts: list[dict] | None) -> list[RetrievalResult]:
    if not chunk_dicts:
        return []
    return [
        RetrievalResult(
            rank=c["rank"], score=c["score"], method="eval", chunk_id=c["chunk_id"], doc_id=c["doc_id"],
            title="", url="", source="", category="", published_at="2024-01-01T00:00:00+00:00",
            text="", position=0,
        )
        for c in chunk_dicts
    ]


def _metrics_block(true_routes, predicted_routes) -> dict:
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
    seq_eval = json.loads((PROJECT_ROOT / SEQUENTIAL_EVAL_PATH).read_text())
    phase8a_full = json.loads((PROJECT_ROOT / PHASE_8A_FULL_EVAL_PATH).read_text())

    seq_by_id = {r["qa_id"]: r for r in seq_eval["records"]}
    phase8a_by_id = {p["qa_id"]: p for p in phase8a_full["predictions"]}

    common_ids = sorted(set(seq_by_id) & set(phase8a_by_id))
    print(f"Comparing on {len(common_ids)} common qa_ids "
          f"(sequential={len(seq_by_id)}, phase8a={len(phase8a_by_id)})")
    if len(common_ids) != 265:
        print(f"  WARNING: expected 265, got {len(common_ids)}")

    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    dev_records = load_qa_records(dev_path)
    dev_by_id = {qa_id(r): r for r in dev_records}

    oracle_routes = []
    phase8a_predicted = []
    phase8a1_predicted = []

    gate1_outcomes: dict[str, int] = {}
    gate2_outcomes: dict[str, int] = {}
    gate1_sufficient_count = 0
    gate2_called_count = 0
    gate2_sufficient_count = 0
    false_sufficiency_examples = []

    total_glm_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    total_latency_ms = 0.0

    for qid in common_ids:
        seq_rec = seq_by_id[qid]
        p8a_rec = phase8a_by_id[qid]
        oracle_routes.append(p8a_rec["oracle_route"])
        phase8a_predicted.append(p8a_rec["predicted_route"])
        phase8a1_predicted.append(seq_rec["route"])

        record = dev_by_id[qid]

        gate1 = seq_rec["gate1"]
        gate1_analysis = analyze_gate_verdict(record, _to_results(seq_rec["hybrid_top5"]), gate1["sufficient"])
        gate1_outcomes[gate1_analysis.outcome] = gate1_outcomes.get(gate1_analysis.outcome, 0) + 1
        if gate1["sufficient"]:
            gate1_sufficient_count += 1
        if gate1_analysis.outcome == "false_sufficiency":
            false_sufficiency_examples.append(
                {"qa_id": qid, "gate": 1, "num_gold_docs": gate1_analysis.num_gold_docs,
                 "num_gold_docs_missing": gate1_analysis.num_gold_docs_missing, "reason": gate1["reason"]}
            )

        gate2 = seq_rec["gate2"]
        if gate2 is not None:
            gate2_called_count += 1
            gate2_analysis = analyze_gate_verdict(record, _to_results(seq_rec["reranked_top5"]), gate2["sufficient"])
            gate2_outcomes[gate2_analysis.outcome] = gate2_outcomes.get(gate2_analysis.outcome, 0) + 1
            if gate2["sufficient"]:
                gate2_sufficient_count += 1
            if gate2_analysis.outcome == "false_sufficiency":
                false_sufficiency_examples.append(
                    {"qa_id": qid, "gate": 2, "num_gold_docs": gate2_analysis.num_gold_docs,
                     "num_gold_docs_missing": gate2_analysis.num_gold_docs_missing, "reason": gate2["reason"]}
                )

        total_glm_calls += seq_rec["num_glm_calls"]
        total_input_tokens += seq_rec["glm_input_tokens"]
        total_output_tokens += seq_rec["glm_output_tokens"]
        if seq_rec["glm_cost_usd"] is not None:
            total_cost += seq_rec["glm_cost_usd"]
        total_latency_ms += seq_rec["total_latency_ms"]

    n = len(common_ids)
    phase8a_metrics = _metrics_block(oracle_routes, phase8a_predicted)
    phase8a1_metrics = _metrics_block(oracle_routes, phase8a1_predicted)

    print(f"\nPhase 8A  (full 265, reused/extended): accuracy={phase8a_metrics['accuracy']:.1%} "
          f"macro_f1={phase8a_metrics['macro_f1']:.3f}")
    print(f"Phase 8A.1 (sequential, full 265):      accuracy={phase8a1_metrics['accuracy']:.1%} "
          f"macro_f1={phase8a1_metrics['macro_f1']:.3f}")
    print(f"Phase 8A.1 under-routing={phase8a1_metrics['under_routing_rate']:.1%} "
          f"over-routing={phase8a1_metrics['over_routing_rate']:.1%}")

    gate1_sufficient_pct = gate1_sufficient_count / n
    gate2_sufficient_pct = gate2_sufficient_count / gate2_called_count if gate2_called_count else 0.0
    print(f"\nGate 1 sufficient: {gate1_sufficient_pct:.1%}  Gate 2 sufficient (of {gate2_called_count} called): "
          f"{gate2_sufficient_pct:.1%}")
    print(f"Gate 1 outcomes: {gate1_outcomes}")
    print(f"Gate 2 outcomes: {gate2_outcomes}")
    print(f"False-sufficiency events (dangerous): {len(false_sufficiency_examples)}")

    # --- cost projection (Phase 8A.1's predicted distribution vs Agentic Multi-Hop RAG) ---
    predicted_route_counts = {r: phase8a1_predicted.count(r) for r in ("SIMPLE", "MEDIUM", "COMPLEX")}
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
    routed_projection = project_workload(predicted_route_counts, unit_costs)
    agentic_multi_hop = agentic_multi_hop_projection(n, unit_costs["agentic"])

    print(f"\nPredicted distribution: {predicted_route_counts}")
    print(f"Routed cost=${routed_projection.total_cost_usd:.4f} vs Agentic Multi-Hop RAG=${agentic_multi_hop.total_cost_usd:.4f}")
    print(f"NOTE: correctness (accuracy={phase8a1_metrics['accuracy']:.1%}) and cost savings are reported "
          f"separately — savings partly reflect over-routing bias (COMPLEX over-predicted), NOT purely "
          f"efficient correct routing.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A.1 evidence-aware sequential router — full report + Phase 8A comparison "
                   "over all 265 non-null DEVELOPMENT questions",
        "n_questions": n,
        "phase_8a_direct_classifier": {**phase8a_metrics, "note": "frozen Phase 8A thresholds/prompt, "
                                        "extended from 79-question router_validation to all 265 dev questions"},
        "phase_8a1_sequential_router": phase8a1_metrics,
        "gate1_sufficient_pct": gate1_sufficient_pct,
        "gate2_called_count": gate2_called_count,
        "gate2_sufficient_pct": gate2_sufficient_pct,
        "gate1_outcomes": gate1_outcomes,
        "gate2_outcomes": gate2_outcomes,
        "false_sufficiency_events": false_sufficiency_examples,
        "router_cost_and_latency": {
            "mean_glm_calls_per_query": total_glm_calls / n,
            "mean_input_tokens_per_query": total_input_tokens / n,
            "mean_output_tokens_per_query": total_output_tokens / n,
            "total_glm_cost_usd": total_cost,
            "mean_glm_cost_usd_per_query": total_cost / n,
            "mean_latency_ms_per_query": total_latency_ms / n,
        },
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
            "caveat": "correctness and cost savings are reported separately; savings are NOT claimed as "
                      "successful efficiency if caused by under-routing (or, here, largely by over-routing "
                      "toward COMPLEX — see phase_8a1_sequential_router.over_routing_rate).",
        },
    }
    out_path = PROJECT_ROOT / OUTPUT_PATH
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
