#!/usr/bin/env python
"""Phase 8A: evaluate the two-stage router on router_validation.

LIVE only for the ambiguous subset — Stage A (heuristic) uses features
already persisted in results/router_dataset.json (no new Qdrant call);
Stage B (GLM 4.7 Flash, via the existing Mantle client) is called ONLY for
questions Stage A was not confident about. Tracks GLM cost separately from
the (zero-cost) heuristic-only calls.

Oracle route labels (results/router_dataset.json's `oracle_route`) are
used HERE, strictly to SCORE the router's predictions after
`mhrag.routing.router.route_question` has already returned a route for
each validation question — never fed into the routing decision itself
(`route_question` is called with `features=` only, the same
`RouterFeatures` object the runtime would have computed live; the oracle
label for that question is looked up separately, afterward, only to
compare against the prediction).

Also produces the cost-aware workload projection (item 13 in the Phase 8A
report): given the PREDICTED route distribution on router_validation,
projects Hybrid-only / Hybrid+Reranker / Agentic workload counts and
cost/latency, compared against Agentic Multi-Hop RAG (100% Agentic) — using
unit costs already measured in earlier phases (does not run those
pipelines again).

Usage:
    python scripts/run_router_validation.py

Writes results/router_validation_report.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.routing.cost_projection import UnitCost, agentic_multi_hop_projection, project_workload
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.heuristic import HeuristicThresholds
from mhrag.routing.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
    under_over_routing_breakdown,
    under_over_routing_rate,
)
from mhrag.routing.router import route_question

ROUTER_DATASET_PATH = "results/router_dataset.json"
ROUTER_SPLIT_PATH = "results/router_split.json"
ROUTER_THRESHOLDS_PATH = "results/router_thresholds.json"

# Unit costs already MEASURED in earlier phases — this script does not
# re-run those pipelines, only projects a workload from them.
HYBRID_LATENCY_SOURCE = "results/retrieval_eval_development.json:latency_ms.hybrid_retrieval.mean"
HYBRID_RERANKER_LATENCY_SOURCE = "results/retrieval_eval_development.json:latency_ms.hybrid_reranker_total.mean"
AGENTIC_UNIT_COST_SOURCE = "results/agentic_budget_calibration_decision.json:metrics_by_budget.4500"


def main() -> None:
    dataset = json.loads((PROJECT_ROOT / ROUTER_DATASET_PATH).read_text())
    split = json.loads((PROJECT_ROOT / ROUTER_SPLIT_PATH).read_text())
    thresholds_artifact = json.loads((PROJECT_ROOT / ROUTER_THRESHOLDS_PATH).read_text())
    thresholds = HeuristicThresholds(**thresholds_artifact["thresholds"])

    validation_ids = set(split["router_validation_qa_ids"])
    records_by_id = {r["qa_id"]: r for r in dataset["records"]}
    validation_records = [records_by_id[qa_id] for qa_id in sorted(validation_ids)]
    print(f"Evaluating router on {len(validation_records)} router_validation questions")
    print(f"Frozen thresholds: {thresholds}")

    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")
    try:
        glm_client = MantleClient(
            model_id=agent_config_yaml["controller"]["model_id"],
            base_url_env=mantle_config["client"]["base_url_env"],
            default_base_url=mantle_config["client"]["default_base_url"],
            api_key_env=mantle_config["client"]["api_key_env"],
            timeout_seconds=mantle_config["client"]["timeout_seconds"],
            temperature=agent_config_yaml["controller"]["temperature"],
            max_output_tokens=agent_config_yaml["controller"]["max_output_tokens"],
            max_retries=mantle_config["client"]["max_retries"],
            retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
        )
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run router validation: {exc}") from exc

    glm_pricing = agent_config_yaml["pricing"]

    true_routes: list[str] = []
    predicted_routes: list[str] = []
    stage_counts = {"heuristic": 0, "glm": 0, "glm_fallback": 0}
    total_glm_cost = 0.0
    total_glm_tokens_in = 0
    total_glm_tokens_out = 0
    latencies_ms: list[float] = []
    per_question = []

    for i, record in enumerate(validation_records):
        features = RouterFeatures(
            query=QueryFeatures(**record["query_features"]),
            retrieval=RetrievalSignals(**record["retrieval_signals"]),
        )
        result = route_question(
            record["query"], None, None, None, None, glm_client, thresholds,
            glm_input_price_per_million=glm_pricing["input_per_million_tokens"],
            glm_output_price_per_million=glm_pricing["output_per_million_tokens"],
            features=features,
        )
        true_routes.append(record["oracle_route"])
        predicted_routes.append(result.route)
        stage_counts[result.stage_used] += 1
        latencies_ms.append(result.total_latency_ms)
        if result.glm_cost is not None and result.glm_cost.total_cost_usd is not None:
            total_glm_cost += result.glm_cost.total_cost_usd
        if result.glm_result is not None:
            usage = result.glm_result.mantle_response.usage
            total_glm_tokens_in += usage.input_tokens or 0
            total_glm_tokens_out += usage.output_tokens or 0

        per_question.append(
            {
                "qa_id": record["qa_id"], "oracle_route": record["oracle_route"],
                "predicted_route": result.route, "stage_used": result.stage_used,
                "correct": result.route == record["oracle_route"],
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == len(validation_records):
            print(f"  [{i + 1}/{len(validation_records)}] stage_used={result.stage_used} "
                  f"predicted={result.route} true={record['oracle_route']}")

    acc = accuracy(true_routes, predicted_routes)
    m_f1 = macro_f1(true_routes, predicted_routes)
    per_class = per_class_metrics(true_routes, predicted_routes)
    matrix = confusion_matrix(true_routes, predicted_routes)
    under_rate, over_rate = under_over_routing_rate(true_routes, predicted_routes)
    breakdown = under_over_routing_breakdown(true_routes, predicted_routes)

    n = len(validation_records)
    heuristic_only_pct = stage_counts["heuristic"] / n
    glm_fallback_pct = (stage_counts["glm"] + stage_counts["glm_fallback"]) / n
    mean_latency_ms = sum(latencies_ms) / n
    mean_glm_cost = total_glm_cost / n

    print(f"\nAccuracy: {acc:.1%}  Macro-F1: {m_f1:.3f}")
    print(f"Heuristic-only: {heuristic_only_pct:.1%}  GLM fallback: {glm_fallback_pct:.1%}")
    print(f"Under-routing: {under_rate:.1%}  Over-routing: {over_rate:.1%}")

    # --- cost-aware workload projection (predicted routes, not oracle) ---
    predicted_route_counts = {
        r: sum(1 for p in predicted_routes if p == r) for r in ("SIMPLE", "MEDIUM", "COMPLEX")
    }
    retrieval_eval = json.loads((PROJECT_ROOT / "results" / "retrieval_eval_development.json").read_text())
    hybrid_latency = retrieval_eval["latency_ms"]["hybrid_retrieval"]["mean"]
    hybrid_reranker_latency = retrieval_eval["latency_ms"]["hybrid_reranker_total"]["mean"]
    agentic_calib = json.loads(
        (PROJECT_ROOT / "results" / "agentic_budget_calibration_decision.json").read_text()
    )
    agentic_metrics = agentic_calib["metrics_by_budget"]["4500"]
    unit_costs = {
        "hybrid_only": UnitCost(cost_usd=0.0, latency_ms=hybrid_latency),
        "hybrid_reranker": UnitCost(cost_usd=0.0, latency_ms=hybrid_reranker_latency),
        "agentic": UnitCost(cost_usd=agentic_metrics["mean_cost_usd"], latency_ms=agentic_metrics["mean_latency_ms"]),
    }
    routed_projection = project_workload(predicted_route_counts, unit_costs)
    agentic_multi_hop = agentic_multi_hop_projection(n, unit_costs["agentic"])

    print(f"\nPredicted route distribution: {predicted_route_counts}")
    print(f"Routed projection: cost=${routed_projection.total_cost_usd:.4f} "
          f"latency_total={routed_projection.total_latency_ms:.0f}ms")
    print(f"Agentic Multi-Hop RAG: cost=${agentic_multi_hop.total_cost_usd:.4f} "
          f"latency_total={agentic_multi_hop.total_latency_ms:.0f}ms")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A router_validation performance report",
        "n_validation_questions": n,
        "thresholds": thresholds_artifact["thresholds"],
        "accuracy": acc,
        "macro_f1": m_f1,
        "per_class_metrics": {k: {"precision": v.precision, "recall": v.recall, "f1": v.f1, "support": v.support}
                               for k, v in per_class.items()},
        "confusion_matrix": matrix,
        "under_routing_rate": under_rate,
        "over_routing_rate": over_rate,
        "under_over_routing_breakdown": breakdown,
        "stage_counts": stage_counts,
        "heuristic_only_pct": heuristic_only_pct,
        "glm_fallback_pct": glm_fallback_pct,
        "glm_cost": {
            "total_usd": total_glm_cost, "mean_usd_per_query": mean_glm_cost,
            "total_input_tokens": total_glm_tokens_in, "total_output_tokens": total_glm_tokens_out,
            "mean_input_tokens_per_query": total_glm_tokens_in / n,
            "mean_output_tokens_per_query": total_glm_tokens_out / n,
        },
        "latency": {"mean_ms_per_query": mean_latency_ms},
        "cost_projection": {
            "predicted_route_distribution": predicted_route_counts,
            "unit_costs_source": {
                "hybrid_only": HYBRID_LATENCY_SOURCE,
                "hybrid_reranker": HYBRID_RERANKER_LATENCY_SOURCE,
                "agentic": AGENTIC_UNIT_COST_SOURCE,
            },
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
        "per_question": per_question,
    }
    out_path = PROJECT_ROOT / "results" / "router_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
