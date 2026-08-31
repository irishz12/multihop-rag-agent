#!/usr/bin/env python
"""Phase 8B: LIVE Adaptive RAG vs Agentic Multi-Hop RAG DEVELOPMENT smoke comparison.

Makes REAL API calls (GLM 4.7 Flash controller + Qwen final answer) to
Amazon Bedrock Mantle and incurs REAL (small) cost — connectivity/behavior
validation over a small, deterministic question set, NOT the full
development generation benchmark.

For EVERY question, runs BOTH pipelines independently and compares them:
  - Adaptive RAG (`mhrag.adaptive.pipeline.run_adaptive_pipeline`): Hybrid ->
    frozen Phase 8A.2 learned router (Stage 1 tau=0.63, Stage 2 tau=0.70)
    -> the cheapest backend the router decides is sufficient -> Qwen.
  - Agentic Multi-Hop RAG (`mhrag.agent.loop.run_agentic_retrieval`,
    UNMODIFIED — this script never edits or reconfigures it): full bounded
    agentic loop for every question regardless of route, exactly as
    Phase 7/7.1 left it.

Both use the SAME embedding/BM25/reranker models, the SAME GLM controller
client, and the SAME Qwen generation client/config — so any cost/latency
difference reflects the router's decisions, not a different model or
prompt.

Reads ONLY data/processed/dev_subset.json — DEV_SPLIT_FILE is a hardcoded
module constant, no CLI flag, no config option, so there is no code path in
this script that can reach final_holdout.json. `results/learned_router_
model.json` (Phase 8A.2's frozen artifact) is read-only; this script never
writes to it or to any other prior-phase output file.

DEFAULT_QA_IDS is a fixed, deterministic 13-question DEVELOPMENT selection
(computed offline from `results/learned_router_dataset.json` + the frozen
Phase 8A.2 model — see the selection rationale below), covering all three
PREDICTED routes plus known hard multi-hop and under-routed cases:
  - 3 predicted SIMPLE (clean comparison_query cases)
  - 3 predicted MEDIUM (all 3 the frozen model predicts MEDIUM anywhere in
    the full 265-question DEVELOPMENT set — Phase 8A.2 already reported
    MEDIUM as its smallest, hardest-to-hit class)
  - 4 predicted COMPLEX, correctly escalated per the oracle, hop_count 3-4
    (hard multi-hop: inference_query needing 4 distinct gold documents)
  - 3 UNDER-ROUTED cases: oracle_route == COMPLEX but the frozen router
    predicts SIMPLE (2 of them hop_count == 4) — deliberately included so
    this smoke run can show what an under-routing failure looks like in
    practice, not just in an offline confusion matrix.

Only `query` text and retrieved chunk text ever reach the controller or the
final-answer model — never gold `answer`, `evidence_list`, or
`question_type` (see mhrag.adaptive.pipeline / mhrag.agent.controller /
mhrag.generation.answer module docstrings for the structural guarantee).
Gold evidence (`evidence_list`) IS used in THIS script, strictly AFTER both
pipelines have already returned their traces, only to score how much gold
evidence ended up in each pipeline's final generation context — the same
"never feeds into a decision, only scores it afterward" pattern as
mhrag.routing.gate_analysis.

Requires `$OPENAI_API_KEY` in the environment. The key is never printed,
logged, or written to any output file by this script.

Usage:
    python scripts/run_adaptive_smoke_comparison.py

Writes results/adaptive_smoke_comparison.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.adaptive.pipeline import run_adaptive_pipeline
from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids, hop_count
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker
from mhrag.routing.learned_router import LinearModel

# Hardcoded to the development split ONLY — no CLI flag, no config option,
# so there is no code path in this script that can reach final_holdout.json.
DEV_SPLIT_FILE = "dev_subset.json"
LEARNED_ROUTER_MODEL_PATH = "results/learned_router_model.json"  # READ-ONLY (Phase 8A.2 frozen artifact)
OUTPUT_PATH = "results/adaptive_smoke_comparison.json"

# See module docstring for the selection rationale. Order: SIMPLE, MEDIUM,
# COMPLEX (hard multi-hop), then known under-routed failures.
DEFAULT_QA_IDS = [
    # 3 predicted SIMPLE
    "90dbb8314f0bcef9", "c452d3bc70dd908f", "929b1254ca1f92a6",
    # 3 predicted MEDIUM (every MEDIUM prediction in the full 265-question set)
    "9083983ca6f3a1a4", "6c05fd835e0da59f", "01889ae036d15cd3",
    # 4 predicted COMPLEX, correctly escalated, hard multi-hop (hop_count 3-4)
    "f153ffa33efe7046", "238d0a7eede51e7b", "83da4f3523125728", "7b06c64023ae5be7",
    # 3 under-routed: oracle COMPLEX, router predicts SIMPLE (2 of them 4-hop)
    "85094ea3df3f689f", "fba6e6f0cb2d6750", "e47266bdce7cc08a",
]


def _load_router_model(d: dict) -> LinearModel:
    return LinearModel(
        feature_names=tuple(d["feature_names"]), scaler_mean=tuple(d["scaler_mean"]),
        scaler_scale=tuple(d["scaler_scale"]), coef=tuple(d["coef"]), intercept=d["intercept"],
        threshold=d["threshold"],
    )


def _trace_summary(trace, route_label: str | None, stage1_p: float | None, stage2_p: float | None) -> dict:
    """Uniform per-pipeline summary — works for both AdaptiveTrace (route_label
    given) and Agentic Multi-Hop RAG's AgenticTrace (route_label=None, always
    effectively COMPLEX-shaped: full agentic loop, no router)."""
    if route_label is not None:  # AdaptiveTrace
        return {
            "predicted_route": trace.route,
            "stage1_probability": trace.stage1_probability,
            "stage2_probability": trace.stage2_probability,
            "num_retrieval_calls": trace.num_retrieval_calls,
            "num_reranker_calls": trace.num_reranker_calls,
            "num_agent_hops": trace.num_agent_hops,
            "num_controller_calls": trace.num_controller_calls,
            "num_generation_calls": trace.num_generation_calls,
            "glm_input_tokens": trace.glm_input_tokens,
            "glm_output_tokens": trace.glm_output_tokens,
            "glm_cost_usd": trace.glm_cost_usd,
            "qwen_input_tokens": trace.qwen_input_tokens,
            "qwen_output_tokens": trace.qwen_output_tokens,
            "qwen_cost_usd": trace.qwen_cost_usd,
            "total_cost_usd": trace.total_cost_usd,
            "retrieval_latency_ms": trace.retrieval_latency_ms,
            "reranking_latency_ms": trace.reranking_latency_ms,
            "controller_latency_ms": trace.controller_latency_ms,
            "generation_latency_ms": trace.generation_latency_ms,
            "total_latency_ms": trace.total_latency_ms,
            "stop_reason": trace.stop_reason,
            "answer": trace.answer,
            "unique_docs_used": trace.unique_docs_used,
            "num_chunks_used_for_generation": len(trace.chunks_used_for_generation),
            "evidence_doc_ids_used": sorted({c.doc_id for c in trace.chunks_used_for_generation}),
        }
    # AgenticTrace (Agentic Multi-Hop RAG)
    return {
        "predicted_route": "AGENTIC_MULTI_HOP",
        "stage1_probability": None,
        "stage2_probability": None,
        "num_retrieval_calls": trace.num_retrieval_calls,
        "num_reranker_calls": trace.num_retrieval_calls,
        "num_agent_hops": trace.num_retrieval_calls,
        "num_controller_calls": trace.num_controller_calls,
        "num_generation_calls": trace.num_generation_calls,
        "glm_input_tokens": trace.cost.glm_input_tokens,
        "glm_output_tokens": trace.cost.glm_output_tokens,
        "glm_cost_usd": trace.cost.glm_cost_usd,
        "qwen_input_tokens": trace.cost.qwen_input_tokens,
        "qwen_output_tokens": trace.cost.qwen_output_tokens,
        "qwen_cost_usd": trace.cost.qwen_cost_usd,
        "total_cost_usd": trace.cost.total_cost_usd,
        "retrieval_latency_ms": trace.retrieval_latency_ms,
        "reranking_latency_ms": trace.reranking_latency_ms,
        "controller_latency_ms": trace.controller_latency_ms,
        "generation_latency_ms": trace.generation_latency_ms,
        "total_latency_ms": trace.total_latency_ms,
        "stop_reason": trace.stop_reason,
        "answer": trace.final_generation.answer,
        "unique_docs_used": trace.unique_documents_retrieved,
        "num_chunks_used_for_generation": trace.chunks_passed_to_final_generation,
        "evidence_doc_ids_used": sorted({c.doc_id for c in trace.final_generation.context.chunks_included}),
    }


def main() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    dev_records = load_qa_records(dev_path)
    dev_by_qa_id = {qa_id(r): r for r in dev_records}
    missing = [q for q in DEFAULT_QA_IDS if q not in dev_by_qa_id]
    if missing:
        raise SystemExit(f"{len(missing)} DEFAULT_QA_IDS not found in dev_subset.json: {missing}")
    records = [dev_by_qa_id[q] for q in DEFAULT_QA_IDS]
    print(f"Loaded {len(records)} DEVELOPMENT smoke questions from {dev_path}")

    router_model_path = PROJECT_ROOT / LEARNED_ROUTER_MODEL_PATH
    router_model = json.loads(router_model_path.read_text())
    stage1_model = _load_router_model(router_model["stage1"])
    stage2_model = _load_router_model(router_model["stage2"])
    print(f"Loaded frozen learned router from {router_model_path} "
          f"(tau1={stage1_model.threshold}, tau2={stage2_model.threshold})")

    print(f"\nLoading embedding model {retrieval_config['embedding']['model_name']} ...")
    embedding_model = EmbeddingModel(
        model_name=retrieval_config["embedding"]["model_name"],
        device=retrieval_config["embedding"].get("device"),
        normalize=retrieval_config["embedding"]["normalize"],
        query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
        batch_size=retrieval_config["embedding"]["batch_size"],
    )
    print(f"Loading BM25 model {retrieval_config['bm25']['model_name']} ...")
    bm25_model = Bm25Model(model_name=retrieval_config["bm25"]["model_name"])
    print(f"Loading reranker model {retrieval_config['reranker']['model_name']} ...")
    reranker = Reranker(
        model_name=retrieval_config["reranker"]["model_name"],
        device=retrieval_config["reranker"].get("device"),
        batch_size=retrieval_config["reranker"]["batch_size"],
    )

    qdrant_client = get_client(retrieval_config["qdrant"]["url"])
    collection_name = retrieval_config["qdrant"]["collection_name"]

    print(f"\nConnecting to Mantle: controller={agent_config_yaml['controller']['model_id']}, "
          f"final={mantle_config['generation']['model_id']}")
    try:
        controller_client = MantleClient(
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
        generation_client = MantleClient(
            model_id=mantle_config["generation"]["model_id"],
            base_url_env=mantle_config["client"]["base_url_env"],
            default_base_url=mantle_config["client"]["default_base_url"],
            api_key_env=mantle_config["client"]["api_key_env"],
            timeout_seconds=mantle_config["client"]["timeout_seconds"],
            temperature=mantle_config["generation"]["temperature"],
            max_output_tokens=mantle_config["generation"]["max_output_tokens"],
            max_retries=mantle_config["client"]["max_retries"],
            retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
        )
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run live Phase 8B smoke comparison: {exc}") from exc

    loop_cfg = agent_config_yaml["loop"]
    agent_pricing = agent_config_yaml["pricing"]
    qwen_pricing = mantle_config["pricing"]
    # ONE shared config for BOTH pipelines — same max_hops/context budget/prompt versions/pricing,
    # so any difference in the comparison reflects routing, not a configuration difference.
    config = AgenticConfig(
        max_hops=loop_cfg["max_hops"],
        hop_top_k=loop_cfg["hop_top_k"],
        max_evidence_chunks=loop_cfg["max_evidence_chunks"],
        max_context_tokens=loop_cfg["max_context_tokens"],
        timeout_seconds=loop_cfg["timeout_seconds"],
        controller_prompt_version=agent_config_yaml["controller"]["prompt_version"],
        generation_prompt_version=mantle_config["generation"]["prompt_version"],
        glm_input_price_per_million=agent_pricing["input_per_million_tokens"],
        glm_output_price_per_million=agent_pricing["output_per_million_tokens"],
        qwen_input_price_per_million=qwen_pricing["input_per_million_tokens"],
        qwen_output_price_per_million=qwen_pricing["output_per_million_tokens"],
    )

    results = []
    adaptive_route_counts: dict[str, int] = {}
    adaptive_total_cost = 0.0
    agentic_total_cost = 0.0
    adaptive_total_latency = 0.0
    agentic_total_latency = 0.0
    under_routed_failures = []

    for i, (qid, record) in enumerate(zip(DEFAULT_QA_IDS, records)):
        print(f"\n{'=' * 70}\n[{i + 1}/{len(records)}] qa_id={qid}  {record.query[:90]!r}")

        gold_ids = gold_doc_ids(record)
        oracle_hop_count = hop_count(record)

        adaptive_trace = run_adaptive_pipeline(
            record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            stage1_model, stage2_model, controller_client, generation_client, agentic_config=config,
        )
        agentic_trace = run_agentic_retrieval(
            record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            controller_client, generation_client, config=config,
        )

        adaptive_summary = _trace_summary(adaptive_trace, "adaptive_rag", None, None)
        agentic_summary = _trace_summary(agentic_trace, None, None, None)

        adaptive_gold_covered = len(gold_ids & set(adaptive_summary["evidence_doc_ids_used"]))
        agentic_gold_covered = len(gold_ids & set(agentic_summary["evidence_doc_ids_used"]))
        adaptive_summary["gold_docs_total"] = len(gold_ids)
        adaptive_summary["gold_docs_covered"] = adaptive_gold_covered
        agentic_summary["gold_docs_total"] = len(gold_ids)
        agentic_summary["gold_docs_covered"] = agentic_gold_covered

        if adaptive_gold_covered > agentic_gold_covered:
            evidence_comparison = "better"
        elif adaptive_gold_covered < agentic_gold_covered:
            evidence_comparison = "worse"
        else:
            evidence_comparison = "same"

        # An "under-routed failure": the router sent this question somewhere cheaper than
        # COMPLEX, but Adaptive RAG's evidence coverage came up short of Agentic Multi-Hop
        # RAG's — the concrete, observable symptom of under-routing (not just "oracle said COMPLEX").
        is_under_routed_failure = (
            adaptive_trace.route in ("SIMPLE", "MEDIUM") and adaptive_gold_covered < agentic_gold_covered
        )
        if is_under_routed_failure:
            under_routed_failures.append(qid)

        print(f"  Adaptive:       route={adaptive_trace.route} stop={adaptive_trace.stop_reason} "
              f"retrieval_calls={adaptive_summary['num_retrieval_calls']} "
              f"controller_calls={adaptive_summary['num_controller_calls']} "
              f"cost=${adaptive_summary['total_cost_usd']:.6f} "
              f"latency={adaptive_summary['total_latency_ms']:.0f}ms "
              f"gold_covered={adaptive_gold_covered}/{len(gold_ids)}")
        print(f"  Agentic Multi-Hop RAG: stop={agentic_summary['stop_reason']} "
              f"retrieval_calls={agentic_summary['num_retrieval_calls']} "
              f"controller_calls={agentic_summary['num_controller_calls']} "
              f"cost=${agentic_summary['total_cost_usd']:.6f} "
              f"latency={agentic_summary['total_latency_ms']:.0f}ms "
              f"gold_covered={agentic_gold_covered}/{len(gold_ids)}")
        print(f"  evidence_comparison={evidence_comparison}  under_routed_failure={is_under_routed_failure}")
        print(f"  Adaptive RAG answer:       {adaptive_summary['answer'][:160]!r}")
        print(f"  Agentic Multi-Hop RAG answer: {agentic_summary['answer'][:160]!r}")

        adaptive_route_counts[adaptive_trace.route] = adaptive_route_counts.get(adaptive_trace.route, 0) + 1
        if adaptive_summary["total_cost_usd"] is not None:
            adaptive_total_cost += adaptive_summary["total_cost_usd"]
        if agentic_summary["total_cost_usd"] is not None:
            agentic_total_cost += agentic_summary["total_cost_usd"]
        adaptive_total_latency += adaptive_summary["total_latency_ms"]
        agentic_total_latency += agentic_summary["total_latency_ms"]

        results.append(
            {
                "qa_id": qid,
                "query": record.query,
                "question_type": record.question_type,
                "hop_count": oracle_hop_count,
                "gold_docs_total": len(gold_ids),
                "adaptive_rag": adaptive_summary,
                "agentic_multi_hop": agentic_summary,
                "evidence_comparison": evidence_comparison,
                "under_routed_failure": is_under_routed_failure,
            }
        )

    n = len(records)
    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8B Adaptive RAG vs Agentic Multi-Hop RAG DEVELOPMENT smoke comparison "
                   "(NOT the full development generation benchmark)",
        "split": "development",
        "n_questions": n,
        "qa_ids": DEFAULT_QA_IDS,
        "controller_model": agent_config_yaml["controller"]["model_id"],
        "generation_model": mantle_config["generation"]["model_id"],
        "router_thresholds": {"stage1": stage1_model.threshold, "stage2": stage2_model.threshold},
        "adaptive_route_distribution": adaptive_route_counts,
        "adaptive_rag_total_cost_usd": adaptive_total_cost,
        "agentic_multi_hop_total_cost_usd": agentic_total_cost,
        "cost_reduction_pct": (
            (agentic_total_cost - adaptive_total_cost) / agentic_total_cost if agentic_total_cost > 0 else None
        ),
        "adaptive_rag_mean_latency_ms": adaptive_total_latency / n,
        "agentic_multi_hop_mean_latency_ms": agentic_total_latency / n,
        "latency_reduction_pct": (
            (agentic_total_latency - adaptive_total_latency) / agentic_total_latency
            if agentic_total_latency > 0 else None
        ),
        "under_routed_failures": under_routed_failures,
        "config": {"loop": loop_cfg, "controller_pricing": agent_pricing, "generation_pricing": qwen_pricing},
        "results": results,
    }
    out_path = PROJECT_ROOT / OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 70}")
    print(f"Adaptive route distribution: {adaptive_route_counts}")
    print(f"Adaptive RAG cost=${adaptive_total_cost:.6f} vs Agentic Multi-Hop RAG cost=${agentic_total_cost:.6f} "
          f"({artifact['cost_reduction_pct']:.1%} reduction)" if artifact["cost_reduction_pct"] is not None else "")
    print(f"Adaptive RAG mean latency={artifact['adaptive_rag_mean_latency_ms']:.0f}ms vs "
          f"Agentic Multi-Hop RAG mean latency={artifact['agentic_multi_hop_mean_latency_ms']:.0f}ms")
    print(f"Under-routed failures: {under_routed_failures}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
