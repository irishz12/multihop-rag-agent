#!/usr/bin/env python
"""FINAL HOLDOUT evaluation — LIVE Agentic Multi-Hop RAG + Adaptive RAG
benchmark run over the frozen 50-question holdout sample, checkpointed per
query.

Deliberately supports ONLY `agentic_multi_hop` and `adaptive_rag` — Dense/Hybrid/
Hybrid+Reranker are NOT re-run against final_holdout (per the Phase 9
holdout spec: "Do not run Dense/Hybrid/Reranker again"; their behavior was
already fully characterized on the development sample).

Same frozen pipelines, config, and models as the development-sample run
(`scripts/run_phase9_benchmark.py` — UNMODIFIED there): `zai.glm-4.7-flash`
controller, `qwen.qwen3-next-80b-a3b-instruct` generation, the frozen
Phase 8A.2 router (results/learned_router_model.json, tau1=0.63,
tau2=0.70). NOTHING is tuned, retrained, or re-thresholded here — this
script only supplies a different (holdout) population of questions to the
exact same pipeline code already used throughout Phase 9.

Reads ONLY data/processed/final_holdout.json (hardcoded
HOLDOUT_SPLIT_FILE, no CLI override) restricted to the qa_ids already
selected by scripts/select_phase9_holdout_sample.py
(results/phase9_holdout_sample.json, read-only here) — this script cannot
run on any question outside that frozen 50-question set.

CHECKPOINTING: writes results/phase9_holdout_{pipeline}_raw.json after
EVERY single question — an interrupted run resumes exactly where it left
off, never repeating a paid Mantle call.

Chunked execution (established CPU-quota-throttling workaround — never
auto-backgrounded).

Usage:
    python scripts/run_phase9_holdout_benchmark.py --pipeline agentic_multi_hop
    python scripts/run_phase9_holdout_benchmark.py --pipeline adaptive_rag
"""

from __future__ import annotations

import argparse
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

HOLDOUT_SPLIT_FILE = "final_holdout.json"
HOLDOUT_SAMPLE_PATH = "results/phase9_holdout_sample.json"
LEARNED_ROUTER_MODEL_PATH = "results/learned_router_model.json"  # READ-ONLY (Phase 8A.2 frozen artifact)
PIPELINES = ("agentic_multi_hop", "adaptive_rag")  # Dense/Hybrid/Hybrid+Reranker deliberately excluded


def _load_router_model(d: dict) -> LinearModel:
    return LinearModel(
        feature_names=tuple(d["feature_names"]), scaler_mean=tuple(d["scaler_mean"]),
        scaler_scale=tuple(d["scaler_scale"]), coef=tuple(d["coef"]), intercept=d["intercept"],
        threshold=d["threshold"],
    )


def _adaptive_summary(trace) -> dict:
    return {
        "predicted_route": trace.route,
        "stage1_probability": trace.stage1_probability, "stage2_probability": trace.stage2_probability,
        "num_retrieval_calls": trace.num_retrieval_calls, "num_reranker_calls": trace.num_reranker_calls,
        "num_agent_hops": trace.num_agent_hops, "num_controller_calls": trace.num_controller_calls,
        "num_generation_calls": trace.num_generation_calls,
        "glm_input_tokens": trace.glm_input_tokens, "glm_output_tokens": trace.glm_output_tokens,
        "glm_cost_usd": trace.glm_cost_usd,
        "qwen_input_tokens": trace.qwen_input_tokens, "qwen_output_tokens": trace.qwen_output_tokens,
        "qwen_cost_usd": trace.qwen_cost_usd, "total_cost_usd": trace.total_cost_usd,
        "retrieval_latency_ms": trace.retrieval_latency_ms, "reranking_latency_ms": trace.reranking_latency_ms,
        "controller_latency_ms": trace.controller_latency_ms, "generation_latency_ms": trace.generation_latency_ms,
        "total_latency_ms": trace.total_latency_ms,
        "stop_reason": trace.stop_reason,
        "answer": trace.answer,
        "num_chunks_used_for_generation": len(trace.chunks_used_for_generation),
        "evidence_doc_ids_used": sorted({c.doc_id for c in trace.chunks_used_for_generation}),
    }


def _agentic_summary(trace) -> dict:
    return {
        "predicted_route": "AGENTIC_MULTI_HOP",
        "stage1_probability": None, "stage2_probability": None,
        "num_retrieval_calls": trace.num_retrieval_calls, "num_reranker_calls": trace.num_retrieval_calls,
        "num_agent_hops": trace.num_retrieval_calls, "num_controller_calls": trace.num_controller_calls,
        "num_generation_calls": trace.num_generation_calls,
        "glm_input_tokens": trace.cost.glm_input_tokens, "glm_output_tokens": trace.cost.glm_output_tokens,
        "glm_cost_usd": trace.cost.glm_cost_usd,
        "qwen_input_tokens": trace.cost.qwen_input_tokens, "qwen_output_tokens": trace.cost.qwen_output_tokens,
        "qwen_cost_usd": trace.cost.qwen_cost_usd, "total_cost_usd": trace.cost.total_cost_usd,
        "retrieval_latency_ms": trace.retrieval_latency_ms, "reranking_latency_ms": trace.reranking_latency_ms,
        "controller_latency_ms": trace.controller_latency_ms, "generation_latency_ms": trace.generation_latency_ms,
        "total_latency_ms": trace.total_latency_ms,
        "stop_reason": trace.stop_reason,
        "answer": trace.final_generation.answer,
        "num_chunks_used_for_generation": trace.chunks_passed_to_final_generation,
        "evidence_doc_ids_used": sorted({c.doc_id for c in trace.final_generation.context.chunks_included}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", required=True, choices=PIPELINES)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_path = f"results/phase9_holdout_{args.pipeline}_raw.json"

    sample_path = PROJECT_ROOT / HOLDOUT_SAMPLE_PATH
    if not sample_path.exists():
        raise SystemExit(f"{sample_path} does not exist — run scripts/select_phase9_holdout_sample.py first")
    sample = json.loads(sample_path.read_text())
    sample_qa_ids = sample["qa_ids"]

    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")

    holdout_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / HOLDOUT_SPLIT_FILE
    all_records = load_qa_records(holdout_path)
    by_qa_id = {qa_id(r): r for r in all_records}
    missing = [q for q in sample_qa_ids if q not in by_qa_id]
    if missing:
        raise SystemExit(f"{len(missing)} sample qa_id(s) not found in final_holdout.json: {missing}")
    population = [by_qa_id[q] for q in sample_qa_ids]

    end = None if args.limit is None else args.offset + args.limit
    slice_records = population[args.offset : end]
    print(f"Population size {len(population)} (frozen holdout sample); processing "
          f"[{args.offset}:{end if end is not None else len(population)}] = {len(slice_records)} for "
          f"pipeline={args.pipeline}")

    out_path = PROJECT_ROOT / output_path
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = {r["qa_id"]: r for r in json.loads(out_path.read_text()).get("records", [])}
        print(f"Found existing {out_path} with {len(existing)} completed record(s) — will skip those")

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

    try:
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
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run live final holdout benchmark: {exc}") from exc

    loop_cfg = agent_config_yaml["loop"]
    agent_pricing = agent_config_yaml["pricing"]
    qwen_pricing = mantle_config["pricing"]
    agentic_config = AgenticConfig(
        max_hops=loop_cfg["max_hops"], hop_top_k=loop_cfg["hop_top_k"],
        max_evidence_chunks=loop_cfg["max_evidence_chunks"], max_context_tokens=loop_cfg["max_context_tokens"],
        timeout_seconds=loop_cfg["timeout_seconds"],
        controller_prompt_version=agent_config_yaml["controller"]["prompt_version"],
        generation_prompt_version=mantle_config["generation"]["prompt_version"],
        glm_input_price_per_million=agent_pricing["input_per_million_tokens"],
        glm_output_price_per_million=agent_pricing["output_per_million_tokens"],
        qwen_input_price_per_million=qwen_pricing["input_per_million_tokens"],
        qwen_output_price_per_million=qwen_pricing["output_per_million_tokens"],
    )

    stage1_model = stage2_model = None
    if args.pipeline == "adaptive_rag":
        router_model = json.loads((PROJECT_ROOT / LEARNED_ROUTER_MODEL_PATH).read_text())
        stage1_model = _load_router_model(router_model["stage1"])
        stage2_model = _load_router_model(router_model["stage2"])
        print(f"Loaded frozen learned router (tau1={stage1_model.threshold}, tau2={stage2_model.threshold})")

    n_processed_this_run = 0
    for i, record in enumerate(slice_records):
        rec_id = qa_id(record)
        if rec_id in existing:
            continue

        gold_ids = gold_doc_ids(record)
        rec_hop_count = hop_count(record)

        if args.pipeline == "agentic_multi_hop":
            trace = run_agentic_retrieval(
                record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                controller_client, generation_client, config=agentic_config,
            )
            summary = _agentic_summary(trace)
        else:  # adaptive_rag
            trace = run_adaptive_pipeline(
                record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                stage1_model, stage2_model, controller_client, generation_client, agentic_config=agentic_config,
            )
            summary = _adaptive_summary(trace)

        existing[rec_id] = {
            "qa_id": rec_id,
            "query": record.query,
            "question_type": record.question_type,
            "hop_count": rec_hop_count,
            "gold_answer": record.answer,
            "gold_docs_total": len(gold_ids),
            **summary,
        }
        n_processed_this_run += 1

        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": f"FINAL HOLDOUT evaluation — LIVE {args.pipeline} pipeline raw traces "
                       "over the frozen 50-question holdout sample",
            "pipeline": args.pipeline,
            "split": "final_holdout",
            "n_questions_total": len(population),
            "n_questions_completed": len(existing),
            "partial_run": len(existing) < len(population),
            "records": list(existing.values()),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

        if (i + 1) % 10 == 0 or (i + 1) == len(slice_records):
            print(f"  [{i + 1}/{len(slice_records)}] qa_id={rec_id} type={record.question_type} "
                  f"route={summary.get('predicted_route')} cost=${summary['total_cost_usd']}")

    print(f"\n{'=' * 70}")
    print(f"This invocation processed {n_processed_this_run} new question(s)")
    print(f"Total accumulated: {len(existing)}/{len(population)} for pipeline={args.pipeline}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
