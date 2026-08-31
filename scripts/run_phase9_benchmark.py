#!/usr/bin/env python
"""Phase 9: LIVE full-DEVELOPMENT-benchmark generation run, one pipeline
at a time, checkpointed per query so an interrupted run never repeats a
paid Mantle call.

Five pipelines, selected via `--pipeline`:
  - dense            : dense_search(top_k=5) -> Qwen answer
  - hybrid            : deterministic_hybrid_search (Dense+BM25+RRF k=60,
                        final_top_k=5, frozen configs/retrieval.yaml
                        defaults) -> Qwen answer
  - hybrid_reranker    : rerank_hybrid_search (+ BAAI/bge-reranker-base,
                        frozen candidate depth 20 -> top-5) -> Qwen answer
  - agentic_multi_hop  : mhrag.agent.loop.run_agentic_retrieval, UNMODIFIED
                        (the frozen Agentic Multi-Hop RAG baseline)
  - adaptive_rag       : mhrag.adaptive.pipeline.run_adaptive_pipeline,
                        UNMODIFIED, using the FROZEN Phase 8A.2 router
                        (results/learned_router_model.json, tau1=0.63,
                        tau2=0.70 — never retrained or re-thresholded here)

ALL FIVE use the exact same Qwen final-generation model/prompt/pricing
(configs/mantle.yaml, unchanged) — the only thing that differs between
pipelines is what evidence reaches that one shared call.

Runs over ALL 300 DEVELOPMENT questions, INCLUDING null_query (unlike
every earlier phase's router-only work, which excluded them) — a
null_query's evidence_list is empty, which every function called here
(gold_doc_ids/hop_count) already handles by returning an empty set/0,
never raising.

Reads ONLY data/processed/dev_subset.json — DEV_SPLIT_FILE is a hardcoded
module constant, no CLI flag, no config option, so there is no code path
in this script that can reach final_holdout.json. `results/learned_router_
model.json` (Phase 8A.2's frozen artifact) is read-only for the adaptive
pipeline; this script never writes to it or to any other prior-phase
output file.

CHECKPOINTING: writes results/phase9_{pipeline}_raw.json after EVERY
single question (not just at the end of an invocation) — an interrupted
run resumes exactly where it left off; already-completed qa_ids in the
existing checkpoint are skipped without re-calling Mantle.

Gold answer/evidence_list/question_type are used ONLY to look up which
question to ask and (for `hop_count`) to label the record for later
offline scoring — NEVER passed into any retrieval/routing/generation call
(see mhrag.adaptive.pipeline / mhrag.agent.loop / mhrag.generation.answer
module docstrings for the structural guarantee that none of those
functions even accept such a parameter).

Chunked execution (established CPU-quota-throttling workaround — this
project's long-running live jobs are NEVER auto-backgrounded): supports
`--offset`/`--limit` to bound one invocation's question count.

Requires `$OPENAI_API_KEY` in the environment (dense/hybrid pipelines
still need it for the one Qwen call/question; agentic_multi_hop/adaptive_rag
also need it for the GLM controller). Never printed/logged/persisted.

Usage:
    python scripts/run_phase9_benchmark.py --pipeline dense --offset 0 --limit 100
    python scripts/run_phase9_benchmark.py --pipeline adaptive_rag --offset 200

Writes results/phase9_{pipeline}_raw.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from mhrag.adaptive.pipeline import run_adaptive_pipeline
from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids, hop_count
from mhrag.generation.answer import generate_answer
from mhrag.generation.context import approximate_token_count
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker, rerank_hybrid_search
from mhrag.retrieval.rrf import deterministic_hybrid_search
from mhrag.routing.learned_router import LinearModel

DEV_SPLIT_FILE = "dev_subset.json"
LEARNED_ROUTER_MODEL_PATH = "results/learned_router_model.json"  # READ-ONLY (Phase 8A.2 frozen artifact)
PIPELINES = ("dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag")
GENERATION_TOP_K = 5


def _load_router_model(d: dict) -> LinearModel:
    return LinearModel(
        feature_names=tuple(d["feature_names"]), scaler_mean=tuple(d["scaler_mean"]),
        scaler_scale=tuple(d["scaler_scale"]), coef=tuple(d["coef"]), intercept=d["intercept"],
        threshold=d["threshold"],
    )


def _direct_summary(retrieved, retrieval_latency_ms: float, generation) -> dict:
    """Uniform record shape for dense/hybrid/hybrid_reranker — one
    retrieval call, no reranker/controller calls, exactly one Qwen call."""
    usage = generation.mantle_response.usage
    return {
        "predicted_route": None,
        "stage1_probability": None, "stage2_probability": None,
        "num_retrieval_calls": 1, "num_reranker_calls": 0, "num_agent_hops": 0,
        "num_controller_calls": 0, "num_generation_calls": 1,
        "glm_input_tokens": 0, "glm_output_tokens": 0, "glm_cost_usd": None,
        "qwen_input_tokens": usage.input_tokens or 0, "qwen_output_tokens": usage.output_tokens or 0,
        "qwen_cost_usd": generation.cost.total_cost_usd, "total_cost_usd": generation.cost.total_cost_usd,
        "retrieval_latency_ms": retrieval_latency_ms, "reranking_latency_ms": 0.0,
        "controller_latency_ms": 0.0, "generation_latency_ms": generation.mantle_response.llm_latency_ms,
        "total_latency_ms": retrieval_latency_ms + generation.mantle_response.total_latency_ms,
        "stop_reason": "single_pass",
        "answer": generation.answer,
        "num_chunks_used_for_generation": len(generation.context.chunks_included),
        "evidence_doc_ids_used": sorted({c.doc_id for c in generation.context.chunks_included}),
    }


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
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--sample-file", default=None,
        help="if given, restrict to the qa_ids listed in this results/phase9_sample.json-shaped file "
             "(e.g. results/phase9_sample.json) instead of all 300 DEVELOPMENT records; --offset/--limit "
             "then slice WITHIN that restricted list, in the sample file's own qa_ids order",
    )
    args = parser.parse_args()

    output_path = args.output or f"results/phase9_{args.pipeline}_raw.json"

    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)  # ALL 300, including null_query
    population = all_records
    if args.sample_file:
        sample = json.loads((PROJECT_ROOT / args.sample_file).read_text())
        sample_qa_ids = sample["qa_ids"]
        by_qa_id = {qa_id(r): r for r in all_records}
        missing_from_dev = [q for q in sample_qa_ids if q not in by_qa_id]
        if missing_from_dev:
            raise SystemExit(f"{len(missing_from_dev)} sample qa_id(s) not found in dev_subset.json: "
                              f"{missing_from_dev}")
        population = [by_qa_id[q] for q in sample_qa_ids]  # sample's own deterministic order
        print(f"Restricting to --sample-file {args.sample_file} — {len(population)} qa_ids")

    end = None if args.limit is None else args.offset + args.limit
    slice_records = population[args.offset : end]
    print(f"Population size {len(population)}; processing "
          f"[{args.offset}:{end if end is not None else len(population)}] = {len(slice_records)} for "
          f"pipeline={args.pipeline}")

    out_path = PROJECT_ROOT / output_path
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = {r["qa_id"]: r for r in json.loads(out_path.read_text()).get("records", [])}
        print(f"Found existing {out_path} with {len(existing)} completed record(s) — will skip those")

    needs_reranker = args.pipeline in ("hybrid_reranker", "agentic_multi_hop", "adaptive_rag")
    needs_agentic = args.pipeline in ("agentic_multi_hop", "adaptive_rag")

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
    reranker = None
    if needs_reranker:
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
        controller_client = None
        if needs_agentic:
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
        raise SystemExit(f"Cannot run live Phase 9 benchmark: {exc}") from exc

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

        if args.pipeline == "dense":
            t0 = time.monotonic()
            retrieved = dense_search(record.query, qdrant_client, collection_name, embedding_model, top_k=GENERATION_TOP_K)
            retrieval_latency_ms = (time.monotonic() - t0) * 1000
        elif args.pipeline == "hybrid":
            t0 = time.monotonic()
            retrieved = deterministic_hybrid_search(
                record.query, qdrant_client, collection_name, embedding_model, bm25_model,
                dense_top_k=retrieval_config["hybrid"]["dense_top_k"],
                bm25_top_k=retrieval_config["hybrid"]["bm25_top_k"], final_top_k=GENERATION_TOP_K,
            )
            retrieval_latency_ms = (time.monotonic() - t0) * 1000
        elif args.pipeline == "hybrid_reranker":
            t0 = time.monotonic()
            retrieved = rerank_hybrid_search(
                record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                final_top_k=GENERATION_TOP_K,
            )
            retrieval_latency_ms = (time.monotonic() - t0) * 1000
        elif args.pipeline == "agentic_multi_hop":
            trace = run_agentic_retrieval(
                record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                controller_client, generation_client, config=agentic_config,
            )
            summary = _agentic_summary(trace)
            retrieved = None
        else:  # adaptive_rag
            trace = run_adaptive_pipeline(
                record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                stage1_model, stage2_model, controller_client, generation_client, agentic_config=agentic_config,
            )
            summary = _adaptive_summary(trace)
            retrieved = None

        if retrieved is not None:
            generation = generate_answer(
                record.query, retrieved, generation_client, approximate_token_count,
                top_k=GENERATION_TOP_K, max_context_tokens=agentic_config.max_context_tokens,
                input_price_per_million=qwen_pricing["input_per_million_tokens"],
                output_price_per_million=qwen_pricing["output_per_million_tokens"],
                prompt_version=mantle_config["generation"]["prompt_version"],
            )
            summary = _direct_summary(retrieved, retrieval_latency_ms, generation)

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

        # Checkpoint after EVERY question — an interrupted run never repeats a paid call.
        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": f"Phase 9 full-DEVELOPMENT benchmark — LIVE {args.pipeline} pipeline raw traces",
            "pipeline": args.pipeline,
            "split": "development",
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
