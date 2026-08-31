#!/usr/bin/env python
"""LIVE Agentic Multi-Hop RAG smoke check.

Makes REAL API calls (GLM 4.7 Flash controller + Qwen final answer) to
Amazon Bedrock Mantle and incurs REAL (small) cost. Connectivity/behavior
validation, NOT answer-quality evaluation — not run by pytest, run
explicitly:

    python scripts/agentic_smoke_check.py

Reads ONLY data/processed/smoke_subset.json — this module has no code path
that can reach final_holdout.json (no CLI flag, no config option; the
split file is a hardcoded module constant, same guard pattern as every
other script in this project). Does NOT run against the full development
benchmark.

Default `--indices` selects 8 SMOKE questions spanning what Phase 6 (single
retrieval, no agentic loop) did with them: some it answered directly with
one retrieval, some it explicitly declined as insufficient, plus three
untested 4-hop questions representing harder multi-hop behavior — so this
run can show whether the agentic loop recovers evidence Phase 6 could not
reach in a single hop.

For each question, prints: initial retrieval -> controller decision ->
follow-up query(s) -> evidence growth per hop -> final answer, plus full
cost/latency/stop-reason tracing.

Only `query` text and retrieved chunk text ever reach the controller or the
final-answer model — never gold `answer`, `evidence_list`, or
`question_type` (see mhrag.agent.controller / mhrag.generation.answer
module docstrings for the structural guarantee).

Requires `$OPENAI_API_KEY` in the environment. The key is never printed,
logged, or written to any output file by this script.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.config import PROJECT_ROOT, load_config
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker

# Hardcoded to the smoke split ONLY — no CLI flag, no config option, so
# there is no code path in this script that can reach final_holdout.json.
SMOKE_SPLIT_FILE = "smoke_subset.json"

# 8 SMOKE indices spanning: Phase 6 answered directly (1, 2, 4), Phase 6
# explicitly declined as insufficient (0, 3), and untested 4-hop questions
# representing harder multi-hop behavior (14, 19, 25). See module docstring.
DEFAULT_INDICES = [0, 1, 2, 3, 4, 14, 19, 25]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices", type=int, nargs="+", default=DEFAULT_INDICES,
        help="smoke_subset.json indices to run (default: 8 curated examples)",
    )
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--mantle-config", default="configs/mantle.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--output", default="results/agentic_smoke_check.json")
    args = parser.parse_args()

    if len(args.indices) > 10:
        print(f"Warning: {len(args.indices)} questions exceeds the recommended 5-10 for this phase.")

    dataset_config = load_config(args.config)
    retrieval_config = load_config(args.retrieval_config)
    mantle_config = load_config(args.mantle_config)
    agent_config_yaml = load_config(args.agent_config)

    smoke_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / SMOKE_SPLIT_FILE
    all_records = json.loads(smoke_path.read_text())
    records = [all_records[i] for i in args.indices]
    print(f"Loaded {len(records)} smoke questions (indices {args.indices}) from {smoke_path}")

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
        raise SystemExit(f"Cannot run live agentic smoke check: {exc}") from exc

    loop_cfg = agent_config_yaml["loop"]
    agent_pricing = agent_config_yaml["pricing"]
    qwen_pricing = mantle_config["pricing"]
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
    total_cost = 0.0
    stop_reason_counts: dict[str, int] = {}

    for i, rec in zip(args.indices, records):
        print(f"\n{'=' * 70}\n[smoke index {i}] {rec['query']}")
        trace = run_agentic_retrieval(
            rec["query"], qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            controller_client, generation_client, config=config,
        )

        for hop in trace.hops:
            print(f"  hop {hop.hop_number}: query={hop.query!r}")
            print(
                f"    retrieved {len(hop.chunk_results)} chunks "
                f"(+{len(hop.new_chunk_ids)} new, {len(hop.duplicate_chunk_ids)} dup) "
                f"retrieval={hop.retrieval_latency_ms:.0f}ms rerank={hop.reranking_latency_ms:.0f}ms"
            )
        for j, cr in enumerate(trace.controller_results):
            print(
                f"  controller call {j + 1}: sufficient={cr.decision.sufficient} "
                f"next_query={cr.decision.next_query!r} reason={cr.decision.reason!r} "
                f"fallback_used={cr.fallback_used}"
            )
        print(f"  STOP REASON: {trace.stop_reason}")
        print(f"  evidence: {trace.unique_chunks_retrieved} unique chunks, "
              f"{trace.unique_documents_retrieved} unique docs, "
              f"{trace.duplicate_chunks_removed} duplicates removed, "
              f"{trace.chunks_passed_to_final_generation} passed to generation, "
              f"{trace.chunks_dropped_for_budget} dropped for budget")
        print(f"  answer: {trace.final_generation.answer[:200]!r}")
        print(
            f"  cost: GLM in={trace.cost.glm_input_tokens} out={trace.cost.glm_output_tokens} "
            f"(${trace.cost.glm_cost_usd}); Qwen in={trace.cost.qwen_input_tokens} "
            f"out={trace.cost.qwen_output_tokens} (${trace.cost.qwen_cost_usd}); "
            f"total=${trace.cost.total_cost_usd}"
        )
        print(
            f"  latency: retrieval={trace.retrieval_latency_ms:.0f}ms rerank={trace.reranking_latency_ms:.0f}ms "
            f"controller={trace.controller_latency_ms:.0f}ms generation={trace.generation_latency_ms:.0f}ms "
            f"total={trace.total_latency_ms:.0f}ms"
        )

        stop_reason_counts[trace.stop_reason] = stop_reason_counts.get(trace.stop_reason, 0) + 1
        if trace.cost.total_cost_usd is not None:
            total_cost += trace.cost.total_cost_usd

        results.append(
            {
                "smoke_index": i,
                "query": rec["query"],
                "question_type_for_display_only": rec["question_type"],
                "stop_reason": trace.stop_reason,
                "num_hops": trace.num_retrieval_calls,
                "num_controller_calls": trace.num_controller_calls,
                "hops": [
                    {
                        "hop_number": h.hop_number,
                        "query": h.query,
                        "num_chunks_retrieved": len(h.chunk_results),
                        "new_chunk_ids": list(h.new_chunk_ids),
                        "duplicate_chunk_ids": list(h.duplicate_chunk_ids),
                        "retrieval_latency_ms": h.retrieval_latency_ms,
                        "reranking_latency_ms": h.reranking_latency_ms,
                    }
                    for h in trace.hops
                ],
                "controller_decisions": [
                    {
                        "sufficient": cr.decision.sufficient,
                        "next_query": cr.decision.next_query,
                        "reason": cr.decision.reason,
                        "fallback_used": cr.fallback_used,
                    }
                    for cr in trace.controller_results
                ],
                "evidence": {
                    "unique_chunks_retrieved": trace.unique_chunks_retrieved,
                    "unique_documents_retrieved": trace.unique_documents_retrieved,
                    "duplicate_chunks_removed": trace.duplicate_chunks_removed,
                    "chunks_passed_to_final_generation": trace.chunks_passed_to_final_generation,
                    "chunks_dropped_for_budget": trace.chunks_dropped_for_budget,
                },
                "answer": trace.final_generation.answer,
                "cost_usd": {
                    "glm_input_tokens": trace.cost.glm_input_tokens,
                    "glm_output_tokens": trace.cost.glm_output_tokens,
                    "glm_cost": trace.cost.glm_cost_usd,
                    "qwen_input_tokens": trace.cost.qwen_input_tokens,
                    "qwen_output_tokens": trace.cost.qwen_output_tokens,
                    "qwen_cost": trace.cost.qwen_cost_usd,
                    "total": trace.cost.total_cost_usd,
                },
                "latency_ms": {
                    "retrieval": trace.retrieval_latency_ms,
                    "reranking": trace.reranking_latency_ms,
                    "controller": trace.controller_latency_ms,
                    "generation": trace.generation_latency_ms,
                    "total": trace.total_latency_ms,
                },
            }
        )

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "agentic connectivity/behavior smoke validation — NOT answer-quality evaluation",
        "split": "smoke",
        "controller_model": agent_config_yaml["controller"]["model_id"],
        "generation_model": mantle_config["generation"]["model_id"],
        "n_questions": len(records),
        "stop_reason_distribution": stop_reason_counts,
        "total_cost_usd": total_cost,
        "config": {"loop": loop_cfg, "controller_pricing": agent_pricing, "generation_pricing": qwen_pricing},
        "results": results,
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 70}")
    print(f"Stop reasons: {stop_reason_counts}")
    print(f"Total smoke-test cost: ${total_cost:.6f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
