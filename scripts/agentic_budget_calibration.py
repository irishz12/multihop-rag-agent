#!/usr/bin/env python
"""Phase 7.1: LIVE token-budget calibration sweep for the Agentic
Multi-Hop RAG loop. Makes REAL API calls (GLM 4.7 Flash controller + Qwen final answer)
to Amazon Bedrock Mantle and incurs REAL (small) cost. Run explicitly, once
per candidate budget (kept separate, not looped in-process, so each run
fits comfortably under this environment's foreground execution window):

    python scripts/agentic_budget_calibration.py --budget 3000
    python scripts/agentic_budget_calibration.py --budget 4500
    python scripts/agentic_budget_calibration.py --budget 6000

Reads ONLY data/processed/dev_subset.json — DEV_SPLIT_FILE is a hardcoded
module constant, no CLI flag, no config option, so there is no code path in
this script that can reach final_holdout.json. TOKEN_BUDGETS is likewise a
hardcoded module constant — `--budget` may only select one of the three
values already in that tuple, it cannot introduce a fourth.

This script changes ONLY `max_context_tokens` (via
`mhrag.calibration.sweep.build_swept_configs`). Every other AgenticConfig
field — controller model, generation model, max_hops, hop_top_k,
max_evidence_chunks, timeout_seconds, prompt versions — comes unmodified
from configs/agent.yaml / configs/mantle.yaml, exactly as
scripts/agentic_smoke_check.py builds it. It does not redesign the agent.

The calibration sample (27 DEVELOPMENT questions, balanced across
question_type and hop_count — see mhrag.calibration.sample) is identical
across all three budget runs: `select_calibration_sample` is deterministic
(fixed seed), so the SAME 27 questions are used for 3000, 4500, and 6000.

Only `record.query` ever reaches the agent (`run_calibration_query` forwards
it to `run_agentic_retrieval` exactly as Phase 7 does). Gold `answer`,
`evidence_list`, and `question_type` are used only by
`mhrag.calibration.sweep.evaluate_against_gold`, strictly AFTER each
question's trace has already been returned — see that module's docstring
for the structural guarantee.

Requires `$OPENAI_API_KEY` in the environment. The key is never printed,
logged, or written to any output file by this script.

Writes results/agentic_budget_calibration_<budget>.json — raw per-question
results for that one budget. A separate script,
scripts/agentic_budget_calibration_decide.py, merges all three files and
applies the objective selection rule (mhrag.calibration.decision).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone

from mhrag.agent.loop import AgenticConfig
from mhrag.calibration.sample import select_calibration_sample
from mhrag.calibration.sweep import run_calibration_query
from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import hop_count
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker

# Hardcoded to the development split ONLY — no CLI flag, no config option,
# so there is no code path in this script that can reach final_holdout.json.
DEV_SPLIT_FILE = "dev_subset.json"

# The exact 3 candidate budgets this phase compares. `--budget` may only
# select one of these — it cannot introduce a fourth value.
TOKEN_BUDGETS = (3000, 4500, 6000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, required=True, choices=TOKEN_BUDGETS)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--mantle-config", default="configs/mantle.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--output", default=None, help="default: results/agentic_budget_calibration_<budget>.json")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="run only the first N of the 27 calibration questions (dry-run / cost-control only; "
             "does not change which split or budgets are used, and the omitted results simply are not "
             "part of this file's aggregate — a --limit run is NOT a substitute for the full 27-question sweep)",
    )
    args = parser.parse_args()

    dataset_config = load_config(args.config)
    retrieval_config = load_config(args.retrieval_config)
    mantle_config = load_config(args.mantle_config)
    agent_config_yaml = load_config(args.agent_config)

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    dev_records = load_qa_records(dev_path)
    print(f"Loaded {len(dev_records)} DEVELOPMENT records from {dev_path}")

    sample = select_calibration_sample(dev_records)
    print(f"Calibration sample: {len(sample)} questions (deterministic, same set across all budgets)")
    if args.limit is not None:
        sample = sample[: args.limit]
        print(f"--limit {args.limit}: running only the first {len(sample)} (dry-run / cost-control only)")

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
        raise SystemExit(f"Cannot run live budget calibration: {exc}") from exc

    loop_cfg = agent_config_yaml["loop"]
    agent_pricing = agent_config_yaml["pricing"]
    qwen_pricing = mantle_config["pricing"]
    base_config = AgenticConfig(
        max_hops=loop_cfg["max_hops"],
        hop_top_k=loop_cfg["hop_top_k"],
        max_evidence_chunks=loop_cfg["max_evidence_chunks"],
        max_context_tokens=loop_cfg["max_context_tokens"],  # overwritten below by --budget
        timeout_seconds=loop_cfg["timeout_seconds"],
        controller_prompt_version=agent_config_yaml["controller"]["prompt_version"],
        generation_prompt_version=mantle_config["generation"]["prompt_version"],
        glm_input_price_per_million=agent_pricing["input_per_million_tokens"],
        glm_output_price_per_million=agent_pricing["output_per_million_tokens"],
        qwen_input_price_per_million=qwen_pricing["input_per_million_tokens"],
        qwen_output_price_per_million=qwen_pricing["output_per_million_tokens"],
    )
    config = dataclasses.replace(base_config, max_context_tokens=args.budget)
    print(f"\n=== Running budget calibration: max_context_tokens={config.max_context_tokens} ===")
    print(f"(every other config field unchanged from configs/agent.yaml: {base_config})")

    results = []
    total_cost = 0.0
    stop_reason_counts: dict[str, int] = {}

    for i, record in enumerate(sample):
        rec_id = qa_id(record)
        rec_hop_count = hop_count(record)
        print(f"\n{'=' * 70}\n[{i + 1}/{len(sample)}] qa_id={rec_id} "
              f"type={record.question_type} hop_count={rec_hop_count}\n  {record.query}")

        result = run_calibration_query(
            record, qa_id=rec_id, hop_count=rec_hop_count,
            qdrant_client=qdrant_client, collection_name=collection_name,
            embedding_model=embedding_model, bm25_model=bm25_model, reranker=reranker,
            controller_client=controller_client, generation_client=generation_client,
            config=config,
        )
        trace = result.trace
        evaluation = result.evaluation

        print(f"  STOP REASON: {trace.stop_reason} | hops={trace.num_retrieval_calls} "
              f"controller_calls={trace.num_controller_calls}")
        print(f"  evidence: {trace.unique_documents_retrieved} unique docs, "
              f"recall={evaluation.recall:.2f} complete_evidence={evaluation.complete_evidence} "
              f"new_docs_per_hop={evaluation.new_unique_docs_per_hop}")
        print(f"  cost=${trace.cost.total_cost_usd} latency={trace.total_latency_ms:.0f}ms")

        stop_reason_counts[trace.stop_reason] = stop_reason_counts.get(trace.stop_reason, 0) + 1
        if trace.cost.total_cost_usd is not None:
            total_cost += trace.cost.total_cost_usd

        results.append(
            {
                "qa_id": result.qa_id,
                "question_type": result.question_type,
                "hop_count": result.hop_count,
                "stop_reason": trace.stop_reason,
                "num_retrieval_calls": trace.num_retrieval_calls,
                "num_controller_calls": trace.num_controller_calls,
                "num_generation_calls": trace.num_generation_calls,
                "hops": [
                    {
                        "hop_number": h.hop_number,
                        "num_chunks_retrieved": len(h.chunk_results),
                        "num_new_chunks": len(h.new_chunk_ids),
                        "num_duplicate_chunks": len(h.duplicate_chunk_ids),
                        "retrieval_latency_ms": h.retrieval_latency_ms,
                        "reranking_latency_ms": h.reranking_latency_ms,
                    }
                    for h in trace.hops
                ],
                "evidence": {
                    "unique_chunks_retrieved": trace.unique_chunks_retrieved,
                    "unique_documents_retrieved": trace.unique_documents_retrieved,
                    "duplicate_chunks_removed": trace.duplicate_chunks_removed,
                    "chunks_passed_to_final_generation": trace.chunks_passed_to_final_generation,
                    "chunks_dropped_for_budget": trace.chunks_dropped_for_budget,
                },
                "evaluation": {
                    "gold_doc_count": len(evaluation.gold_doc_ids),
                    "recall": evaluation.recall,
                    "complete_evidence": evaluation.complete_evidence,
                    "new_unique_docs_per_hop": list(evaluation.new_unique_docs_per_hop),
                },
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
        "purpose": "Phase 7.1 token-budget calibration sweep (development-only tuning)",
        "split": "development",
        "token_budget": config.max_context_tokens,
        "config": dataclasses.asdict(config),
        "n_questions": len(sample),
        "stop_reason_distribution": stop_reason_counts,
        "total_cost_usd": total_cost,
        "results": results,
    }
    out_path = PROJECT_ROOT / (args.output or f"results/agentic_budget_calibration_{args.budget}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 70}")
    print(f"budget={args.budget} stop reasons: {stop_reason_counts}")
    print(f"Total calibration cost for this budget: ${total_cost:.6f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
