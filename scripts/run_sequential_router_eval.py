#!/usr/bin/env python
"""Phase 8A.1: LIVE evaluation of the evidence-aware sequential router over
ALL 265 non-null DEVELOPMENT questions.

Runs the frozen retrieval pipeline (Dense+BM25 -> deterministic RRF k=60 ->
top-20 candidate pool; cross-encoder reranker, BAAI/bge-reranker-base, only
when Gate 1 says insufficient) plus the GLM 4.7 Flash Evidence Sufficiency
Gate (`mhrag.routing.sequential_router.route_question_sequential`) for
every question — NO heuristic shortcut this phase (Gate 1 always runs for
real). Retrieval/reranker config is unchanged from Phases 2-5; nothing is
tuned here.

Reads ONLY data/processed/dev_subset.json — DEV_SPLIT_FILE is a hardcoded
module constant, no CLI flag, no config option, so there is no code path
in this script that can reach final_holdout.json.

Chunked execution (CPU-quota-throttling workaround, same pattern as
scripts/run_retrieval_eval.py and Phase 7.1's calibration runs): supports
`--offset`/`--limit` to process a slice per foreground invocation. Results
are UPSERTED by qa_id into the existing --output file across runs (each
invocation loads what's already there, merges its own slice's results in,
and rewrites the whole file), so multiple chunked invocations accumulate
into one complete artifact without a separate merge step.

Does NOT write to, read from, or otherwise reference any Phase 8A output
file (router_dataset.json / router_split.json / router_thresholds.json /
router_validation_report.json) — those are preserved as the Phase 8A
baseline, untouched (see tests/test_sequential_router_eval_guard.py).

Usage:
    python scripts/run_sequential_router_eval.py --offset 0 --limit 90
    python scripts/run_sequential_router_eval.py --offset 90 --limit 90
    python scripts/run_sequential_router_eval.py --offset 180

Writes results/sequential_router_eval_raw.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker
from mhrag.routing.sequential_router import route_question_sequential

# Hardcoded to the development split ONLY — no CLI flag, no config option,
# so there is no code path in this script that can reach final_holdout.json.
DEV_SPLIT_FILE = "dev_subset.json"
DEFAULT_OUTPUT = "results/sequential_router_eval_raw.json"


def _serialize_chunks(results):
    return [{"chunk_id": r.chunk_id, "doc_id": r.doc_id, "rank": r.rank, "score": r.score} for r in results]


def _serialize_gate(gate_result):
    if gate_result is None:
        return None
    d = gate_result.decision
    return {
        "sufficient": d.sufficient,
        "raw_sufficient": d.raw_sufficient,
        "supporting_chunk_ids": list(d.supporting_chunk_ids),
        "missing_information": list(d.missing_information),
        "reason": d.reason,
        "conservative_override": d.conservative_override,
        "fallback_used": gate_result.fallback_used,
        "input_tokens": gate_result.mantle_response.usage.input_tokens,
        "output_tokens": gate_result.mantle_response.usage.output_tokens,
        "latency_ms": gate_result.mantle_response.llm_latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="max questions to process starting at --offset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)
    non_null_records = [r for r in all_records if r.question_type != "null_query"]
    end = None if args.limit is None else args.offset + args.limit
    slice_records = non_null_records[args.offset : end]
    print(f"Loaded {len(non_null_records)} non-null DEVELOPMENT records; processing "
          f"[{args.offset}:{end if end is not None else len(non_null_records)}] = {len(slice_records)} questions")

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
        raise SystemExit(f"Cannot run live sequential router eval: {exc}") from exc

    glm_pricing = agent_config_yaml["pricing"]

    out_path = PROJECT_ROOT / args.output
    existing_records: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        existing_records = {r["qa_id"]: r for r in existing.get("records", [])}
        print(f"Found existing {out_path} with {len(existing_records)} record(s) — will upsert this slice into it")

    stop_reason_counts: dict[str, int] = {}
    total_cost = 0.0

    for i, record in enumerate(slice_records):
        rec_id = qa_id(record)
        result = route_question_sequential(
            record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker, glm_client,
            glm_input_price_per_million=glm_pricing["input_per_million_tokens"],
            glm_output_price_per_million=glm_pricing["output_per_million_tokens"],
        )

        stop_reason_counts[result.route] = stop_reason_counts.get(result.route, 0) + 1
        if result.glm_cost is not None and result.glm_cost.total_cost_usd is not None:
            total_cost += result.glm_cost.total_cost_usd

        existing_records[rec_id] = {
            "qa_id": rec_id,
            "question_type": record.question_type,
            "query": record.query,
            "route": result.route,
            "hybrid_top5": _serialize_chunks(result.hybrid_top5),
            "reranked_top5": _serialize_chunks(result.reranked_top5) if result.reranked_top5 is not None else None,
            "gate1": _serialize_gate(result.gate1_result),
            "gate2": _serialize_gate(result.gate2_result),
            "num_glm_calls": result.num_glm_calls,
            "glm_input_tokens": result.glm_input_tokens,
            "glm_output_tokens": result.glm_output_tokens,
            "glm_cost_usd": result.glm_cost.total_cost_usd if result.glm_cost else None,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "reranking_latency_ms": result.reranking_latency_ms,
            "gate_latency_ms": result.gate_latency_ms,
            "total_latency_ms": result.total_latency_ms,
        }

        if (i + 1) % 20 == 0 or (i + 1) == len(slice_records):
            print(f"  [{i + 1}/{len(slice_records)}] qa_id={rec_id} route={result.route} "
                  f"glm_calls={result.num_glm_calls}")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A.1 evidence-aware sequential router — LIVE evaluation over all "
                   "265 non-null DEVELOPMENT questions",
        "split": "development",
        "n_questions_total": len(non_null_records),
        "n_questions_completed": len(existing_records),
        "partial_run": len(existing_records) < len(non_null_records),
        "route_distribution_this_slice": stop_reason_counts,
        "this_slice_cost_usd": total_cost,
        "records": list(existing_records.values()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 70}")
    print(f"This slice: {stop_reason_counts}, cost=${total_cost:.6f}")
    print(f"Total accumulated: {len(existing_records)}/{len(non_null_records)} questions")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
