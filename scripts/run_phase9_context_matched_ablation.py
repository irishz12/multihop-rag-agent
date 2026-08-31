#!/usr/bin/env python
"""AUDIT ABLATION (dev-only, not a Phase 9 script): "Hybrid+Reranker,
context-matched, single pass" — isolates whether Agentic Multi-Hop RAG's
measured improvement over the fixed-5-chunk Hybrid+Reranker baseline comes
from iterative retrieval itself, or simply from passing more chunks to the
same frozen generation call.

For each of the 50 DEVELOPMENT-sample questions already used throughout
Phase 9 (results/phase9_sample.json), retrieves with the SAME frozen
`rerank_hybrid_search` (dense_top_k=bm25_top_k=RERANK_CANDIDATE_DEPTH=20,
RRF k=60) already used by the `hybrid_reranker` baseline — the ONLY thing
that varies is `final_top_k`, set PER QUESTION to that question's own
already-persisted Agentic Multi-Hop RAG `num_chunks_used_for_generation`
(from results/phase9_always_agentic_raw.json) instead of the baseline's
fixed `GENERATION_TOP_K = 5`. Generation uses the SAME frozen
`generate_answer` — same model, same prompt version, same temperature,
same pricing config, same 4,500-token budget — as every existing Phase 9
pipeline.

READ-ONLY w.r.t. every existing artifact: `results/phase9_sample.json` and
`results/phase9_always_agentic_raw.json` are only ever opened for reading.
This script's own output is a brand-new file
(results/phase9_hybrid_reranker_matched_raw.json) that does not collide
with any existing tracked or untracked results/*.json.

DEV-ONLY BY CONSTRUCTION: `DEV_SPLIT_FILE` is a hardcoded module constant
(same pattern as scripts/run_phase9_benchmark.py) — there is no CLI flag,
config option, or code path anywhere in this script that can reach
final_holdout.json. See tests/test_context_matched_ablation_guard.py for
the structural proof.

Requires $OPENAI_API_KEY in the environment (never printed/logged/persisted).

Usage:
    python scripts/run_phase9_context_matched_ablation.py

Writes results/phase9_hybrid_reranker_matched_raw.json, checkpointed after
every question (resumable, never repeats a completed paid call).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids, hop_count
from mhrag.generation.answer import generate_answer
from mhrag.generation.context import approximate_token_count
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker, rerank_hybrid_search

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json
SAMPLE_FILE = "results/phase9_sample.json"  # READ-ONLY — the frozen 50-qa_id Phase 9 sample
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"  # READ-ONLY — source of the per-question match target N
OUTPUT_FILE = "results/phase9_hybrid_reranker_matched_raw.json"  # this script's ONLY write target


def main() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")  # only for max_context_tokens (4500) — same budget as every pipeline

    sample = json.loads((PROJECT_ROOT / SAMPLE_FILE).read_text())
    sample_qa_ids = sample["qa_ids"]
    print(f"Loaded {len(sample_qa_ids)} qa_ids from {SAMPLE_FILE} (seed={sample.get('seed')})")

    agentic_raw = json.loads((PROJECT_ROOT / AGENTIC_RAW_FILE).read_text())
    agentic_by_qa_id = {r["qa_id"]: r for r in agentic_raw["records"]}
    missing = [q for q in sample_qa_ids if q not in agentic_by_qa_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} sample qa_id(s) have no persisted Agentic Multi-Hop RAG record in "
            f"{AGENTIC_RAW_FILE} — cannot determine their match target N: {missing}"
        )
    target_n_by_qa_id = {q: agentic_by_qa_id[q]["num_chunks_used_for_generation"] for q in sample_qa_ids}

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)  # DEVELOPMENT split only
    by_qa_id = {qa_id(r): r for r in all_records}
    missing_from_dev = [q for q in sample_qa_ids if q not in by_qa_id]
    if missing_from_dev:
        raise SystemExit(f"{len(missing_from_dev)} sample qa_id(s) not found in {DEV_SPLIT_FILE}: {missing_from_dev}")
    slice_records = [by_qa_id[q] for q in sample_qa_ids]  # the sample's own deterministic order

    out_path = PROJECT_ROOT / OUTPUT_FILE
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
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run the context-matched ablation: {exc}") from exc

    qwen_pricing = mantle_config["pricing"]
    max_context_tokens = agent_config_yaml["loop"]["max_context_tokens"]  # 4500 — same budget every pipeline uses

    n_processed_this_run = 0
    for i, record in enumerate(slice_records):
        rec_id = qa_id(record)
        if rec_id in existing:
            continue

        target_n = target_n_by_qa_id[rec_id]
        gold_ids = gold_doc_ids(record)
        rec_hop_count = hop_count(record)

        t0 = time.monotonic()
        retrieved = rerank_hybrid_search(
            record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            final_top_k=target_n,
        )
        retrieval_latency_ms = (time.monotonic() - t0) * 1000

        generation = generate_answer(
            record.query, retrieved, generation_client, approximate_token_count,
            top_k=target_n, max_context_tokens=max_context_tokens,
            input_price_per_million=qwen_pricing["input_per_million_tokens"],
            output_price_per_million=qwen_pricing["output_per_million_tokens"],
            prompt_version=mantle_config["generation"]["prompt_version"],
        )
        usage = generation.mantle_response.usage

        existing[rec_id] = {
            "qa_id": rec_id,
            "query": record.query,
            "question_type": record.question_type,
            "hop_count": rec_hop_count,
            "gold_answer": record.answer,
            "gold_docs_total": len(gold_ids),
            "target_n_chunks": target_n,  # matched to Agentic Multi-Hop RAG's realized chunk count for this qa_id
            "num_chunks_retrieved_before_budget": len(retrieved),
            "num_chunks_used_for_generation": len(generation.context.chunks_included),  # post-4500-token-budget, comparable field name to every other pipeline
            "num_chunks_dropped_for_budget": len(generation.context.chunks_dropped),
            "total_token_count": generation.context.total_token_count,
            "predicted_route": "HYBRID_RERANKER_CONTEXT_MATCHED",
            "num_retrieval_calls": 1, "num_reranker_calls": 1, "num_agent_hops": 0,
            "num_controller_calls": 0, "num_generation_calls": 1,
            "glm_input_tokens": 0, "glm_output_tokens": 0, "glm_cost_usd": None,
            "qwen_input_tokens": usage.input_tokens or 0, "qwen_output_tokens": usage.output_tokens or 0,
            "qwen_cost_usd": generation.cost.total_cost_usd, "total_cost_usd": generation.cost.total_cost_usd,
            "retrieval_latency_ms": retrieval_latency_ms, "reranking_latency_ms": 0.0,  # rerank_hybrid_search times retrieval+rerank as one call, matching the existing hybrid_reranker baseline's own field convention
            "controller_latency_ms": 0.0, "generation_latency_ms": generation.mantle_response.llm_latency_ms,
            "total_latency_ms": retrieval_latency_ms + generation.mantle_response.total_latency_ms,
            "stop_reason": "single_pass_context_matched",
            "answer": generation.answer,
            "evidence_doc_ids_used": sorted({c.doc_id for c in generation.context.chunks_included}),
        }
        n_processed_this_run += 1

        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "AUDIT ABLATION (dev-only) — Hybrid+Reranker, context-matched (per-question top_k = "
                       "Agentic Multi-Hop RAG's realized num_chunks_used_for_generation), single pass, no iteration",
            "pipeline": "hybrid_reranker_matched",
            "split": "development",
            "sample_file": SAMPLE_FILE,
            "match_target_source_file": AGENTIC_RAW_FILE,
            "n_questions_total": len(slice_records),
            "n_questions_completed": len(existing),
            "partial_run": len(existing) < len(slice_records),
            "records": list(existing.values()),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

        print(f"  [{i + 1}/{len(slice_records)}] qa_id={rec_id} target_n={target_n} "
              f"actual_n={len(generation.context.chunks_included)} cost=${generation.cost.total_cost_usd}")

    print(f"\n{'=' * 70}")
    print(f"This invocation processed {n_processed_this_run} new question(s)")
    print(f"Total accumulated: {len(existing)}/{len(slice_records)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
