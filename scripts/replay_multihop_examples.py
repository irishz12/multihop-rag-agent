#!/usr/bin/env python
"""Phase 6A, Option B (LIVE — real, small Mantle cost; approved by user for
exactly this scope): replay the existing, UNMODIFIED, frozen
`mhrag.agent.loop.run_agentic_retrieval` on ONLY the qa_ids already
selected by scripts/analyze_multihop_success.py's deterministic,
offline example-selection function — never a hand-picked or ad hoc list.

WHY THIS SCRIPT EXISTS: the original Phase 9 benchmark run
(scripts/run_phase9_benchmark.py) persisted only aggregate fields
(num_agent_hops, stop_reason, evidence_doc_ids_used) — it never serialized
the per-hop `HopRecord` trace `run_agentic_retrieval` builds internally, so
the verbatim hop-2/3 follow-up query text does not exist in any committed
artifact. This script recovers it for exactly the 5 portfolio-selected
questions by re-running the real production loop, unmodified, on the same
qa_ids, using the same frozen retrieval/controller/generation
config — every stage is zero-temperature (configs/agent.yaml,
configs/mantle.yaml), so this replay is EXPECTED to reproduce the
originally-persisted stop_reason/hop-count/evidence/answer for these 5
qa_ids exactly; any discrepancy is reported, never silently discarded (see
scripts/analyze_multihop_success.py's sibling report step run after this).

RESTRICTIONS (all structurally enforced, not just documented):
  - qa_ids come ONLY from results/multihop_success_analysis.json's
    `selected_example_qa_ids` field — this script has no CLI flag, no
    other input path, and cannot be pointed at an arbitrary qa_id list.
  - Every selected qa_id is looked up ONLY in data/processed/dev_subset.json
    (DEV_SPLIT_FILE is a hardcoded module constant) — if a selected qa_id
    is not found there, this script raises rather than silently falling
    back to any other file, so there is no code path that can reach
    final_holdout.json.
  - Writes ONLY results/multihop_examples_replay.json (a NEW file) —
    never modifies results/phase9_always_agentic_raw.json or any other
    existing artifact.
  - Checkpointed per-question (same pattern as run_phase9_benchmark.py):
    an interrupted run resumes without repeating a completed paid call.

Requires $OPENAI_API_KEY in the environment. Never printed/logged/persisted.

Usage:
    python scripts/replay_multihop_examples.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json
SELECTION_FILE = "results/multihop_success_analysis.json"  # READ-ONLY — sole source of qa_ids to replay
OUTPUT_FILE = "results/multihop_examples_replay.json"  # this script's ONLY write target


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def _hop_record_to_dict(hop) -> dict:
    return {
        "hop_number": hop.hop_number,
        "query": hop.query,  # the verbatim query for this hop — hop 1 == record.query, hops 2/3 == controller output
        "retrieved_chunks": [
            {"rank": c.rank, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "score": c.score}
            for c in hop.chunk_results
        ],
        "new_chunk_ids": list(hop.new_chunk_ids),
        "duplicate_chunk_ids": list(hop.duplicate_chunk_ids),
    }


def main() -> None:
    selection = _load(SELECTION_FILE)
    selected_qa_ids = selection["selected_example_qa_ids"]
    assert 1 <= len(selected_qa_ids) <= 5, f"expected 1-5 selected qa_ids, got {len(selected_qa_ids)}"
    print(f"Replaying ONLY these {len(selected_qa_ids)} qa_ids (from {SELECTION_FILE}): {selected_qa_ids}")

    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)
    records_by_qa_id = {compute_qa_id(r): r for r in all_records}

    missing = [q for q in selected_qa_ids if q not in records_by_qa_id]
    if missing:
        raise SystemExit(f"{len(missing)} selected qa_id(s) not found in {DEV_SPLIT_FILE}: {missing}")
    records = [records_by_qa_id[q] for q in selected_qa_ids]  # exactly the 5, in selection order

    out_path = PROJECT_ROOT / OUTPUT_FILE
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = {r["qa_id"]: r for r in json.loads(out_path.read_text()).get("records", [])}
        print(f"Found existing {out_path} with {len(existing)} completed record(s) — will skip those")

    print(f"Loading embedding model {retrieval_config['embedding']['model_name']} ...")
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
        raise SystemExit(f"Cannot run live replay: {exc}") from exc

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

    for record in records:
        qid = compute_qa_id(record)
        if qid in existing:
            print(f"  skip (already replayed) {qid}")
            continue

        trace = run_agentic_retrieval(
            record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            controller_client, generation_client, config=agentic_config,
        )

        existing[qid] = {
            "qa_id": qid,
            "query": record.query,
            "question_type": record.question_type,
            "hops": [_hop_record_to_dict(h) for h in trace.hops],
            "stop_reason": trace.stop_reason,
            "num_retrieval_calls": trace.num_retrieval_calls,
            "num_controller_calls": trace.num_controller_calls,
            "evidence_doc_ids_used_final": sorted({c.doc_id for c in trace.final_generation.context.chunks_included}),
            "answer": trace.final_generation.answer,
            "total_cost_usd": trace.cost.total_cost_usd,
            "total_latency_ms": trace.total_latency_ms,
        }

        # Checkpoint after EVERY question — an interrupted run never repeats a paid call.
        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "Phase 6A Option B — LIVE replay of the frozen, unmodified agentic loop on ONLY "
                       "the 5 qa_ids selected by scripts/analyze_multihop_success.py's deterministic "
                       "example-selection function, to recover the verbatim per-hop query trace "
                       "(HopRecord) the original Phase 9 benchmark run never persisted.",
            "selected_from": SELECTION_FILE,
            "split": "development",
            "n_questions_total": len(records),
            "n_questions_completed": len(existing),
            "records": [existing[q] for q in selected_qa_ids if q in existing],
        }
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"  replayed {qid}: stop_reason={trace.stop_reason} hops={trace.num_retrieval_calls} "
              f"cost=${trace.cost.total_cost_usd}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
