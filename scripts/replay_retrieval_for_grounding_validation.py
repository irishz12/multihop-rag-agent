#!/usr/bin/env python
"""Phase 5A, STEP 2 (dev-only, VALIDATION ONLY, NO LLM/API CALLS — local
retrieval/rerank models only): replay the existing frozen retrieval
pipeline for the 265 non-null DEVELOPMENT questions, for three pipelines:

  A. hybrid_reranker       — the existing frozen `rerank_hybrid_search`,
                             final_top_k = GENERATION_TOP_K = 5 (same
                             constant scripts/run_phase9_benchmark.py uses)
  B. hybrid_reranker_matched — same frozen function, final_top_k = N,
                             where N is read from the ALREADY-PERSISTED
                             results/phase9_always_agentic_raw.json's
                             num_chunks_used_for_generation for that
                             qa_id — the exact same source this session's
                             earlier context-matched ablation used
  C. agentic_hop1          — hop 1 of the agentic loop. NOTE (a genuine
                             validation finding, not a shortcut): hop 1's
                             retrieval config (dense_top_k=bm25_top_k=
                             RERANK_CANDIDATE_DEPTH=20, RRF k=60, rerank
                             top_k=hop_top_k=5) is IDENTICAL to pipeline
                             A's, and hop 1's query is always the
                             original question — the same query pipeline
                             A uses. So A and C are mathematically
                             GUARANTEED to produce identical output for
                             the same qa_id; this script computes each
                             query's retrieval ONCE and labels the result
                             under both "hybrid_reranker" and
                             "agentic_hop1", rather than paying for (or
                             risking floating-point drift from) calling
                             the identical function twice.

THE QUERY PASSED TO RETRIEVAL IS ALWAYS `record["query"]` — the original
question text, read from data/processed/dev_subset.json (never
final_holdout.json). Never `Evidence.fact`, never the gold answer, never
any derived/concatenated string. See
tests/test_replay_retrieval_for_grounding_validation_guard.py.

Writes ONLY results/fact_grounding_replay_raw.json — never modifies
results/phase9_hybrid_reranker_raw.json, results/phase9_always_agentic_raw.json,
results/phase9_hybrid_reranker_matched_full_raw.json, or any other
existing artifact.

Usage:
    python scripts/replay_retrieval_for_grounding_validation.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker, rerank_hybrid_search

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"  # READ-ONLY — source of pipeline B's target N
GENERATION_TOP_K = 5  # matches scripts/run_phase9_benchmark.py's GENERATION_TOP_K exactly
OUTPUT_FILE = "results/fact_grounding_replay_raw.json"  # this script's ONLY write target


def _retrieved_record(query: str, results, latency_ms: float) -> dict:
    return {
        "query": query,
        "replayed_chunk_ids": [r.chunk_id for r in results],
        "replayed_doc_ids_ranked": [r.doc_id for r in results],  # rank order, may repeat a doc_id across chunks
        "replayed_doc_ids_unique": sorted({r.doc_id for r in results}),
        "ranks": [r.rank for r in results],
        "scores": [r.score for r in results],
        "rerank_scores": [r.rerank_score for r in results],
        "num_replayed_chunks": len(results),
        "retrieval_latency_ms": latency_ms,
    }


def main() -> None:
    import time

    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)
    non_null = [r for r in all_records if r.question_type != "null_query"]
    print(f"Non-null development questions: {len(non_null)}")

    agentic_raw = json.loads((PROJECT_ROOT / AGENTIC_RAW_FILE).read_text())
    target_n_by_qa_id = {r["qa_id"]: r["num_chunks_used_for_generation"] for r in agentic_raw["records"]}
    missing_n = [compute_qa_id(r) for r in non_null if compute_qa_id(r) not in target_n_by_qa_id]
    print(f"Records missing an Agentic target-N (pipeline B will be skipped for these): {len(missing_n)}")

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

    out_path = PROJECT_ROOT / OUTPUT_FILE
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text()).get("records", {})

    for i, record in enumerate(non_null):
        rec_id = compute_qa_id(record)
        if rec_id in existing:
            continue

        query = record.query  # THE ONLY THING passed to retrieval — never Evidence.fact, never gold answer

        t0 = time.monotonic()
        top5 = rerank_hybrid_search(
            query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            final_top_k=GENERATION_TOP_K,
        )
        lat_a = (time.monotonic() - t0) * 1000
        pipeline_a = _retrieved_record(query, top5, lat_a)

        entry = {
            "qa_id": rec_id, "question_type": record.question_type,
            "hybrid_reranker": pipeline_a,
            "agentic_hop1": pipeline_a,  # IDENTICAL computation — see module docstring; not recomputed
            "agentic_hop1_note": "identical retrieval config + identical query to hybrid_reranker; "
                                  "computed once, see module docstring",
        }

        target_n = target_n_by_qa_id.get(rec_id)
        if target_n is not None:
            t0 = time.monotonic()
            topn = rerank_hybrid_search(
                query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                final_top_k=target_n,
            )
            lat_b = (time.monotonic() - t0) * 1000
            entry["hybrid_reranker_matched"] = {**_retrieved_record(query, topn, lat_b), "target_n": target_n}
        else:
            entry["hybrid_reranker_matched"] = None

        existing[rec_id] = entry

        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "Phase 5A STEP 2 (validation only) — retrieval replay for fact-grounding validation, "
                       "dev-only, zero LLM/API calls (local retrieval/rerank models only)",
            "split": "development",
            "query_source": "record.query (data/processed/dev_subset.json) — never Evidence.fact, never gold answer",
            "n_questions_total": len(non_null),
            "n_questions_completed": len(existing),
            "records": existing,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

        if (i + 1) % 20 == 0 or (i + 1) == len(non_null):
            print(f"  [{i + 1}/{len(non_null)}] qa_id={rec_id}")

    print(f"\nTotal accumulated: {len(existing)}/{len(non_null)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
