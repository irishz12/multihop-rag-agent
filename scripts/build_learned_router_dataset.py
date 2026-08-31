#!/usr/bin/env python
"""Phase 8A.2: build the learned-router training dataset — LIVE Qdrant +
embedding + BM25 + cross-encoder reranking, NO Mantle/LLM calls anywhere
(this script has zero LLM cost).

For every non-null DEVELOPMENT question, computes:
  - `QueryFeatures` + `RetrievalSignals` (`mhrag.routing.features`,
    unmodified) from the frozen Hybrid RRF baseline's own dense/BM25/fused
    output — exactly Stage 1's runtime feature set.
  - `RerankSignals` (`mhrag.routing.rerank_features`, new this phase) —
    computed UNCONDITIONALLY (every question's fused-20 pool is reranked
    regardless of what any gate/model would decide), so Stage 2 has clean,
    complete training data for all 265 questions, not just the ones a real
    router would have escalated.
  - `oracle_route` (`mhrag.routing.oracle`, sourced from the already-frozen
    `results/retrieval_eval_development.json` — no new retrieval run needed
    to produce it, exactly Phase 8A's `build_router_dataset.py` pattern).

Reads ONLY data/processed/dev_subset.json — DEV_SPLIT_FILE is a hardcoded
module constant, no CLI flag, no config option, so there is no code path in
this script that can reach final_holdout.json.

Chunked execution (CPU-quota-throttling workaround, same pattern as
scripts/run_sequential_router_eval.py): supports `--offset`/`--limit` to
process a slice per foreground invocation. Results are UPSERTED by qa_id
into the existing --output file across runs.

Does NOT write to, read from (other than the frozen, already-existing
results/retrieval_eval_development.json), or otherwise reference any prior
Phase 8A/8A.1 output file (router_dataset.json / router_split.json /
router_thresholds.json / router_validation_report.json /
sequential_router_eval_raw.json / router_full_dev_eval.json /
sequential_router_report.json) — those remain untouched (see
tests/test_build_learned_router_dataset_guard.py).

Usage:
    python scripts/build_learned_router_dataset.py --offset 0 --limit 90
    python scripts/build_learned_router_dataset.py --offset 90 --limit 90
    python scripts/build_learned_router_dataset.py --offset 180

Writes results/learned_router_dataset.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import RERANK_CANDIDATE_DEPTH, Reranker, rerank_results
from mhrag.retrieval.rrf import RRF_K, rrf_fuse
from mhrag.routing.features import extract_query_features, extract_retrieval_signals
from mhrag.routing.oracle import compute_oracle_labels
from mhrag.routing.rerank_features import extract_rerank_signals

# Hardcoded to the development split ONLY — no CLI flag, no config option,
# so there is no code path in this script that can reach final_holdout.json.
DEV_SPLIT_FILE = "dev_subset.json"
RETRIEVAL_EVAL_ARTIFACT = "results/retrieval_eval_development.json"
DEFAULT_OUTPUT = "results/learned_router_dataset.json"
TOP_K = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="max questions to process starting at --offset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    dev_records = load_qa_records(dev_path)
    dev_by_qa_id = {qa_id(r): r for r in dev_records if r.question_type != "null_query"}
    print(f"Loaded {len(dev_records)} DEVELOPMENT records ({len(dev_by_qa_id)} non-null) from {dev_path}")

    eval_artifact_path = PROJECT_ROOT / RETRIEVAL_EVAL_ARTIFACT
    eval_artifact = json.loads(eval_artifact_path.read_text())
    oracle_labels = compute_oracle_labels(eval_artifact)
    print(f"Computed {len(oracle_labels)} oracle route labels from {eval_artifact_path}")

    missing = [label.qa_id for label in oracle_labels if label.qa_id not in dev_by_qa_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} oracle label qa_id(s) not found in dev_subset.json — "
            f"e.g. {missing[:5]} (dev_subset.json and the frozen retrieval eval artifact must "
            "come from the exact same DEVELOPMENT split)"
        )

    end = None if args.limit is None else args.offset + args.limit
    slice_labels = oracle_labels[args.offset : end]
    print(f"Processing [{args.offset}:{end if end is not None else len(oracle_labels)}] = "
          f"{len(slice_labels)} questions")

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
    dense_top_k = retrieval_config["hybrid"]["dense_top_k"]
    bm25_top_k = retrieval_config["hybrid"]["bm25_top_k"]

    out_path = PROJECT_ROOT / args.output
    existing_records: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        existing_records = {r["qa_id"]: r for r in existing.get("records", [])}
        print(f"Found existing {out_path} with {len(existing_records)} record(s) — will upsert this slice into it")

    for i, label in enumerate(slice_labels):
        record = dev_by_qa_id[label.qa_id]
        question = record.query

        dense_results = dense_search(question, qdrant_client, collection_name, embedding_model, top_k=dense_top_k)
        bm25_results = bm25_search(question, qdrant_client, collection_name, bm25_model, top_k=bm25_top_k)
        fused_candidates = rrf_fuse(dense_results, bm25_results, k=RRF_K, final_top_k=RERANK_CANDIDATE_DEPTH)

        query_features = extract_query_features(question)
        retrieval_signals = extract_retrieval_signals(dense_results, bm25_results, fused_candidates[:10])

        # UNCONDITIONAL reranking — every question, regardless of any gate
        # decision, so Stage 2 gets complete training data for all 265.
        reranked_top5 = rerank_results(question, fused_candidates, reranker, top_k=TOP_K)
        rerank_signals = extract_rerank_signals(fused_candidates, reranked_top5)

        existing_records[label.qa_id] = {
            "qa_id": label.qa_id,
            "question_type": label.question_type,
            "hop_count": label.hop_count,
            "oracle_route": label.route,
            "oracle_hybrid_complete_evidence_at_5": label.hybrid_complete_evidence_at_5,
            "oracle_hybrid_reranker_complete_evidence_at_5": label.hybrid_reranker_complete_evidence_at_5,
            "query": question,
            "query_features": asdict(query_features),
            "retrieval_signals": asdict(retrieval_signals),
            "rerank_signals": asdict(rerank_signals),
        }

        if (i + 1) % 20 == 0 or (i + 1) == len(slice_labels):
            print(f"  [{i + 1}/{len(slice_labels)}] qa_id={label.qa_id} oracle_route={label.route}")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A.2 learned-router training dataset (query features + Hybrid retrieval "
                   "signals + UNCONDITIONAL reranker signals + oracle route label, EVALUATOR-only pairing)",
        "split": "development",
        "n_questions_total": len(oracle_labels),
        "n_questions_completed": len(existing_records),
        "partial_run": len(existing_records) < len(oracle_labels),
        "hybrid_config": {"dense_top_k": dense_top_k, "bm25_top_k": bm25_top_k,
                           "rerank_candidate_depth": RERANK_CANDIDATE_DEPTH},
        "records": list(existing_records.values()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 70}")
    print(f"Total accumulated: {len(existing_records)}/{len(oracle_labels)} questions")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
