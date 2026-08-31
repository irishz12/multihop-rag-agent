#!/usr/bin/env python
"""Phase 8A: build the router feature dataset — LIVE Qdrant + embedding +
BM25 retrieval (the frozen Hybrid RRF baseline), NO Mantle calls, so this
script has zero LLM cost.

For every non-null DEVELOPMENT question, computes `RouterFeatures`
(`mhrag.routing.features.compute_router_features` — query text features +
cheap initial Hybrid retrieval signals) and pairs it with that question's
oracle route label (`mhrag.routing.oracle`, computed from the already-
frozen `results/retrieval_eval_development.json`, not from a new
retrieval run). The oracle label is included in this dataset file for
EVALUATOR use only (threshold tuning, validation scoring) — writing it to
disk alongside the features does not mean the runtime router ever sees it;
`mhrag.routing.router.route_question` never reads this file.

Reads ONLY data/processed/dev_subset.json — DEV_SPLIT_FILE is a hardcoded
module constant, no CLI flag, no config option, so there is no code path
in this script that can reach final_holdout.json. Also reads
results/retrieval_eval_development.json (already dev-only, already
null_query-excluded).

Usage:
    python scripts/build_router_dataset.py

Writes results/router_dataset.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.routing.features import compute_router_features
from mhrag.routing.oracle import compute_oracle_labels

# Hardcoded to the development split ONLY — no CLI flag, no config option,
# so there is no code path in this script that can reach final_holdout.json.
DEV_SPLIT_FILE = "dev_subset.json"
RETRIEVAL_EVAL_ARTIFACT = "results/retrieval_eval_development.json"


def main() -> None:
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

    qdrant_client = get_client(retrieval_config["qdrant"]["url"])
    collection_name = retrieval_config["qdrant"]["collection_name"]
    dense_top_k = retrieval_config["hybrid"]["dense_top_k"]
    bm25_top_k = retrieval_config["hybrid"]["bm25_top_k"]

    records_out = []
    for i, label in enumerate(oracle_labels):
        record = dev_by_qa_id[label.qa_id]
        features = compute_router_features(
            record.query, qdrant_client, collection_name, embedding_model, bm25_model,
            dense_top_k=dense_top_k, bm25_top_k=bm25_top_k,
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(oracle_labels):
            print(f"  [{i + 1}/{len(oracle_labels)}] {record.query[:80]!r}")

        records_out.append(
            {
                "qa_id": label.qa_id,
                "question_type": label.question_type,
                "hop_count": label.hop_count,
                "oracle_route": label.route,
                "oracle_hybrid_complete_evidence_at_5": label.hybrid_complete_evidence_at_5,
                "oracle_hybrid_reranker_complete_evidence_at_5": label.hybrid_reranker_complete_evidence_at_5,
                "query": record.query,
                "query_features": asdict(features.query),
                "retrieval_signals": asdict(features.retrieval),
            }
        )

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A router feature dataset (query features + cheap Hybrid retrieval signals + "
                   "oracle route label, EVALUATOR-only pairing)",
        "split": "development",
        "n_questions": len(records_out),
        "hybrid_config": {"dense_top_k": dense_top_k, "bm25_top_k": bm25_top_k},
        "records": records_out,
    }
    out_path = PROJECT_ROOT / "results" / "router_dataset.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
