#!/usr/bin/env python
"""Phase 8A.1 support script: extend Phase 8A's FROZEN direct classifier
(heuristic thresholds + GLM fallback) to all 265 non-null DEVELOPMENT
questions, so Phase 8A vs. Phase 8A.1 can be compared on the identical
question set ("for a stable comparison", per the Phase 8A.1 spec).

Phase 8A itself only ever evaluated its classifier on the 79-question
router_validation split (results/router_validation_report.json). This
script does NOT change Phase 8A's thresholds, prompt, or code — it reuses
them exactly as frozen — and does NOT overwrite that file. It:

  1. Reuses the 79 router_validation predictions ALREADY computed and
     persisted in results/router_validation_report.json (`per_question`)
     — no redundant GLM calls for those.
  2. Computes fresh predictions, LIVE (GLM only for cases Stage A is not
     confident about, same as Phase 8A), for the remaining ~186
     router_tune questions — using features already cached in
     results/router_dataset.json (no new Qdrant/embedding/BM25 calls
     needed; that dataset has no code path to final_holdout.json either).

Reads results/router_dataset.json, results/router_thresholds.json, and
results/router_validation_report.json — all Phase 8A outputs, READ-ONLY,
never modified. Writes a NEW file, results/router_full_dev_eval.json.

Usage:
    python scripts/run_phase8a_full_dev_eval.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.heuristic import HeuristicThresholds
from mhrag.routing.router import route_question

ROUTER_DATASET_PATH = "results/router_dataset.json"
ROUTER_THRESHOLDS_PATH = "results/router_thresholds.json"
ROUTER_VALIDATION_REPORT_PATH = "results/router_validation_report.json"  # READ-ONLY
OUTPUT_PATH = "results/router_full_dev_eval.json"


def main() -> None:
    dataset = json.loads((PROJECT_ROOT / ROUTER_DATASET_PATH).read_text())
    thresholds_artifact = json.loads((PROJECT_ROOT / ROUTER_THRESHOLDS_PATH).read_text())
    thresholds = HeuristicThresholds(**thresholds_artifact["thresholds"])
    validation_report = json.loads((PROJECT_ROOT / ROUTER_VALIDATION_REPORT_PATH).read_text())

    already_predicted = {q["qa_id"]: q for q in validation_report["per_question"]}
    print(f"Reusing {len(already_predicted)} already-computed router_validation predictions")

    records_by_id = {r["qa_id"]: r for r in dataset["records"]}
    remaining_ids = sorted(set(records_by_id) - set(already_predicted))
    print(f"Computing {len(remaining_ids)} new predictions for router_tune questions")

    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")
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
        raise SystemExit(f"Cannot run Phase 8A full-dev eval: {exc}") from exc
    glm_pricing = agent_config_yaml["pricing"]

    predictions = dict(already_predicted)
    for i, qa_id in enumerate(remaining_ids):
        record = records_by_id[qa_id]
        features = RouterFeatures(
            query=QueryFeatures(**record["query_features"]), retrieval=RetrievalSignals(**record["retrieval_signals"]),
        )
        result = route_question(
            record["query"], None, None, None, None, glm_client, thresholds,
            glm_input_price_per_million=glm_pricing["input_per_million_tokens"],
            glm_output_price_per_million=glm_pricing["output_per_million_tokens"],
            features=features,
        )
        predictions[qa_id] = {
            "qa_id": qa_id, "oracle_route": record["oracle_route"],
            "predicted_route": result.route, "stage_used": result.stage_used,
            "correct": result.route == record["oracle_route"],
        }
        if (i + 1) % 30 == 0 or (i + 1) == len(remaining_ids):
            print(f"  [{i + 1}/{len(remaining_ids)}] qa_id={qa_id} stage={result.stage_used} route={result.route}")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A direct classifier predictions extended to all 265 non-null DEVELOPMENT "
                   "questions (for Phase 8A vs Phase 8A.1 comparison) — reuses frozen Phase 8A "
                   "thresholds/prompt unchanged; treat as development evidence, not a fresh test",
        "n_questions": len(predictions),
        "n_reused_from_router_validation_report": len(already_predicted),
        "n_newly_computed": len(remaining_ids),
        "predictions": list(predictions.values()),
    }
    out_path = PROJECT_ROOT / OUTPUT_PATH
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
