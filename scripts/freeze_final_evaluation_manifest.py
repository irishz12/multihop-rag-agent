#!/usr/bin/env python
"""FINAL HOLDOUT evaluation — pre-access configuration freeze.

MUST be run and its output committed BEFORE any script in this project
reads data/processed/final_holdout.json for the first time. This is the
one-time, no-more-tuning commitment point: every model id, retrieval
config, router threshold/weight, agent config, prompt template, and judge
config that the holdout evaluation will use is hashed HERE, while
final_holdout has still never been touched by anything in this codebase.
After this point, none of these frozen files may change before the
holdout evaluation completes — `scripts/analyze_phase9_holdout.py`
re-hashes the same file list at the end and the aggregation script raises
if a single hash differs, so a violation is caught mechanically, not just
by convention.

OFFLINE — no live call, no Mantle client, does not read
data/processed/final_holdout.json or data/processed/dev_subset.json
itself (only the CONFIG/CODE/ARTIFACT files that govern behavior).

Writes results/final_evaluation_manifest.json.

Usage:
    python scripts/freeze_final_evaluation_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config

# Every file whose content governs Agentic Multi-Hop RAG / Adaptive RAG pipeline behavior
# for the holdout evaluation. Config files (model ids, pricing, retrieval/agent/judge settings),
# the frozen Phase 8A.2 router artifact (weights + thresholds), every prompt template
# module, and the core pipeline/routing/eval source modules that execute them.
FROZEN_FILES = (
    "configs/dataset.yaml",
    "configs/retrieval.yaml",
    "configs/mantle.yaml",
    "configs/agent.yaml",
    "configs/judge.yaml",
    "configs/models.yaml",
    "results/learned_router_model.json",
    "src/mhrag/generation/prompts.py",
    "src/mhrag/generation/answer.py",
    "src/mhrag/generation/context.py",
    "src/mhrag/generation/cost.py",
    "src/mhrag/generation/mantle_client.py",
    "src/mhrag/agent/prompts.py",
    "src/mhrag/agent/controller.py",
    "src/mhrag/agent/evidence.py",
    "src/mhrag/agent/loop.py",
    "src/mhrag/adaptive/pipeline.py",
    "src/mhrag/routing/learned_router.py",
    "src/mhrag/routing/learned_features.py",
    "src/mhrag/routing/rerank_features.py",
    "src/mhrag/routing/features.py",
    "src/mhrag/eval/judge.py",
    "src/mhrag/eval/judge_prompts.py",
    "src/mhrag/eval/answer_metrics.py",
    "src/mhrag/eval/phase9_sample.py",
    "src/mhrag/retrieval/dense.py",
    "src/mhrag/retrieval/bm25.py",
    "src/mhrag/retrieval/rrf.py",
    "src/mhrag/retrieval/rerank.py",
)

OUTPUT_PATH = "results/final_evaluation_manifest.json"


def _sha1_of(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise SystemExit(f"cannot freeze manifest — expected file does not exist: {relative_path}")
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main() -> None:
    file_hashes = {rel: _sha1_of(rel) for rel in FROZEN_FILES}

    retrieval_config = load_config("configs/retrieval.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    agent_config_yaml = load_config("configs/agent.yaml")
    judge_config = load_config("configs/judge.yaml")
    router_model = json.loads((PROJECT_ROOT / "results" / "learned_router_model.json").read_text())

    frozen_values = {
        "embedding_model": retrieval_config["embedding"]["model_name"],
        "bm25_model": retrieval_config["bm25"]["model_name"],
        "reranker_model": retrieval_config["reranker"]["model_name"],
        "hybrid_dense_top_k": retrieval_config["hybrid"]["dense_top_k"],
        "hybrid_bm25_top_k": retrieval_config["hybrid"]["bm25_top_k"],
        "generation_model_id": mantle_config["generation"]["model_id"],
        "generation_prompt_version": mantle_config["generation"]["prompt_version"],
        "generation_temperature": mantle_config["generation"]["temperature"],
        "controller_model_id": agent_config_yaml["controller"]["model_id"],
        "controller_prompt_version": agent_config_yaml["controller"]["prompt_version"],
        "agent_max_hops": agent_config_yaml["loop"]["max_hops"],
        "agent_max_context_tokens": agent_config_yaml["loop"]["max_context_tokens"],
        "judge_model_id": judge_config["judge"]["model_id"],
        "judge_temperature": judge_config["judge"]["temperature"],
        "judge_prompt_version": judge_config["judge"]["prompt_version"],
        "router_stage1_threshold": router_model["stage1"]["threshold"],
        "router_stage2_threshold": router_model["stage2"]["threshold"],
        "router_stage1_feature_count": len(router_model["stage1"]["feature_names"]),
        "router_stage2_feature_count": len(router_model["stage2"]["feature_names"]),
    }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "FINAL HOLDOUT evaluation — pre-access configuration freeze manifest. "
                   "Computed and persisted BEFORE any final_holdout.json access. "
                   "No tuning is permitted after this manifest exists.",
        "final_holdout_access_status": "NOT_YET_ACCESSED",
        "file_hashes_sha1": file_hashes,
        "frozen_values": frozen_values,
    }
    out_path = PROJECT_ROOT / OUTPUT_PATH
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Froze {len(file_hashes)} file(s) + {len(frozen_values)} value(s) into {out_path}")
    print("final_holdout_access_status = NOT_YET_ACCESSED")


if __name__ == "__main__":
    main()
