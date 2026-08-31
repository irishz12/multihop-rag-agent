#!/usr/bin/env python
"""Phase 9 (reduced scope): select the fixed 50-question DEVELOPMENT
benchmark sample and report how much of it can be reused from already-
completed pipeline checkpoints. OFFLINE — no live calls, no Mantle client
constructed, nothing here can incur cost.

Selection: `mhrag.eval.phase9_sample.select_phase9_sample` — two-level
stratified (question_type, then hop_count where applicable), frozen seed
`PHASE9_SAMPLE_SEED`, deterministic.

Reads ONLY data/processed/dev_subset.json (never final_holdout.json — no
CLI flag, no config option exists to select a different split) plus
whatever `results/phase9_{pipeline}_raw.json` checkpoints already exist
(read-only; this script never writes to them).

Writes results/phase9_sample.json: selected qa_ids, seed, distribution,
question_type rollup, and dataset_hash (SHA-1 of the exact dev_subset.json
bytes used) — the provenance record for the reduced-scope sample.

Usage:
    python scripts/select_phase9_sample.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.legacy_pipeline_names import to_legacy_name
from mhrag.eval.phase9_sample import PHASE9_SAMPLE_SEED, PHASE9_SAMPLE_SIZE, dataset_hash, select_phase9_sample

DEV_SPLIT_FILE = "dev_subset.json"
SAMPLE_OUTPUT_PATH = "results/phase9_sample.json"
PIPELINES = ("dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag")


def main() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    records = load_qa_records(dev_path)
    print(f"Loaded {len(records)} DEVELOPMENT records (all question types) from {dev_path}")

    sample_meta, selected = select_phase9_sample(records, size=PHASE9_SAMPLE_SIZE, seed=PHASE9_SAMPLE_SEED)
    selected_qa_ids = [qa_id(r) for r in selected]

    print(f"\nSelected {len(selected_qa_ids)} questions (seed={PHASE9_SAMPLE_SEED})")
    print(f"By question_type: {sample_meta.question_type_distribution}")
    print("By stratum (question_type::hop):")
    for key in sorted(sample_meta.distribution):
        print(f"  {key}: {sample_meta.distribution[key]}")

    # --- reuse check against existing pipeline checkpoints (read-only) ---
    reuse_report: dict[str, dict] = {}
    for pipeline in PIPELINES:
        raw_path = PROJECT_ROOT / "results" / f"phase9_{to_legacy_name(pipeline)}_raw.json"
        if not raw_path.exists():
            reuse_report[pipeline] = {
                "checkpoint_exists": False, "n_completed_total": 0,
                "n_reusable_for_sample": 0, "n_still_needed": len(selected_qa_ids),
                "missing_qa_ids": list(selected_qa_ids),
            }
            continue
        existing = json.loads(raw_path.read_text())
        existing_ids = {r["qa_id"] for r in existing["records"]}
        reusable = [q for q in selected_qa_ids if q in existing_ids]
        missing = [q for q in selected_qa_ids if q not in existing_ids]
        reuse_report[pipeline] = {
            "checkpoint_exists": True,
            "n_completed_total": len(existing_ids),
            "n_reusable_for_sample": len(reusable),
            "n_still_needed": len(missing),
            "missing_qa_ids": missing,
        }

    print(f"\n{'=' * 70}\nReuse against existing checkpoints (of {len(selected_qa_ids)} selected):")
    total_still_needed = 0
    for pipeline in PIPELINES:
        r = reuse_report[pipeline]
        print(f"  {pipeline:18s} completed_total={r['n_completed_total']:>3}  "
              f"reusable={r['n_reusable_for_sample']:>3}  still_needed={r['n_still_needed']:>3}")
        total_still_needed += r["n_still_needed"]
    print(f"\nTotal NEW live pipeline runs still needed across all 5 pipelines: {total_still_needed} "
          f"(of a possible {5 * len(selected_qa_ids)})")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 9 (reduced scope) — fixed 50-question stratified DEVELOPMENT benchmark sample "
                   "+ reuse report against existing pipeline checkpoints",
        "split": "development",
        "seed": PHASE9_SAMPLE_SEED,
        "size": PHASE9_SAMPLE_SIZE,
        "dataset_hash_sha1": dataset_hash(dev_path),
        "dataset_path": str(dev_path.relative_to(PROJECT_ROOT)),
        "qa_ids": selected_qa_ids,
        "question_type_distribution": sample_meta.question_type_distribution,
        "stratum_distribution": sample_meta.distribution,
        "reuse_report": reuse_report,
        "total_new_live_runs_still_needed": total_still_needed,
    }
    out_path = PROJECT_ROOT / SAMPLE_OUTPUT_PATH
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
