#!/usr/bin/env python
"""FINAL HOLDOUT evaluation — deterministic 50-question sample selection.

THE ONE SCRIPT IN THIS ENTIRE CODEBASE THAT READS
data/processed/final_holdout.json. Every other script in this project
(dev/smoke evaluation, router training, Phase 8B/9 development-sample
work) is hardcoded to dev_subset.json/smoke_subset.json with no CLI flag
or config option that could reach final_holdout — that guarantee holds
unbroken; this script is a deliberate, clearly-named, one-time exception,
run only once final_evaluation_manifest.json already exists (see
scripts/freeze_final_evaluation_manifest.py, which MUST run first — this
script does not enforce that ordering in code, but the whole project's
audit trail depends on it having already happened).

Same selection method as Phase 9's development sample
(`mhrag.eval.phase9_sample.select_phase9_sample` — two-level stratified,
question_type then hop_count where applicable, unmodified), the SAME
frozen seed (`PHASE9_SAMPLE_SEED`), applied to a DIFFERENT population
(final_holdout instead of dev_subset) — deliberately reusing the exact
same method and seed constant rather than choosing a new one, so there is
no appearance of seed-shopping for a particular holdout sample.

OFFLINE — no live call, no Mantle client. Reads ONLY
data/processed/final_holdout.json (this file's one deliberate exception)
and writes ONLY results/phase9_holdout_sample.json.

Usage:
    python scripts/select_phase9_holdout_sample.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.phase9_sample import PHASE9_SAMPLE_SEED, PHASE9_SAMPLE_SIZE, dataset_hash, select_phase9_sample

HOLDOUT_SPLIT_FILE = "final_holdout.json"
SAMPLE_OUTPUT_PATH = "results/phase9_holdout_sample.json"
MANIFEST_PATH = "results/final_evaluation_manifest.json"


def main() -> None:
    manifest_path = PROJECT_ROOT / MANIFEST_PATH
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} does not exist — run scripts/freeze_final_evaluation_manifest.py "
            "BEFORE selecting the holdout sample (the whole point is freezing config before any "
            "final_holdout access)"
        )
    manifest = json.loads(manifest_path.read_text())
    print(f"Pre-access manifest found (status={manifest['final_holdout_access_status']}, "
          f"generated_at={manifest['generated_at']})")

    dataset_config = load_config("configs/dataset.yaml")
    holdout_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / HOLDOUT_SPLIT_FILE
    records = load_qa_records(holdout_path)
    print(f"Loaded {len(records)} FINAL HOLDOUT records (all question types) from {holdout_path}")

    sample_meta, selected = select_phase9_sample(records, size=PHASE9_SAMPLE_SIZE, seed=PHASE9_SAMPLE_SEED)
    selected_qa_ids = [qa_id(r) for r in selected]

    print(f"\nSelected {len(selected_qa_ids)} questions (seed={PHASE9_SAMPLE_SEED})")
    print(f"By question_type: {sample_meta.question_type_distribution}")
    print("By stratum (question_type::hop):")
    for key in sorted(sample_meta.distribution):
        print(f"  {key}: {sample_meta.distribution[key]}")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "FINAL HOLDOUT evaluation — fixed 50-question stratified sample selection",
        "split": "final_holdout",
        "seed": PHASE9_SAMPLE_SEED,
        "size": PHASE9_SAMPLE_SIZE,
        "dataset_hash_sha1": dataset_hash(holdout_path),
        "dataset_path": str(holdout_path.relative_to(PROJECT_ROOT)),
        "pre_access_manifest_generated_at": manifest["generated_at"],
        "qa_ids": selected_qa_ids,
        "question_type_distribution": sample_meta.question_type_distribution,
        "stratum_distribution": sample_meta.distribution,
        # Persisted here (rather than re-read from final_holdout.json downstream) so the
        # offline aggregation script never needs a second final_holdout access — this
        # script stays the ONE place in the codebase that opens that file.
        "gold_doc_ids_by_qa_id": {qid: sorted(gold_doc_ids(r)) for qid, r in zip(selected_qa_ids, selected)},
    }
    out_path = PROJECT_ROOT / SAMPLE_OUTPUT_PATH
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
