#!/usr/bin/env python
"""Phase 8A: build the deterministic router_tune / router_validation split.

Offline — reads only results/router_dataset.json (already built by
scripts/build_router_dataset.py from DEVELOPMENT-only data), makes no live
calls. `mhrag.routing.split.split_tune_validation` stratifies by oracle
route label, ~70/30, deterministic (fixed seed).

Usage:
    python scripts/build_router_split.py

Writes results/router_split.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT
from mhrag.routing.oracle import OracleLabel, label_distribution
from mhrag.routing.split import TUNE_FRACTION, TUNE_VALIDATION_SEED, split_tune_validation

ROUTER_DATASET_PATH = "results/router_dataset.json"


def main() -> None:
    dataset_path = PROJECT_ROOT / ROUTER_DATASET_PATH
    dataset = json.loads(dataset_path.read_text())

    labels = [
        OracleLabel(
            qa_id=r["qa_id"], question_type=r["question_type"], hop_count=r["hop_count"],
            route=r["oracle_route"],
            hybrid_complete_evidence_at_5=r["oracle_hybrid_complete_evidence_at_5"],
            hybrid_reranker_complete_evidence_at_5=r["oracle_hybrid_reranker_complete_evidence_at_5"],
        )
        for r in dataset["records"]
    ]
    print(f"Loaded {len(labels)} oracle labels from {dataset_path}")
    print(f"Distribution: {label_distribution(labels)}")

    tune_ids, validation_ids = split_tune_validation(labels)
    print(f"router_tune: {len(tune_ids)} ({len(tune_ids) / len(labels):.1%})")
    print(f"router_validation: {len(validation_ids)} ({len(validation_ids) / len(labels):.1%})")

    tune_dist = label_distribution([label for label in labels if label.qa_id in set(tune_ids)])
    validation_dist = label_distribution([label for label in labels if label.qa_id in set(validation_ids)])
    print(f"router_tune distribution: {tune_dist}")
    print(f"router_validation distribution: {validation_dist}")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A router_tune / router_validation split (stratified by oracle route label)",
        "seed": TUNE_VALIDATION_SEED,
        "tune_fraction": TUNE_FRACTION,
        "router_tune_qa_ids": tune_ids,
        "router_validation_qa_ids": validation_ids,
        "router_tune_size": len(tune_ids),
        "router_validation_size": len(validation_ids),
        "router_tune_distribution": tune_dist,
        "router_validation_distribution": validation_dist,
    }
    out_path = PROJECT_ROOT / "results" / "router_split.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
