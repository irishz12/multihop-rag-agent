#!/usr/bin/env python
"""Phase 8A: fit Stage A's frozen `HeuristicThresholds` on router_tune ONLY.

Offline — reads results/router_dataset.json + results/router_split.json,
makes no live calls. Uses `mhrag.routing.tune_thresholds.fit_thresholds`
(evaluator-only module — see that module's docstring). Reports tune-set
accuracy/coverage for visibility, but the thresholds are what gets frozen
and persisted; router performance is reported separately, on
router_validation, by scripts/run_router_validation.py.

Usage:
    python scripts/tune_router_thresholds.py

Writes results/router_thresholds.json.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.heuristic import classify_heuristic
from mhrag.routing.tune_thresholds import TuneExample, fit_thresholds

ROUTER_DATASET_PATH = "results/router_dataset.json"
ROUTER_SPLIT_PATH = "results/router_split.json"


def _tune_example(record: dict) -> TuneExample:
    qf = record["query_features"]
    rs = record["retrieval_signals"]
    features = RouterFeatures(query=QueryFeatures(**qf), retrieval=RetrievalSignals(**rs))
    return TuneExample(qa_id=record["qa_id"], features=features, oracle_route=record["oracle_route"])


def main() -> None:
    dataset = json.loads((PROJECT_ROOT / ROUTER_DATASET_PATH).read_text())
    split = json.loads((PROJECT_ROOT / ROUTER_SPLIT_PATH).read_text())
    tune_ids = set(split["router_tune_qa_ids"])

    records_by_id = {r["qa_id"]: r for r in dataset["records"]}
    tune_examples = [_tune_example(records_by_id[qa_id]) for qa_id in sorted(tune_ids)]
    print(f"Fitting thresholds on {len(tune_examples)} router_tune examples")

    thresholds = fit_thresholds(tune_examples)
    print(f"Frozen thresholds: {thresholds}")

    correct = 0
    confident = 0
    for ex in tune_examples:
        verdict = classify_heuristic(ex.features, thresholds)
        if verdict.confident:
            confident += 1
            if verdict.route == ex.oracle_route:
                correct += 1
    coverage = confident / len(tune_examples)
    tune_accuracy = correct / confident if confident else 0.0
    print(f"Tune-set heuristic coverage: {coverage:.1%} ({confident}/{len(tune_examples)})")
    print(f"Tune-set heuristic accuracy (among confident calls): {tune_accuracy:.1%} ({correct}/{confident})")

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 8A frozen Stage A heuristic thresholds (fit on router_tune only)",
        "thresholds": dataclasses.asdict(thresholds),
        "tune_set_size": len(tune_examples),
        "tune_set_heuristic_coverage": coverage,
        "tune_set_heuristic_accuracy_among_confident": tune_accuracy,
        "provenance": {
            "router_dataset_file": ROUTER_DATASET_PATH,
            "router_split_file": ROUTER_SPLIT_PATH,
            "fit_method": "mhrag.routing.tune_thresholds.fit_thresholds (grid search over tune-set quantiles)",
        },
    }
    out_path = PROJECT_ROOT / "results" / "router_thresholds.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
