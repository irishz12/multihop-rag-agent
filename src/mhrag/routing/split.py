"""EVALUATOR-ONLY: deterministic router_tune / router_validation split.

Splits `OracleLabel`s (already DEVELOPMENT-only, already null_query-
excluded — see `mhrag.routing.oracle`) into two DISJOINT groups,
stratified by oracle route label, ~70/30. `router_tune` is used only by
`mhrag.routing.tune_thresholds.fit_thresholds`; router performance is
reported on `router_validation`, which threshold-fitting never sees.

Both groups are drawn from `dev_subset.json` — which Phase 1.1's benchmark
manifest already establishes is disjoint from `final_holdout.json` by
construction (`mhrag.data.benchmark.build_benchmark_splits`) — so this
split's qa_ids are structurally guaranteed to stay outside final_holdout
too; tests/test_routing_split.py additionally checks this directly against
the real final_holdout.json qa_ids as a second, independent guard.

Allocation logic intentionally mirrors `mhrag.data.sampling._allocate_
counts`'s largest-remainder approach (kept self-contained here rather than
importing that private helper, since this module's group key is the
oracle route label, not question_type, and this package should not modify
or reach into `mhrag.data.sampling`).
"""

from __future__ import annotations

import random

from mhrag.routing.oracle import OracleLabel

TUNE_VALIDATION_SEED = 4242
TUNE_FRACTION = 0.7


def _allocate_tune_counts(group_sizes: dict[str, int], fraction: float) -> dict[str, int]:
    exact = {k: n * fraction for k, n in group_sizes.items()}
    floors = {k: min(int(v), group_sizes[k]) for k, v in exact.items()}
    total_target = round(sum(group_sizes.values()) * fraction)
    remaining = total_target - sum(floors.values())

    remainders = sorted(group_sizes, key=lambda k: (-(exact[k] - int(exact[k])), k))
    for key in remainders:
        if remaining <= 0:
            break
        if floors[key] < group_sizes[key]:
            floors[key] += 1
            remaining -= 1
    return floors


def split_tune_validation(
    labels: list[OracleLabel],
    tune_fraction: float = TUNE_FRACTION,
    seed: int = TUNE_VALIDATION_SEED,
) -> tuple[list[str], list[str]]:
    """Returns (router_tune_qa_ids, router_validation_qa_ids) — disjoint,
    covering every qa_id in `labels` exactly once, stratified by
    `label.route` as evenly as the ~70/30 target allows. Deterministic:
    same `labels` + `seed` always produces the same split."""
    groups: dict[str, list[OracleLabel]] = {}
    for label in labels:
        groups.setdefault(label.route, []).append(label)

    tune_counts = _allocate_tune_counts({k: len(v) for k, v in groups.items()}, tune_fraction)

    rng = random.Random(seed)
    tune_ids: list[str] = []
    validation_ids: list[str] = []
    for route in sorted(groups):
        pool = sorted(groups[route], key=lambda label: label.qa_id)
        tune_sample = set(rng.sample([label.qa_id for label in pool], k=tune_counts[route]))
        for label in pool:
            (tune_ids if label.qa_id in tune_sample else validation_ids).append(label.qa_id)

    tune_ids.sort()
    validation_ids.sort()
    return tune_ids, validation_ids
