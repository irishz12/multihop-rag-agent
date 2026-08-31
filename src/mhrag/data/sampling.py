"""Deterministic, stratified dev/eval subset construction.

Given a fixed seed, `stratified_sample` always returns the same set of
records in the same order, regardless of machine or run — this is what
lets later phases (retrieval, agentic routing, cost tracking) be compared
against a stable evaluation slice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mhrag.data.schema import QARecord


@dataclass(frozen=True)
class SubsetSpec:
    size: int
    seed: int = 42


def allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` slots proportional to group sizes.

    Deterministic: ties in the remainder are broken by sorted group key, and
    no group is allocated more slots than it actually has.
    """
    population = sum(group_sizes.values())
    exact = {k: total * n / population for k, n in group_sizes.items()}
    floors = {k: min(int(v), group_sizes[k]) for k, v in exact.items()}
    remaining = total - sum(floors.values())

    remainders = sorted(
        group_sizes,
        key=lambda k: (-(exact[k] - int(exact[k])), k),
    )
    for key in remainders:
        if remaining <= 0:
            break
        if floors[key] < group_sizes[key]:
            floors[key] += 1
            remaining -= 1
    return floors


def stratified_sample(records: list[QARecord], spec: SubsetSpec) -> list[QARecord]:
    """Sample `spec.size` records, stratified proportionally by `question_type`.

    Group membership and per-group order are fixed by the input list order
    before any randomness is applied, so the only source of variation across
    runs is `spec.seed`.
    """
    if spec.size <= 0:
        raise ValueError(f"Subset size must be positive, got {spec.size}")
    if spec.size > len(records):
        raise ValueError(
            f"Requested subset size {spec.size} exceeds population {len(records)}"
        )

    groups: dict[str, list[QARecord]] = {}
    for rec in records:
        groups.setdefault(rec.question_type, []).append(rec)

    counts = allocate_counts({k: len(v) for k, v in groups.items()}, spec.size)

    rng = random.Random(spec.seed)
    sample: list[QARecord] = []
    for qtype in sorted(groups):
        sample.extend(rng.sample(groups[qtype], k=counts[qtype]))

    if len(sample) != spec.size:
        raise AssertionError(
            f"Stratified allocation produced {len(sample)} records, expected {spec.size}"
        )

    # Deterministic output order, independent of sampling order.
    sample.sort(key=lambda r: (r.question_type, r.query))
    return sample
