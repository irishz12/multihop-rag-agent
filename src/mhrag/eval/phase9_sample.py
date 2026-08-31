"""EVALUATOR-ONLY: deterministic selection of the Phase 9 REDUCED
50-question DEVELOPMENT benchmark sample.

Two-level stratification, reusing `mhrag.data.sampling.allocate_counts`
(the SAME proportional largest-remainder allocation `stratified_sample`
itself uses to build `dev_subset.json`, unmodified) rather than a new
allocation rule:

  1. question_type (`inference_query` / `comparison_query` /
     `temporal_query` / `null_query`).
  2. WITHIN each non-null question_type, hop_count bucket (2/3/4 — however
     many a given type actually has; `comparison_query`/`temporal_query`
     top out at 3-hop, `inference_query` reaches 4-hop, matching
     MultiHop-RAG's real question construction). `null_query` has no
     hop_count to sub-stratify (its evidence_list is always empty,
     hop_count always 0) — it is kept as ONE single stratum, "hop-count
     stratification where applicable" per the Phase 9 spec.

Each (question_type[, hop_count]) combination is one stratum; `size` slots
are allocated across ALL strata proportional to population share (largest-
remainder, deterministic ties broken by stratum key), then `seed` drives
`random.Random.sample` within each stratum — same "fixed seed -> same
sample everywhere" contract as `mhrag.data.sampling.stratified_sample`.

Deterministic output order: stratum key, then query text — independent of
population file order or dict iteration order.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from mhrag.data.sampling import allocate_counts
from mhrag.data.schema import QARecord
from mhrag.eval.ground_truth import hop_count

PHASE9_SAMPLE_SEED = 2029  # frozen — changing this changes the whole sample; not re-derived per run
PHASE9_SAMPLE_SIZE = 50


def stratum_key(record: QARecord) -> str:
    """`"null_query"` for null_query (no hop sub-stratification — its
    hop_count is always 0 by construction); `"{question_type}::hop{N}"`
    for every other question."""
    if record.question_type == "null_query":
        return "null_query"
    return f"{record.question_type}::hop{hop_count(record)}"


@dataclass(frozen=True, slots=True)
class Phase9Sample:
    qa_ids: tuple[str, ...]
    distribution: dict[str, int]  # stratum key -> count actually selected
    question_type_distribution: dict[str, int]  # question_type -> count actually selected (rollup)
    seed: int
    size: int


def select_phase9_sample(
    records: list[QARecord], size: int = PHASE9_SAMPLE_SIZE, seed: int = PHASE9_SAMPLE_SEED
) -> tuple[Phase9Sample, list[QARecord]]:
    """Returns (`Phase9Sample` metadata, the actual selected `QARecord`
    list in deterministic output order) — the caller derives qa_ids from
    the records itself (this module does not import `mhrag.data.benchmark`
    to keep a minimal, focused dependency set)."""
    if size > len(records):
        raise ValueError(f"requested sample size {size} exceeds population {len(records)}")

    groups: dict[str, list[QARecord]] = {}
    for r in records:
        groups.setdefault(stratum_key(r), []).append(r)

    counts = allocate_counts({k: len(v) for k, v in groups.items()}, size)

    rng = random.Random(seed)
    selected: list[QARecord] = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], k=counts[key]))

    selected.sort(key=lambda r: (r.question_type, hop_count(r), r.query))

    if len(selected) != size:
        raise AssertionError(f"stratified allocation produced {len(selected)} records, expected {size}")

    question_type_rollup: dict[str, int] = {}
    for r in selected:
        question_type_rollup[r.question_type] = question_type_rollup.get(r.question_type, 0) + 1

    return (
        Phase9Sample(
            qa_ids=(),  # filled in by the caller once it computes qa_id(record) for each
            distribution=dict(counts),
            question_type_distribution=question_type_rollup,
            seed=seed,
            size=size,
        ),
        selected,
    )


def dataset_hash(dev_subset_path) -> str:
    """SHA-1 of the exact dev_subset.json file bytes used for selection —
    provenance: proves which exact file this sample was drawn from."""
    from pathlib import Path

    return hashlib.sha1(Path(dev_subset_path).read_bytes()).hexdigest()
