"""Hardened benchmark split construction: development / final-holdout / smoke.

Builds three evaluation slices from the MultiHop-RAG QA population, all
stratified by `question_type` and seeded deterministically (see
configs/dataset.yaml: `subset` for development, `benchmark` for the rest):

- development: the Phase 1 300-question stratified subset (unchanged).
- final_holdout: 300 questions sampled from the population MINUS
  development — disjoint from development by construction.
- smoke: ~40 questions sampled from development ONLY — a subset of
  development by construction, for cheap iteration.

This module builds on `mhrag.data.sampling.stratified_sample` without
modifying it — Phase 1's sampling code and tests are untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from mhrag.data.sampling import SubsetSpec, stratified_sample
from mhrag.data.schema import QARecord

# Real unit-separator control char — cheap, collision-resistant field join
# for hashing (won't appear in natural-language query/answer text).
_FIELD_SEP = "\x1f"


class BenchmarkIntegrityError(ValueError):
    """Raised when a benchmark split violates a required invariant.

    Kept distinct from ValueError-subclasses used elsewhere (e.g.
    SchemaValidationError) so callers can catch benchmark-integrity
    failures specifically.
    """


def qa_id(record: QARecord) -> str:
    """Stable content-derived id for a QA record.

    The dataset has no native id field. The id is a hash of
    (question_type, query, answer) rather than query alone, so a
    hypothetical duplicate query paired with a different answer would not
    collide. Deterministic across runs and machines.
    """
    basis = _FIELD_SEP.join((record.question_type, record.query, record.answer))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def question_type_distribution(records: list[QARecord]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in records:
        dist[r.question_type] = dist.get(r.question_type, 0) + 1
    return dist


def ensure_disjoint(a: set[str], b: set[str], *, a_name: str, b_name: str) -> None:
    overlap = a & b
    if overlap:
        preview = sorted(overlap)[:5]
        raise BenchmarkIntegrityError(
            f"{a_name} and {b_name} overlap on {len(overlap)} record(s), e.g. {preview}"
        )


def ensure_subset(sub: set[str], sup: set[str], *, sub_name: str, sup_name: str) -> None:
    if not sub <= sup:
        preview = sorted(sub - sup)[:5]
        raise BenchmarkIntegrityError(
            f"{sub_name} is not a subset of {sup_name}; e.g. {preview} not in {sup_name}"
        )


@dataclass(frozen=True)
class BenchmarkSplits:
    development: list[QARecord]
    final_holdout: list[QARecord]
    smoke: list[QARecord]


def build_benchmark_splits(
    population: list[QARecord],
    dev_spec: SubsetSpec,
    final_spec: SubsetSpec,
    smoke_spec: SubsetSpec,
) -> BenchmarkSplits:
    """Build development / final-holdout / smoke splits with invariants enforced.

    Raises BenchmarkIntegrityError if, despite the disjoint-by-construction
    design, an invariant is ever violated (e.g. an id collision in
    `population` masking two distinct records as the same record).
    """
    development = stratified_sample(population, dev_spec)
    dev_ids = {qa_id(r) for r in development}
    if len(dev_ids) != len(development):
        raise BenchmarkIntegrityError("qa_id collision within development split")

    remaining = [r for r in population if qa_id(r) not in dev_ids]
    if len(remaining) != len(population) - len(development):
        raise BenchmarkIntegrityError(
            "qa_id collision detected while excluding development records from population"
        )

    final_holdout = stratified_sample(remaining, final_spec)
    final_ids = {qa_id(r) for r in final_holdout}
    if len(final_ids) != len(final_holdout):
        raise BenchmarkIntegrityError("qa_id collision within final_holdout split")
    ensure_disjoint(dev_ids, final_ids, a_name="development", b_name="final_holdout")

    smoke = stratified_sample(development, smoke_spec)
    smoke_ids = {qa_id(r) for r in smoke}
    ensure_subset(smoke_ids, dev_ids, sub_name="smoke", sup_name="development")

    return BenchmarkSplits(development=development, final_holdout=final_holdout, smoke=smoke)


def _split_summary(records: list[QARecord]) -> dict:
    return {
        "size": len(records),
        "question_type_distribution": question_type_distribution(records),
        "qa_ids": sorted(qa_id(r) for r in records),
    }


def build_manifest(
    *,
    dataset_source: dict[str, str],
    population: list[QARecord],
    splits: BenchmarkSplits,
    seeds: dict[str, int],
    raw_file_sha256: dict[str, str],
) -> dict:
    """Build a JSON-serializable manifest documenting the benchmark splits.

    Contains: dataset source, dataset size, seeds, split sizes, per-split
    question_type distributions, and stable qa_id lists per split (used both
    for provenance and as the ground truth in overlap/leakage tests).
    """
    return {
        "dataset_source": dataset_source,
        "dataset_size": {"qa_records_total": len(population)},
        "raw_file_sha256": raw_file_sha256,
        "seeds": seeds,
        "splits": {
            "development": _split_summary(splits.development),
            "final_holdout": _split_summary(splits.final_holdout),
            "smoke": _split_summary(splits.smoke),
        },
    }
