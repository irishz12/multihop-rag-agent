"""Benchmark-split hardening tests: reproduction, disjointness, containment,
and leakage-guard behavior.

Uses a synthetic QARecord population (same style as test_sampling.py) sized
and proportioned like the real dataset, so split sizes (300/300/40) fit
comfortably and tests stay fast and offline.
"""

from __future__ import annotations

import pytest

from mhrag.data.benchmark import (
    BenchmarkIntegrityError,
    build_benchmark_splits,
    ensure_disjoint,
    ensure_subset,
    qa_id,
    question_type_distribution,
)
from mhrag.data.sampling import SubsetSpec
from mhrag.data.schema import QARecord

DEV_SEED = 42
FINAL_SEED = 123
SMOKE_SEED = 7


def _make_records(counts: dict[str, int]) -> list[QARecord]:
    records = []
    for qtype, n in counts.items():
        for i in range(n):
            records.append(
                QARecord(
                    query=f"{qtype}-query-{i}",
                    answer=f"{qtype}-answer-{i}",
                    question_type=qtype,
                    evidence_list=(),
                )
            )
    return records


# Mirrors real dataset proportions (comparison 33.5%, inference 31.9%,
# null 11.8%, temporal 22.8%) scaled up to comfortably fit 300 + 300 + 40.
POPULATION = _make_records(
    {
        "comparison_query": 850,
        "inference_query": 800,
        "null_query": 300,
        "temporal_query": 580,
    }
)  # total = 2530, close to the real population's 2556

DEV_SPEC = SubsetSpec(size=300, seed=DEV_SEED)
FINAL_SPEC = SubsetSpec(size=300, seed=FINAL_SEED)
SMOKE_SPEC = SubsetSpec(size=40, seed=SMOKE_SEED)


def _build():
    return build_benchmark_splits(POPULATION, DEV_SPEC, FINAL_SPEC, SMOKE_SPEC)


# --- deterministic reproduction -------------------------------------------------


def test_split_sizes_match_spec():
    splits = _build()
    assert len(splits.development) == 300
    assert len(splits.final_holdout) == 300
    assert len(splits.smoke) == 40


def test_deterministic_reproduction():
    a = _build()
    b = _build()
    assert [qa_id(r) for r in a.development] == [qa_id(r) for r in b.development]
    assert [qa_id(r) for r in a.final_holdout] == [qa_id(r) for r in b.final_holdout]
    assert [qa_id(r) for r in a.smoke] == [qa_id(r) for r in b.smoke]


def test_reproduction_is_stable_across_fresh_population_objects():
    """Rebuilding from an independently-constructed (but content-identical)
    population must reproduce the same splits — proves determinism doesn't
    depend on object identity or list ordering artifacts."""
    fresh_population = _make_records(
        {
            "comparison_query": 850,
            "inference_query": 800,
            "null_query": 300,
            "temporal_query": 580,
        }
    )
    a = build_benchmark_splits(POPULATION, DEV_SPEC, FINAL_SPEC, SMOKE_SPEC)
    b = build_benchmark_splits(fresh_population, DEV_SPEC, FINAL_SPEC, SMOKE_SPEC)
    assert {qa_id(r) for r in a.development} == {qa_id(r) for r in b.development}
    assert {qa_id(r) for r in a.final_holdout} == {qa_id(r) for r in b.final_holdout}


# --- disjointness / containment --------------------------------------------------


def test_zero_overlap_development_final_holdout():
    splits = _build()
    dev_ids = {qa_id(r) for r in splits.development}
    final_ids = {qa_id(r) for r in splits.final_holdout}
    assert dev_ids & final_ids == set()


def test_smoke_is_subset_of_development():
    splits = _build()
    dev_ids = {qa_id(r) for r in splits.development}
    smoke_ids = {qa_id(r) for r in splits.smoke}
    assert smoke_ids <= dev_ids
    assert len(smoke_ids) == 40


def test_final_holdout_disjoint_from_smoke_too():
    """smoke ⊆ development and development ∩ final_holdout = ∅ together
    imply smoke ∩ final_holdout = ∅ — assert it directly as a belt-and-braces
    check, since it's the property later phases actually rely on."""
    splits = _build()
    smoke_ids = {qa_id(r) for r in splits.smoke}
    final_ids = {qa_id(r) for r in splits.final_holdout}
    assert smoke_ids & final_ids == set()


def test_question_type_distributions_stratified():
    splits = _build()
    pop_dist = question_type_distribution(POPULATION)
    pop_total = len(POPULATION)
    for name, records in (
        ("development", splits.development),
        ("final_holdout", splits.final_holdout),
    ):
        dist = question_type_distribution(records)
        for qtype, pop_n in pop_dist.items():
            expected_share = pop_n / pop_total
            actual_share = dist.get(qtype, 0) / len(records)
            assert actual_share == pytest.approx(expected_share, abs=0.03), name


# --- leakage guards: final holdout cannot accidentally become development -------


def test_ensure_disjoint_raises_on_overlap():
    """Directly exercises the guard build_benchmark_splits relies on: given
    two id sets that *do* overlap (simulating an accidental merge of final
    holdout into development), it must raise, not silently pass."""
    a = {"id1", "id2", "id3"}
    b = {"id3", "id4"}
    with pytest.raises(BenchmarkIntegrityError, match="overlap"):
        ensure_disjoint(a, b, a_name="development", b_name="final_holdout")


def test_ensure_disjoint_passes_when_actually_disjoint():
    a = {"id1", "id2"}
    b = {"id3", "id4"}
    ensure_disjoint(a, b, a_name="development", b_name="final_holdout")  # no raise


def test_ensure_subset_raises_when_not_a_subset():
    smoke = {"id1", "id2", "idX"}  # idX not in development
    dev = {"id1", "id2", "id3"}
    with pytest.raises(BenchmarkIntegrityError, match="not a subset"):
        ensure_subset(smoke, dev, sub_name="smoke", sup_name="development")


def test_final_holdout_records_are_never_valid_development_ids():
    """The concrete accidental-reuse scenario: take a final_holdout record
    and check it against the development id set, the way a later-phase
    evaluation script would before treating a record as "safe to have been
    seen during development". It must never be found there."""
    splits = _build()
    dev_ids = {qa_id(r) for r in splits.development}
    for record in splits.final_holdout:
        assert qa_id(record) not in dev_ids


def test_development_seed_never_reproduces_final_holdout_content():
    """Sampling the full population with the development spec must always
    yield the development split, never final_holdout — i.e. an engineer who
    mistakenly re-derives "development" from the full population (instead of
    loading the persisted split) still cannot end up with holdout data,
    because the two are seeded from disjoint pools by construction."""
    from mhrag.data.sampling import stratified_sample

    rederived_dev = stratified_sample(POPULATION, DEV_SPEC)
    splits = _build()
    assert [qa_id(r) for r in rederived_dev] == [qa_id(r) for r in splits.development]
    final_ids = {qa_id(r) for r in splits.final_holdout}
    assert {qa_id(r) for r in rederived_dev} & final_ids == set()
