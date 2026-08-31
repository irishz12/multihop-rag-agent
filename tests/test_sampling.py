"""Deterministic stratified sampling tests.

Uses synthetic QARecord populations (not the real dataset) so group sizes
and edge cases are controlled exactly — the sampling algorithm itself is
what's under test here, independent of dataset content.
"""

from __future__ import annotations

import pytest

from mhrag.data.sampling import SubsetSpec, stratified_sample
from mhrag.data.schema import QARecord


def _make_records(counts: dict[str, int]) -> list[QARecord]:
    records = []
    for qtype, n in counts.items():
        for i in range(n):
            records.append(
                QARecord(
                    query=f"{qtype}-query-{i}",
                    answer=f"answer-{i}",
                    question_type=qtype,
                    evidence_list=(),
                )
            )
    return records


POPULATION = _make_records(
    {
        "comparison_query": 340,
        "inference_query": 320,
        "null_query": 120,
        "temporal_query": 230,
    }
)  # mirrors real dataset proportions at 1/2.5 scale, total = 1010


def test_same_seed_is_deterministic():
    spec = SubsetSpec(size=100, seed=42)
    a = stratified_sample(POPULATION, spec)
    b = stratified_sample(POPULATION, spec)
    assert [r.query for r in a] == [r.query for r in b]


def test_different_seed_can_differ():
    a = stratified_sample(POPULATION, SubsetSpec(size=100, seed=42))
    b = stratified_sample(POPULATION, SubsetSpec(size=100, seed=7))
    assert [r.query for r in a] != [r.query for r in b]


def test_subset_size_is_exact():
    for size in (10, 100, 500, len(POPULATION)):
        subset = stratified_sample(POPULATION, SubsetSpec(size=size, seed=42))
        assert len(subset) == size


def test_subset_has_no_duplicates():
    subset = stratified_sample(POPULATION, SubsetSpec(size=300, seed=42))
    assert len(set(r.query for r in subset)) == len(subset)


def test_stratification_is_proportional_within_rounding():
    subset = stratified_sample(POPULATION, SubsetSpec(size=200, seed=42))
    pop_total = len(POPULATION)
    pop_counts: dict[str, int] = {}
    sub_counts: dict[str, int] = {}
    for r in POPULATION:
        pop_counts[r.question_type] = pop_counts.get(r.question_type, 0) + 1
    for r in subset:
        sub_counts[r.question_type] = sub_counts.get(r.question_type, 0) + 1

    for qtype, pop_n in pop_counts.items():
        expected_share = pop_n / pop_total
        actual_share = sub_counts.get(qtype, 0) / len(subset)
        assert actual_share == pytest.approx(expected_share, abs=0.02)


def test_all_question_types_represented_when_subset_large_enough():
    subset = stratified_sample(POPULATION, SubsetSpec(size=200, seed=42))
    assert {r.question_type for r in subset} == {
        "comparison_query",
        "inference_query",
        "null_query",
        "temporal_query",
    }


def test_oversized_subset_raises():
    with pytest.raises(ValueError, match="exceeds population"):
        stratified_sample(POPULATION, SubsetSpec(size=len(POPULATION) + 1, seed=42))


def test_zero_or_negative_size_raises():
    with pytest.raises(ValueError, match="positive"):
        stratified_sample(POPULATION, SubsetSpec(size=0, seed=42))


def test_full_population_size_returns_everything():
    subset = stratified_sample(POPULATION, SubsetSpec(size=len(POPULATION), seed=42))
    assert len(subset) == len(POPULATION)
    assert {r.query for r in subset} == {r.query for r in POPULATION}


def test_small_group_not_starved_by_rounding():
    """At a subset size small enough that naive floor-rounding would drop
    the smallest group's exact share below 1, the largest-remainder
    allocation must still give it a slot."""
    subset = stratified_sample(POPULATION, SubsetSpec(size=10, seed=42))
    assert len(subset) == 10
    sub_types = {r.question_type for r in subset}
    assert "null_query" in sub_types
