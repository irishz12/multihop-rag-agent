"""Phase 9 reduced-sample stratified selection tests — offline, no live
call, no dependency on the real dev_subset.json (synthetic populations
built in-line so the tests are self-contained and fast)."""

from __future__ import annotations

from mhrag.data.schema import Evidence, QARecord
from mhrag.eval.phase9_sample import (
    PHASE9_SAMPLE_SEED,
    PHASE9_SAMPLE_SIZE,
    select_phase9_sample,
    stratum_key,
)


def _evidence(n: int) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(
            title=f"t{i}", author=None, url=f"https://example.com/doc{i}", source="s", category="c",
            published_at="2024-01-01T00:00:00+00:00", fact=f"fact{i}",
        )
        for i in range(n)
    )


def _record(question_type: str, n_gold_docs: int, i: int) -> QARecord:
    return QARecord(
        query=f"{question_type} question {i}", answer="Insufficient information." if n_gold_docs == 0 else f"answer{i}",
        question_type=question_type, evidence_list=_evidence(n_gold_docs),
    )


def _synthetic_population() -> list[QARecord]:
    # A GLOBAL counter (not reset per bucket) keeps every query string unique across
    # the whole population — otherwise e.g. "comparison_query question 5" would exist
    # in both the hop2 and hop3 buckets, a fixture bug, not a real dataset property.
    counter = iter(range(10_000))
    records = []
    # comparison_query: 75 x hop2, 26 x hop3 (mirrors the real dev_subset.json shape)
    records += [_record("comparison_query", 2, next(counter)) for _ in range(75)]
    records += [_record("comparison_query", 3, next(counter)) for _ in range(26)]
    # inference_query: 20 x hop2, 45 x hop3, 31 x hop4
    records += [_record("inference_query", 2, next(counter)) for _ in range(20)]
    records += [_record("inference_query", 3, next(counter)) for _ in range(45)]
    records += [_record("inference_query", 4, next(counter)) for _ in range(31)]
    # temporal_query: 37 x hop2, 31 x hop3
    records += [_record("temporal_query", 2, next(counter)) for _ in range(37)]
    records += [_record("temporal_query", 3, next(counter)) for _ in range(31)]
    # null_query: 35 x hop0
    records += [_record("null_query", 0, next(counter)) for _ in range(35)]
    return records


# --- stratum_key ---------------------------------------------------------------------------


def test_stratum_key_null_query_has_no_hop_suffix():
    rec = _record("null_query", 0, 0)
    assert stratum_key(rec) == "null_query"


def test_stratum_key_non_null_includes_hop_count():
    rec = _record("inference_query", 3, 0)
    assert stratum_key(rec) == "inference_query::hop3"


# --- select_phase9_sample -------------------------------------------------------------------


def test_default_size_is_50():
    assert PHASE9_SAMPLE_SIZE == 50


def test_selection_returns_exactly_the_requested_size():
    population = _synthetic_population()
    assert len(population) == 300
    sample_meta, selected = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    assert len(selected) == 50
    assert sum(sample_meta.distribution.values()) == 50


def test_selection_covers_every_question_type_present_in_population():
    population = _synthetic_population()
    _, selected = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    selected_types = {r.question_type for r in selected}
    assert selected_types == {"comparison_query", "inference_query", "temporal_query", "null_query"}


def test_selection_includes_multiple_hop_counts_where_applicable():
    """Every non-null question_type in the synthetic population spans
    multiple hop counts — the 50-question sample must include more than
    one hop count per such type (not collapse to a single hop bucket)."""
    from mhrag.eval.ground_truth import hop_count

    population = _synthetic_population()
    _, selected = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    for qtype in ("comparison_query", "inference_query", "temporal_query"):
        hops = {hop_count(r) for r in selected if r.question_type == qtype}
        assert len(hops) >= 2, f"{qtype} sample only covers hop count(s) {hops}"


def test_no_duplicate_records_selected():
    population = _synthetic_population()
    _, selected = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    queries = [r.query for r in selected]
    assert len(queries) == len(set(queries))


def test_deterministic_given_fixed_seed():
    population = _synthetic_population()
    _, selected1 = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    _, selected2 = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    assert [r.query for r in selected1] == [r.query for r in selected2]


def test_different_seed_gives_a_different_sample():
    population = _synthetic_population()
    _, selected1 = select_phase9_sample(population, size=50, seed=1)
    _, selected2 = select_phase9_sample(population, size=50, seed=2)
    assert [r.query for r in selected1] != [r.query for r in selected2]


def test_output_order_is_deterministic_by_type_then_hop_then_query():
    from mhrag.eval.ground_truth import hop_count

    population = _synthetic_population()
    _, selected = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    keys = [(r.question_type, hop_count(r), r.query) for r in selected]
    assert keys == sorted(keys)


def test_question_type_distribution_rollup_matches_selected_records():
    population = _synthetic_population()
    sample_meta, selected = select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
    from collections import Counter

    assert sample_meta.question_type_distribution == dict(Counter(r.question_type for r in selected))


def test_size_larger_than_population_raises():
    import pytest

    population = _synthetic_population()[:10]
    with pytest.raises(ValueError):
        select_phase9_sample(population, size=50, seed=PHASE9_SAMPLE_SEED)
