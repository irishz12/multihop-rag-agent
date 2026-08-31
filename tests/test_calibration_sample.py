"""Calibration sample selection tests — offline, synthetic QARecord
population (same style as tests/test_benchmark.py). Proves the sample is
DEVELOPMENT-shaped (null_query excluded, deterministic, balanced) without
needing the real dataset file."""

from __future__ import annotations

import pytest

from mhrag.calibration.sample import CELL_TARGETS, select_calibration_sample
from mhrag.data.schema import Evidence, QARecord
from mhrag.eval.ground_truth import hop_count


def _evidence(url: str) -> Evidence:
    return Evidence(
        title="t", author=None, url=url, source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", fact="fact",
    )


def _record(question_type: str, n_docs: int, idx: int) -> QARecord:
    urls = [f"https://example.com/{question_type}-{idx}-doc{i}" for i in range(n_docs)]
    return QARecord(
        query=f"{question_type} question {idx} ({n_docs} docs)",
        answer="an answer",
        question_type=question_type,
        evidence_list=tuple(_evidence(u) for u in urls),
    )


def _synthetic_population() -> list[QARecord]:
    records = []
    # Enough candidates per real-data-shaped cell to satisfy CELL_TARGETS
    # with headroom (10 each), matching the real constraint that 4-hop
    # only exists for inference_query.
    for qtype, hops in [
        ("inference_query", 2), ("inference_query", 3), ("inference_query", 4),
        ("comparison_query", 2), ("comparison_query", 3),
        ("temporal_query", 2), ("temporal_query", 3),
    ]:
        for i in range(10):
            records.append(_record(qtype, hops, i))
    # a null_query record that must never be selected
    records.append(QARecord(query="null q", answer="Insufficient information", question_type="null_query", evidence_list=()))
    return records


def test_sample_size_matches_cell_targets_sum():
    sample = select_calibration_sample(_synthetic_population())
    assert len(sample) == sum(CELL_TARGETS.values()) == 27


def test_sample_never_includes_null_query():
    sample = select_calibration_sample(_synthetic_population())
    assert all(r.question_type != "null_query" for r in sample)


def test_sample_matches_exact_cell_distribution():
    sample = select_calibration_sample(_synthetic_population())
    from collections import Counter

    counts = Counter((r.question_type, hop_count(r)) for r in sample)
    assert dict(counts) == CELL_TARGETS


def test_sample_is_deterministic_across_calls():
    pop = _synthetic_population()
    first = select_calibration_sample(pop)
    second = select_calibration_sample(pop)
    assert [r.query for r in first] == [r.query for r in second]


def test_sample_is_deterministic_across_fresh_population_objects():
    first = select_calibration_sample(_synthetic_population())
    second = select_calibration_sample(_synthetic_population())
    assert [r.query for r in first] == [r.query for r in second]


def test_different_seed_can_change_selection_within_a_cell():
    pop = _synthetic_population()
    default_sample = select_calibration_sample(pop, seed=42)
    other_sample = select_calibration_sample(pop, seed=7)
    assert [r.query for r in default_sample] != [r.query for r in other_sample]


def test_insufficient_candidates_in_a_cell_raises_not_silently_underfills():
    small_pop = [_record("inference_query", 2, i) for i in range(2)]  # cell target is 3
    with pytest.raises(ValueError, match="only 2 candidate"):
        select_calibration_sample(small_pop, cell_targets={("inference_query", 2): 3})


def test_custom_cell_targets_respected():
    pop = _synthetic_population()
    custom = {("inference_query", 2): 2, ("comparison_query", 3): 1}
    sample = select_calibration_sample(pop, cell_targets=custom)
    assert len(sample) == 3
    from collections import Counter

    counts = Counter((r.question_type, hop_count(r)) for r in sample)
    assert dict(counts) == custom
