"""router_tune / router_validation split tests — offline, synthetic
OracleLabels, plus one real-data check against final_holdout.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mhrag.routing.oracle import OracleLabel
from mhrag.routing.split import TUNE_FRACTION, split_tune_validation


def _labels(n_simple, n_medium, n_complex) -> list[OracleLabel]:
    labels = []
    for i in range(n_simple):
        labels.append(OracleLabel(f"simple_{i}", "inference_query", 2, "SIMPLE", True, True))
    for i in range(n_medium):
        labels.append(OracleLabel(f"medium_{i}", "comparison_query", 2, "MEDIUM", False, True))
    for i in range(n_complex):
        labels.append(OracleLabel(f"complex_{i}", "temporal_query", 3, "COMPLEX", False, False))
    return labels


def test_split_is_disjoint_and_covers_every_qa_id():
    labels = _labels(20, 20, 20)
    tune, validation = split_tune_validation(labels)
    assert set(tune) & set(validation) == set()
    assert set(tune) | set(validation) == {label.qa_id for label in labels}


def test_split_is_approximately_70_30():
    labels = _labels(50, 50, 50)
    tune, validation = split_tune_validation(labels)
    total = len(labels)
    assert abs(len(tune) / total - TUNE_FRACTION) < 0.05


def test_split_stratified_by_route_label():
    labels = _labels(20, 20, 20)
    tune, validation = split_tune_validation(labels)
    by_id = {label.qa_id: label.route for label in labels}
    tune_routes = [by_id[q] for q in tune]
    for route in ("SIMPLE", "MEDIUM", "COMPLEX"):
        # each route's tune share should also land near 70%
        route_total = sum(1 for label in labels if label.route == route)
        route_in_tune = sum(1 for r in tune_routes if r == route)
        assert abs(route_in_tune / route_total - TUNE_FRACTION) < 0.15


def test_split_is_deterministic_across_calls():
    labels = _labels(15, 15, 15)
    t1, v1 = split_tune_validation(labels)
    t2, v2 = split_tune_validation(labels)
    assert t1 == t2
    assert v1 == v2


def test_different_seed_can_change_split():
    labels = _labels(15, 15, 15)
    t1, _ = split_tune_validation(labels, seed=1)
    t2, _ = split_tune_validation(labels, seed=2)
    assert t1 != t2


def test_small_route_group_still_split_without_crashing():
    labels = _labels(3, 1, 3)  # MEDIUM has only 1 example
    tune, validation = split_tune_validation(labels)
    assert set(tune) | set(validation) == {label.qa_id for label in labels}


def test_router_split_qa_ids_never_overlap_final_holdout():
    """Real-data guard: router_tune/validation are built from
    dev_subset.json qa_ids (via oracle labels derived from the dev-only
    retrieval eval artifact) — must never intersect final_holdout.json."""
    root = Path(__file__).parent.parent
    dev_path = root / "data" / "processed" / "dev_subset.json"
    holdout_path = root / "data" / "processed" / "final_holdout.json"
    if not (dev_path.exists() and holdout_path.exists()):
        pytest.skip("real split files not present in this checkout")

    from mhrag.data.benchmark import qa_id
    from mhrag.data.loader import load_qa_records

    dev_records = load_qa_records(dev_path)
    holdout_records = load_qa_records(holdout_path)
    holdout_ids = {qa_id(r) for r in holdout_records}

    non_null = [r for r in dev_records if r.question_type != "null_query"]
    labels = [
        OracleLabel(qa_id(r), r.question_type, 2, "SIMPLE", True, True) for r in non_null
    ]  # route value irrelevant for this overlap check
    tune, validation = split_tune_validation(labels)
    assert not (set(tune) | set(validation)) & holdout_ids
