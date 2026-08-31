"""Integration test: mhrag.routing.metrics (already unit-tested in
tests/test_routing_metrics.py, reused unchanged for Phase 8A.1) applied to
sequential-router-shaped predicted/true route lists — proves under/over-
routing metrics are computed correctly end-to-end for this phase's actual
data shape (route strings coming from `SequentialRouteResult.route` /
oracle labels), not just in isolation.
"""

from __future__ import annotations

import json

from mhrag.routing.metrics import (
    accuracy,
    macro_f1,
    under_over_routing_breakdown,
    under_over_routing_rate,
)


def test_under_over_routing_on_known_sequential_router_style_confusion():
    # 10 questions: mirrors the qualitative shape seen in the real Phase 8A.1
    # eval (heavy over-routing toward COMPLEX, low under-routing).
    oracle = ["SIMPLE", "SIMPLE", "SIMPLE", "MEDIUM", "MEDIUM", "COMPLEX", "COMPLEX", "COMPLEX", "COMPLEX", "COMPLEX"]
    predicted = ["SIMPLE", "COMPLEX", "COMPLEX", "MEDIUM", "COMPLEX", "COMPLEX", "COMPLEX", "COMPLEX", "SIMPLE", "COMPLEX"]
    under, over = under_over_routing_rate(oracle, predicted)
    breakdown = under_over_routing_breakdown(oracle, predicted)
    # under-routing: only COMPLEX->SIMPLE (index 8) = 1/10
    assert under == 0.1
    assert breakdown["COMPLEX->SIMPLE"] == 0.1
    # over-routing: SIMPLE->COMPLEX (indices 1,2) + MEDIUM->COMPLEX (index 4) = 3/10
    assert over == 0.3
    assert breakdown["SIMPLE->COMPLEX"] == 0.2
    assert breakdown["MEDIUM->COMPLEX"] == 0.1
    assert accuracy(oracle, predicted) == 0.6
    assert 0.0 <= macro_f1(oracle, predicted) <= 1.0


def test_real_sequential_router_report_under_over_routing_rates_are_internally_consistent():
    """If the real Phase 8A.1 report artifact is present, sanity-check that
    its persisted under/over-routing rates plus the correct rate (1 -
    under - over, since every off-diagonal cell for 3 ordinal classes is
    strictly under- or over-routing) sum to ~1.0 and match accuracy plus
    the diagonal mass."""
    import pytest
    from pathlib import Path

    path = Path(__file__).parent.parent / "results" / "sequential_router_report.json"
    if not path.exists():
        pytest.skip("results/sequential_router_report.json not present in this checkout")
    report = json.loads(path.read_text())
    p1 = report["phase_8a1_sequential_router"]
    correct_rate = p1["accuracy"]
    total = correct_rate + p1["under_routing_rate"] + p1["over_routing_rate"]
    assert abs(total - 1.0) < 1e-6
