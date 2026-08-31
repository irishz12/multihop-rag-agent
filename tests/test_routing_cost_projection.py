"""Cost-aware workload projection tests — offline, pure arithmetic."""

from __future__ import annotations

import pytest

from mhrag.routing.cost_projection import (
    UnitCost,
    agentic_multi_hop_projection,
    project_workload,
    route_counts_to_backend_counts,
)


def test_route_counts_to_backend_counts_maps_correctly():
    counts = route_counts_to_backend_counts({"SIMPLE": 10, "MEDIUM": 5, "COMPLEX": 3})
    assert counts == {"hybrid_only": 10, "hybrid_reranker": 5, "agentic": 3}


def test_project_workload_computes_weighted_totals():
    unit_costs = {
        "hybrid_only": UnitCost(cost_usd=0.0, latency_ms=60.0),
        "hybrid_reranker": UnitCost(cost_usd=0.0, latency_ms=200.0),
        "agentic": UnitCost(cost_usd=0.0012, latency_ms=8550.0),
    }
    projection = project_workload({"SIMPLE": 10, "MEDIUM": 5, "COMPLEX": 3}, unit_costs)
    assert projection.n_queries == 18
    expected_cost = 3 * 0.0012
    assert projection.total_cost_usd == pytest.approx(expected_cost)
    assert projection.mean_cost_usd == pytest.approx(expected_cost / 18)
    expected_latency = 10 * 60.0 + 5 * 200.0 + 3 * 8550.0
    assert projection.total_latency_ms == pytest.approx(expected_latency)


def test_project_workload_missing_backend_raises():
    with pytest.raises(ValueError):
        project_workload({"SIMPLE": 1}, {"hybrid_only": UnitCost(0.0, 1.0)})


def test_project_workload_requires_nonzero_queries():
    unit_costs = {b: UnitCost(0.0, 0.0) for b in ("hybrid_only", "hybrid_reranker", "agentic")}
    with pytest.raises(ValueError):
        project_workload({"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 0}, unit_costs)


def test_agentic_multi_hop_projection_routes_everything_to_agentic():
    projection = agentic_multi_hop_projection(27, UnitCost(cost_usd=0.0012, latency_ms=8550.0))
    assert projection.route_counts == {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": 27}
    assert projection.backend_counts["agentic"] == 27
    assert projection.total_cost_usd == pytest.approx(27 * 0.0012)
    assert projection.mean_cost_usd == pytest.approx(0.0012)


def test_agentic_multi_hop_projection_requires_positive_n():
    with pytest.raises(ValueError):
        agentic_multi_hop_projection(0, UnitCost(0.0, 0.0))


def test_routed_workload_cheaper_than_agentic_multi_hop_when_most_queries_are_simple():
    unit_costs = {
        "hybrid_only": UnitCost(cost_usd=0.0, latency_ms=60.0),
        "hybrid_reranker": UnitCost(cost_usd=0.0, latency_ms=200.0),
        "agentic": UnitCost(cost_usd=0.0012, latency_ms=8550.0),
    }
    routed = project_workload({"SIMPLE": 20, "MEDIUM": 5, "COMPLEX": 2}, unit_costs)
    agentic_multi_hop = agentic_multi_hop_projection(27, unit_costs["agentic"])
    assert routed.total_cost_usd < agentic_multi_hop.total_cost_usd
    assert routed.total_latency_ms < agentic_multi_hop.total_latency_ms
