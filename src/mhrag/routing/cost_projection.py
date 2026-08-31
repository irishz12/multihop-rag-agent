"""EVALUATOR-ONLY: cost-aware routing-level projection.

Given a predicted route DISTRIBUTION (counts of SIMPLE/MEDIUM/COMPLEX from
`mhrag.routing.router.route_question` results) and known per-query
cost/latency unit figures for each backend, projects the total/average
workload cost and latency this routing would produce — WITHOUT actually
running the Hybrid+Reranker or Agentic pipelines for real. This is a
routing-level projection only, not the final end-to-end project result
(the Phase 8A spec is explicit about this).

Unit costs/latencies come from ALREADY-MEASURED artifacts from earlier
phases (results/retrieval_eval_development.json for Hybrid/Hybrid+
Reranker latency, results/agentic_budget_calibration_4500.json for
Agentic cost/latency at the frozen Phase 7.1 budget) — this module does
not measure anything itself, it only combines numbers already measured
elsewhere with a predicted route distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUTE_TO_BACKEND = {
    "SIMPLE": "hybrid_only",
    "MEDIUM": "hybrid_reranker",
    "COMPLEX": "agentic",
}
BACKENDS = ("hybrid_only", "hybrid_reranker", "agentic")


@dataclass(frozen=True, slots=True)
class UnitCost:
    """Per-query cost (USD) and latency (ms) for one backend."""

    cost_usd: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class WorkloadProjection:
    route_counts: dict[str, int]
    backend_counts: dict[str, int]
    n_queries: int
    total_cost_usd: float
    mean_cost_usd: float
    total_latency_ms: float
    mean_latency_ms: float


def route_counts_to_backend_counts(route_counts: dict[str, int]) -> dict[str, int]:
    backend_counts = {b: 0 for b in BACKENDS}
    for route, count in route_counts.items():
        backend_counts[ROUTE_TO_BACKEND[route]] += count
    return backend_counts


def project_workload(route_counts: dict[str, int], unit_costs: dict[str, UnitCost]) -> WorkloadProjection:
    """`unit_costs` keyed by backend name (`BACKENDS`)."""
    missing = set(BACKENDS) - set(unit_costs)
    if missing:
        raise ValueError(f"unit_costs missing backend(s): {sorted(missing)}")

    backend_counts = route_counts_to_backend_counts(route_counts)
    n = sum(backend_counts.values())
    if n == 0:
        raise ValueError("project_workload requires at least one routed query")

    total_cost = sum(backend_counts[b] * unit_costs[b].cost_usd for b in BACKENDS)
    total_latency = sum(backend_counts[b] * unit_costs[b].latency_ms for b in BACKENDS)

    return WorkloadProjection(
        route_counts=dict(route_counts),
        backend_counts=backend_counts,
        n_queries=n,
        total_cost_usd=total_cost,
        mean_cost_usd=total_cost / n,
        total_latency_ms=total_latency,
        mean_latency_ms=total_latency / n,
    )


def agentic_multi_hop_projection(n_queries: int, agentic_unit_cost: UnitCost) -> WorkloadProjection:
    """The comparison baseline: 100% of queries go through Agentic Multi-Hop
    RAG retrieval — what Phase 7/7.1 already measured directly, expressed
    here only for a like-for-like comparison against the routed projection."""
    if n_queries <= 0:
        raise ValueError("agentic_multi_hop_projection requires n_queries > 0")
    route_counts = {"SIMPLE": 0, "MEDIUM": 0, "COMPLEX": n_queries}
    unit_costs = {
        "hybrid_only": UnitCost(0.0, 0.0),
        "hybrid_reranker": UnitCost(0.0, 0.0),
        "agentic": agentic_unit_cost,
    }
    return project_workload(route_counts, unit_costs)
