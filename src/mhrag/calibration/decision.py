"""Objective decision rule for selecting the frozen agentic token budget.

Pure function of measured aggregate metrics — no live calls, fully
unit-testable with synthetic `BudgetMetrics`. Prefers the SMALLEST budget
that, relative to the smallest candidate tested (the baseline):

  1. materially reduces premature `token_budget` stops
     (>= TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD percentage points lower)
  2. allows meaningful second/third-hop evidence recovery
     (mean new unique documents gained after hop 1 >= MIN_HOP2_PLUS_NEW_DOCS)
  3. does not cause disproportionate cost/latency growth
     (<= MAX_COST_LATENCY_GROWTH_RATIO x baseline, for both)
  4. does not materially worsen evidence quality
     (mean recall and mean complete-evidence rate each no more than
     MAX_QUALITY_REGRESSION below baseline)

Iterates candidates in ascending budget order (excluding the baseline
itself) and returns the FIRST one meeting all four criteria — never simply
the highest budget, and the baseline is kept if none qualify.

Thresholds are named constants, not inline literals, so the rule is
auditable and applies identically regardless of which budget "wins."
"""

from __future__ import annotations

from dataclasses import dataclass

TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD = 0.15  # >= 15 percentage points lower than baseline
MIN_HOP2_PLUS_NEW_DOCS = 0.3  # mean new unique docs gained after hop 1, across all sampled queries
MAX_COST_LATENCY_GROWTH_RATIO = 1.5  # candidate mean cost/latency must not exceed 1.5x baseline
MAX_QUALITY_REGRESSION = 0.02  # candidate mean recall / complete-evidence rate must not drop > 2pp


@dataclass(frozen=True, slots=True)
class BudgetMetrics:
    token_budget: int
    token_budget_stop_rate: float  # fraction of queries stopped by stop_reason="token_budget"
    mean_new_unique_docs_after_hop1: float  # mean(sum(new_unique_docs_per_hop[1:])) across queries
    mean_recall: float
    mean_complete_evidence_rate: float
    mean_cost_usd: float
    mean_latency_ms: float


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    selected_token_budget: int
    baseline_token_budget: int
    rationale: str
    criteria_by_candidate: dict[int, dict[str, bool]]


def select_budget(candidates: list[BudgetMetrics]) -> BudgetDecision:
    if not candidates:
        raise ValueError("select_budget requires at least one candidate")

    ordered = sorted(candidates, key=lambda c: c.token_budget)
    baseline = ordered[0]
    criteria_by_candidate: dict[int, dict[str, bool]] = {}

    for candidate in ordered[1:]:
        reduces_budget_stops = (
            baseline.token_budget_stop_rate - candidate.token_budget_stop_rate
        ) >= TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD
        meaningful_recovery = candidate.mean_new_unique_docs_after_hop1 >= MIN_HOP2_PLUS_NEW_DOCS
        cost_ok = candidate.mean_cost_usd <= baseline.mean_cost_usd * MAX_COST_LATENCY_GROWTH_RATIO
        latency_ok = candidate.mean_latency_ms <= baseline.mean_latency_ms * MAX_COST_LATENCY_GROWTH_RATIO
        recall_ok = candidate.mean_recall >= baseline.mean_recall - MAX_QUALITY_REGRESSION
        complete_evidence_ok = (
            candidate.mean_complete_evidence_rate
            >= baseline.mean_complete_evidence_rate - MAX_QUALITY_REGRESSION
        )

        criteria = {
            "reduces_budget_stops": reduces_budget_stops,
            "meaningful_recovery": meaningful_recovery,
            "cost_ok": cost_ok,
            "latency_ok": latency_ok,
            "recall_ok": recall_ok,
            "complete_evidence_ok": complete_evidence_ok,
        }
        criteria_by_candidate[candidate.token_budget] = criteria

        if all(criteria.values()):
            return BudgetDecision(
                selected_token_budget=candidate.token_budget,
                baseline_token_budget=baseline.token_budget,
                rationale=(
                    f"{candidate.token_budget} selected: reduced token_budget stop rate by "
                    f"{(baseline.token_budget_stop_rate - candidate.token_budget_stop_rate):.1%} "
                    f"(>= {TOKEN_BUDGET_STOP_REDUCTION_THRESHOLD:.0%} threshold), gained "
                    f"{candidate.mean_new_unique_docs_after_hop1:.2f} new unique docs/query after "
                    f"hop 1 on average (>= {MIN_HOP2_PLUS_NEW_DOCS} threshold), cost/latency within "
                    f"{MAX_COST_LATENCY_GROWTH_RATIO}x baseline, and evidence quality (recall, "
                    f"complete-evidence rate) did not regress beyond {MAX_QUALITY_REGRESSION:.0%}."
                ),
                criteria_by_candidate=criteria_by_candidate,
            )

    return BudgetDecision(
        selected_token_budget=baseline.token_budget,
        baseline_token_budget=baseline.token_budget,
        rationale=(
            f"kept baseline {baseline.token_budget}: no larger candidate met all four criteria "
            f"(reduces_budget_stops, meaningful_recovery, cost_ok/latency_ok, recall_ok/"
            f"complete_evidence_ok) — see criteria_by_candidate for the per-candidate breakdown."
        ),
        criteria_by_candidate=criteria_by_candidate,
    )
