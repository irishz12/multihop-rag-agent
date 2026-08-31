"""EVALUATOR-ONLY: fit Stage A's `HeuristicThresholds` on `router_tune`.

Touches oracle route labels — this module must NEVER be imported by the
runtime router (`mhrag.routing.router`, `mhrag.routing.heuristic`,
`mhrag.routing.features`, `mhrag.routing.glm_router`); see
tests/test_routing_no_gold_leakage.py.

Fitting method (deterministic, small, auditable — not a black-box
optimizer): the decision STRUCTURE in `mhrag.routing.heuristic.
classify_heuristic` is fixed code; only its 5 numeric thresholds vary here.
`fit_thresholds` grid-searches over quantiles of the tune set's own
`hybrid_top1_score` (for the three score thresholds, kept ordered
complex <= medium <= simple by construction of the search) and
`dense_bm25_jaccard_top10` (for the two agreement thresholds, kept ordered
complex <= simple), evaluating tune-set accuracy for every valid ordered
combination and keeping the single best-scoring one (ties broken toward
fewer confident-but-wrong calls, then toward the combination appearing
first when thresholds are sorted ascending, for full determinism).
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.routing.features import RouterFeatures
from mhrag.routing.heuristic import HeuristicThresholds, classify_heuristic

# Quantile points (as fractions) sampled from the tune set's own signal
# distributions to build candidate threshold values — deliberately coarse
# (7 points) to keep the grid small, auditable, and fast; not fit to
# validation data at any point.
CANDIDATE_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True, slots=True)
class TuneExample:
    qa_id: str
    features: RouterFeatures
    oracle_route: str  # ground truth — evaluator-only


def _quantiles(values: list[float], fractions: tuple[float, ...]) -> list[float]:
    if not values:
        return [0.0 for _ in fractions]
    ordered = sorted(values)
    n = len(ordered)
    out = []
    for f in fractions:
        idx = min(n - 1, max(0, round(f * (n - 1))))
        out.append(ordered[idx])
    # dedupe while preserving order (identical quantile values collapse the grid, which is fine)
    seen: list[float] = []
    for v in out:
        if v not in seen:
            seen.append(v)
    return seen


def _score_thresholds(examples: list[TuneExample], thresholds: HeuristicThresholds) -> tuple[int, int]:
    """Returns (num_correct_confident, num_confident) — used to rank
    candidates by tune accuracy among CONFIDENT calls (the ambiguous zone
    always defers to Stage B, so it doesn't count against or for Stage A's
    own accuracy) while `fit_thresholds` also enforces a minimum coverage
    floor separately."""
    correct = 0
    confident = 0
    for ex in examples:
        verdict = classify_heuristic(ex.features, thresholds)
        if verdict.confident:
            confident += 1
            if verdict.route == ex.oracle_route:
                correct += 1
    return correct, confident


def fit_thresholds(
    examples: list[TuneExample],
    candidate_quantiles: tuple[float, ...] = CANDIDATE_QUANTILES,
    min_coverage: float = 0.3,
) -> HeuristicThresholds:
    """Grid search over `examples` (router_tune only) for the
    `HeuristicThresholds` combination maximizing confident-call accuracy,
    subject to covering (making a confident call for) at least
    `min_coverage` of the tune set — a threshold set that is "accurate"
    only because it almost never commits to an answer is not useful, so
    near-zero-coverage solutions are excluded from consideration rather
    than winning by default.

    Deterministic: candidates come from `examples`' own signal quantiles,
    ties broken by iteration order over a fixed, sorted candidate list.
    """
    if not examples:
        raise ValueError("fit_thresholds requires at least one tune example")

    scores = [ex.features.retrieval.hybrid_top1_score for ex in examples]
    agreements = [ex.features.retrieval.dense_bm25_jaccard_top10 for ex in examples]
    score_candidates = sorted(_quantiles(scores, candidate_quantiles))
    agreement_candidates = sorted(_quantiles(agreements, candidate_quantiles))

    best: HeuristicThresholds | None = None
    best_key: tuple[float, int] | None = None  # (accuracy, coverage) — maximize both, accuracy first

    for s_low in score_candidates:
        for s_mid in [s for s in score_candidates if s >= s_low]:
            for s_high in [s for s in score_candidates if s >= s_mid]:
                for a_low in agreement_candidates:
                    for a_high in [a for a in agreement_candidates if a >= a_low]:
                        candidate = HeuristicThresholds(
                            simple_min_top1_score=s_high,
                            simple_min_agreement=a_high,
                            complex_max_top1_score=s_low,
                            complex_max_agreement=a_low,
                            medium_min_top1_score=s_mid,
                        )
                        correct, confident = _score_thresholds(examples, candidate)
                        coverage = confident / len(examples)
                        if coverage < min_coverage:
                            continue
                        accuracy = correct / confident if confident else 0.0
                        key = (accuracy, coverage)
                        if best_key is None or key > best_key:
                            best_key = key
                            best = candidate

    if best is None:
        raise ValueError(
            f"no threshold combination reached the minimum coverage ({min_coverage:.0%}) on the "
            "tune set — widen candidate_quantiles or lower min_coverage"
        )
    return best
