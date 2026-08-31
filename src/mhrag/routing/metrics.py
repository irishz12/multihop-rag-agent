"""EVALUATOR-ONLY: router performance metrics.

Pure functions of (true route, predicted route) pairs — no live call, no
retrieval, no model. `true` routes come from `mhrag.routing.oracle`
(evaluator-only); `predicted` routes come from `mhrag.routing.router.
route_question` results, scored strictly AFTER routing has already run.

Under-routing vs over-routing (Phase 8A spec): under-routing sends a
question to a CHEAPER route than the oracle says it needs (harms answer
quality — the escalation that would have found the missing evidence never
happens); over-routing sends it to a MORE EXPENSIVE route than needed
(harms cost/latency only, quality is preserved or improved). Reported
separately, never combined into one "error rate", per the spec's explicit
statement that they carry different costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.routing.oracle import ROUTE_LABELS

# Ordinal cost order: SIMPLE (cheapest) < MEDIUM < COMPLEX (most expensive).
_ROUTE_RANK = {"SIMPLE": 0, "MEDIUM": 1, "COMPLEX": 2}

UNDER_ROUTING_PAIRS = frozenset(
    {("COMPLEX", "SIMPLE"), ("COMPLEX", "MEDIUM"), ("MEDIUM", "SIMPLE")}
)
OVER_ROUTING_PAIRS = frozenset(
    {("SIMPLE", "MEDIUM"), ("SIMPLE", "COMPLEX"), ("MEDIUM", "COMPLEX")}
)


def confusion_matrix(true_routes: list[str], predicted_routes: list[str]) -> dict[str, dict[str, int]]:
    """`matrix[true][predicted] = count` — every (true, predicted) cell
    from `ROUTE_LABELS` x `ROUTE_LABELS` present (0 if unobserved), so
    downstream consumers never need a `.get(..., 0)`."""
    if len(true_routes) != len(predicted_routes):
        raise ValueError("true_routes and predicted_routes must be the same length")
    matrix = {t: {p: 0 for p in ROUTE_LABELS} for t in ROUTE_LABELS}
    for t, p in zip(true_routes, predicted_routes):
        matrix[t][p] += 1
    return matrix


def accuracy(true_routes: list[str], predicted_routes: list[str]) -> float:
    if not true_routes:
        raise ValueError("accuracy requires at least one example")
    correct = sum(1 for t, p in zip(true_routes, predicted_routes) if t == p)
    return correct / len(true_routes)


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int  # number of true examples of this class


def per_class_metrics(true_routes: list[str], predicted_routes: list[str]) -> dict[str, ClassMetrics]:
    matrix = confusion_matrix(true_routes, predicted_routes)
    result: dict[str, ClassMetrics] = {}
    for label in ROUTE_LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in ROUTE_LABELS if p != label)
        fp = sum(matrix[t][label] for t in ROUTE_LABELS if t != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[label] = ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)
    return result


def macro_f1(true_routes: list[str], predicted_routes: list[str]) -> float:
    per_class = per_class_metrics(true_routes, predicted_routes)
    return sum(m.f1 for m in per_class.values()) / len(per_class)


def under_over_routing_rate(true_routes: list[str], predicted_routes: list[str]) -> tuple[float, float]:
    """Returns (under_routing_rate, over_routing_rate) — each the fraction
    of ALL examples (not just misclassified ones) whose (true, predicted)
    pair falls in `UNDER_ROUTING_PAIRS` / `OVER_ROUTING_PAIRS`
    respectively. The two rates plus the correct-rate plus any "other
    misroute" rate (none exist here — every off-diagonal cell for 3
    ordinal classes is either strictly under- or strictly over-routing) sum
    to 1.0."""
    if len(true_routes) != len(predicted_routes):
        raise ValueError("true_routes and predicted_routes must be the same length")
    n = len(true_routes)
    if n == 0:
        raise ValueError("under_over_routing_rate requires at least one example")
    under = sum(1 for t, p in zip(true_routes, predicted_routes) if (t, p) in UNDER_ROUTING_PAIRS)
    over = sum(1 for t, p in zip(true_routes, predicted_routes) if (t, p) in OVER_ROUTING_PAIRS)
    return under / n, over / n


def under_over_routing_breakdown(true_routes: list[str], predicted_routes: list[str]) -> dict[str, float]:
    """Per-pair breakdown (fraction of all examples), keyed
    "TRUE->PREDICTED", for exactly the 6 pairs the spec calls out by name —
    useful for the item-11/12 report tables."""
    n = len(true_routes)
    if n == 0:
        raise ValueError("under_over_routing_breakdown requires at least one example")
    pairs = list(UNDER_ROUTING_PAIRS | OVER_ROUTING_PAIRS)
    counts = {f"{t}->{p}": 0 for t, p in pairs}
    for t, p in zip(true_routes, predicted_routes):
        key = f"{t}->{p}"
        if key in counts:
            counts[key] += 1
    return {k: v / n for k, v in counts.items()}
