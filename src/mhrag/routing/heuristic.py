"""Stage A: deterministic heuristic/confidence router — RUNTIME, no gold,
no LLM call.

A fixed, hand-designed decision structure (documented below) over exactly
two engineered `RetrievalSignals`: `hybrid_top1_score` and
`dense_bm25_jaccard_top10`. The FORM of the rule is fixed code, kept
deliberately small and interpretable; only the four numeric THRESHOLDS are
calibrated — via `mhrag.routing.tune_thresholds.fit_thresholds`, on
`router_tune` only — and then frozen (see `HeuristicThresholds`, persisted
with provenance to results/router_thresholds.json).

Decision structure, in order:

  1. top1 score high AND dense/bm25 agree strongly -> SIMPLE, confident
  2. top1 score low AND dense/bm25 agree weakly -> COMPLEX, confident
     (weak signal from both retrievers = hard query)
  3. top1 score moderate-or-better -> MEDIUM, confident (decent match, not
     top-tier — a plausible reranker win)
  4. otherwise -> not confident -> Stage B (GLM) decides

DESIGN NOTE — a lexical "comparison/temporal marker present" feature was
tried and DROPPED from this decision structure after calibration showed
the intuitive assumption behind it (marker present => harder, multi-
document question) is backwards on the measured oracle labels: of the 210
router_dataset questions WITH a comparison/temporal marker, 97 are
oracle-SIMPLE vs. 87 oracle-COMPLEX; of the 55 WITHOUT one, 35 are oracle-
COMPLEX vs. only 16 SIMPLE. Comparison/temporal MultiHop-RAG questions
often quote article titles/claims directly, which the frozen Hybrid RRF
baseline matches lexically very well — the opposite of what "looks like a
hard multi-hop question" would suggest for a human reader. Rather than
hard-code the (data-contradicted) opposite assumption, Stage A relies only
on the two quantitative retrieval-confidence signals; the marker features
remain available and are still sent to Stage B (GLM), which can use them
as context without a hard-coded (and wrong) prior baked into the decision
tree. See Phase 8A report, "issues discovered".

Never imports `mhrag.routing.oracle` — this module has no notion of a gold
route label, only the frozen numeric thresholds it's handed.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.routing.features import RouterFeatures


@dataclass(frozen=True, slots=True)
class HeuristicThresholds:
    simple_min_top1_score: float
    simple_min_agreement: float
    complex_max_top1_score: float
    complex_max_agreement: float
    medium_min_top1_score: float


# Placeholder only — every real run loads frozen, tuned values from
# results/router_thresholds.json (see scripts/tune_router_thresholds.py).
# Kept permissive (never confident) so an accidental use of the default
# without tuning fails safe by always deferring to Stage B, not by
# silently making confident-but-uncalibrated decisions.
DEFAULT_THRESHOLDS = HeuristicThresholds(
    simple_min_top1_score=float("inf"),
    simple_min_agreement=float("inf"),
    complex_max_top1_score=float("-inf"),
    complex_max_agreement=float("-inf"),
    medium_min_top1_score=float("inf"),
)


@dataclass(frozen=True, slots=True)
class HeuristicVerdict:
    route: str | None  # one of "SIMPLE"/"MEDIUM"/"COMPLEX", or None if not confident
    confident: bool
    reason: str


def classify_heuristic(features: RouterFeatures, thresholds: HeuristicThresholds) -> HeuristicVerdict:
    """Pure, deterministic function of `features` and `thresholds` — same
    inputs always produce the same `HeuristicVerdict`."""
    s = features.retrieval.hybrid_top1_score
    a = features.retrieval.dense_bm25_jaccard_top10

    if s >= thresholds.simple_min_top1_score and a >= thresholds.simple_min_agreement:
        return HeuristicVerdict(
            "SIMPLE", True,
            f"top1_score={s:.4f}>={thresholds.simple_min_top1_score:.4f} and "
            f"agreement={a:.2f}>={thresholds.simple_min_agreement:.2f}",
        )

    if s <= thresholds.complex_max_top1_score and a <= thresholds.complex_max_agreement:
        return HeuristicVerdict(
            "COMPLEX", True,
            f"top1_score={s:.4f}<={thresholds.complex_max_top1_score:.4f} and "
            f"agreement={a:.2f}<={thresholds.complex_max_agreement:.2f}",
        )

    if s >= thresholds.medium_min_top1_score:
        return HeuristicVerdict(
            "MEDIUM", True,
            f"top1_score={s:.4f}>={thresholds.medium_min_top1_score:.4f}, below SIMPLE bar",
        )

    return HeuristicVerdict(None, False, "no rule matched with sufficient confidence — escalate to GLM")
