"""EVALUATOR-ONLY: per-gate mistake analysis for the sequential router.

For every Evidence Gate call, compares the gate's OWN verdict (after
`mhrag.routing.evidence_gate`'s conservative correction) against ground
truth Complete-Evidence@5 (`mhrag.eval.metrics.complete_evidence_at_k`,
using gold document ids from `mhrag.eval.ground_truth.gold_doc_ids`) for
the SAME chunks the gate actually saw — computed strictly AFTER the gate
has already returned its decision, never fed into it.

Since a gate's stage IS the oracle route's defining criterion (Gate 1 vs.
Hybrid CE@5, Gate 2 vs. Hybrid+Reranker CE@5 — see `mhrag.routing.oracle`),
a gate mistake is, by construction, one of exactly two kinds:

  - FALSE SUFFICIENCY (dangerous): ground truth CE@5 is False (the
    required evidence is NOT actually all present in what the gate saw)
    but the gate said `sufficient=True` anyway. This is Phase 8A.1's
    "evidence was incomplete but GLM incorrectly declared sufficient" —
    it causes harmful UNDER-routing (a later, more thorough stage is
    skipped even though evidence is genuinely missing).
  - FALSE INSUFFICIENCY (wasteful, not harmful): ground truth CE@5 is
    True (the evidence WAS actually all present) but the gate said
    `sufficient=False`. This is Phase 8A.1's "evidence was present but GLM
    incorrectly judged it insufficient" — it causes unnecessary
    OVER-routing (extra retrieval cost/latency, not an evidence gap).

A verdict that agrees with ground truth is "correct" (further split into
correct_sufficient / correct_insufficient for visibility). "required
evidence absent from top-5" (Phase 8A.1's analysis item 1) is not a
separate category here — it is definitionally implied whenever ground
truth CE@5 is False, so it is folded into false_sufficiency's own
definition and reported as an explicit `num_gold_docs_missing` count on
each `GateMistake`, rather than a fourth mutually exclusive bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.data.schema import QARecord
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.metrics import collapse_to_unique_documents, complete_evidence_at_k
from mhrag.retrieval.schema import RetrievalResult

GATE_OUTCOMES = (
    "correct_sufficient",
    "correct_insufficient",
    "false_sufficiency",
    "false_insufficiency",
)

CE_K = 5  # matches mhrag.routing.oracle.ORACLE_K — the same answer-context depth


@dataclass(frozen=True, slots=True)
class GateMistakeAnalysis:
    outcome: str  # one of GATE_OUTCOMES
    ground_truth_sufficient: bool
    gate_sufficient: bool
    num_gold_docs: int
    num_gold_docs_present: int
    num_gold_docs_missing: int


def analyze_gate_verdict(
    record: QARecord,
    retrieved_chunks: list[RetrievalResult],
    gate_sufficient: bool,
) -> GateMistakeAnalysis:
    """Evaluator-only — call strictly AFTER the gate has already returned
    its decision. `retrieved_chunks` is exactly what the gate was shown
    (the same top-5 list); `record.evidence_list` is used ONLY here."""
    gold = gold_doc_ids(record)
    unique_doc_ids = collapse_to_unique_documents(retrieved_chunks)
    ground_truth_sufficient = bool(complete_evidence_at_k(unique_doc_ids, gold, CE_K))

    present = gold & set(unique_doc_ids[:CE_K])
    missing = gold - present

    if ground_truth_sufficient and gate_sufficient:
        outcome = "correct_sufficient"
    elif not ground_truth_sufficient and not gate_sufficient:
        outcome = "correct_insufficient"
    elif not ground_truth_sufficient and gate_sufficient:
        outcome = "false_sufficiency"
    else:  # ground_truth_sufficient and not gate_sufficient
        outcome = "false_insufficiency"

    return GateMistakeAnalysis(
        outcome=outcome,
        ground_truth_sufficient=ground_truth_sufficient,
        gate_sufficient=gate_sufficient,
        num_gold_docs=len(gold),
        num_gold_docs_present=len(present),
        num_gold_docs_missing=len(missing),
    )
