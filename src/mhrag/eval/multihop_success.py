"""EVALUATOR-ONLY: multi-hop success analysis.

Answers a narrower, doc-level question than `mhrag.eval.fact_grounding`:
**"did a later agentic hop retrieve a required (gold) document that hop 1
did not have, and did that translate into a better graded answer?"** This
is document-level (via already-persisted `evidence_doc_ids_used`), not
fact-level, and it is computed over the FULL final evidence pool (all
hops), not a hop-1-only lower bound — so it must NEVER be blended with, or
described using, `mhrag.eval.fact_grounding`'s Tier A / Tier B language.
This module does not import `mhrag.eval.fact_grounding` and never will
(see tests/test_multihop_success_no_fact_grounding_mixing.py).

Every function here is pure: no I/O, no model call, no randomness. Callers
(scripts/analyze_multihop_success.py) are responsible for loading the
already-persisted Phase 9 raw traces, the Phase 5A hop-1 replay, and the
judge score files from disk and assembling `QuestionOutcome` objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """Everything needed to classify one multi-hop-resolved question,
    assembled entirely from already-persisted artifacts."""

    qa_id: str
    question_type: str
    stop_reason: str
    num_agent_hops: int
    gold_doc_ids: frozenset[str]
    hop1_doc_ids: frozenset[str]  # from the Phase 5A hop-1 replay — mathematically == agentic hop 1
    final_doc_ids: frozenset[str]  # agentic's final evidence_doc_ids_used (ALL hops)
    baseline_doc_ids: frozenset[str] | None  # Hybrid+Reranker's evidence_doc_ids_used, if present
    matched_doc_ids: frozenset[str] | None  # Context-Matched's evidence_doc_ids_used, if present
    agentic_grade: str | None
    baseline_grade: str | None
    matched_grade: str | None


def coverage(doc_ids: frozenset[str] | None, gold: frozenset[str]) -> float | None:
    """Fraction of `gold` present in `doc_ids`. None if gold is empty (no
    coverage is defined for a null/gold-less question) or doc_ids is
    unknown for this pipeline."""
    if not gold or doc_ids is None:
        return None
    return len(doc_ids & gold) / len(gold)


def added_required_evidence(outcome: QuestionOutcome) -> frozenset[str]:
    """Gold doc_ids present in the final agentic evidence pool but absent
    from hop 1 — the documents a later hop is responsible for recovering.
    Empty iff no later hop added anything the question actually needed."""
    return (outcome.final_doc_ids - outcome.hop1_doc_ids) & outcome.gold_doc_ids


@dataclass(frozen=True, slots=True)
class ExampleCandidate:
    qa_id: str
    question_type: str
    stop_reason: str
    tier: int  # 1 = agentic correct AND both baseline+matched incorrect; 2 = agentic correct AND beats >=1


def classify_tier(outcome: QuestionOutcome) -> int | None:
    """None if this question isn't example-worthy at all (agentic itself
    wasn't graded correct, or no judge grade exists).
    Tier 1: agentic correct, baseline AND matched both incorrect — iteration
        is the cleanest explanation available for the win.
    Tier 2: agentic correct, beats at least one of baseline/matched — a
        weaker but still real signal, used only as a fallback (see
        select_examples)."""
    if outcome.agentic_grade != "correct":
        return None
    baseline_wrong = outcome.baseline_grade == "incorrect"
    matched_wrong = outcome.matched_grade == "incorrect"
    if baseline_wrong and matched_wrong:
        return 1
    if baseline_wrong or matched_wrong:
        return 2
    return None


def select_examples(
    candidates: list[ExampleCandidate],
    excluded_qa_ids: frozenset[str],
    max_examples: int = 5,
) -> list[str]:
    """Deterministic, diversity-preferring example selection — a pure
    function of its inputs, never hand-picked. Same input always produces
    the same output (see tests/test_multihop_success.py::test_selection_is_deterministic).

    Algorithm:
      1. Drop excluded qa_ids (known evaluator-quirk cases).
      2. Process Tier 1 candidates before any Tier 2 candidate — Tier 1
         is strictly preferred, never diluted by a "more diverse" Tier 2
         pick.
      3. Within a tier, greedily pick the candidate that covers the most
         NEW (question_type, stop_reason) facets not yet represented among
         already-selected examples — a walkthrough that repeats the same
         question_type/stop_reason five times is less informative than one
         that shows the mechanism firing under different conditions.
      4. Ties broken by ascending qa_id — qa_id is a content hash (see
         mhrag.data.benchmark.qa_id), not a database auto-increment or run
         order, so this tie-break introduces no hidden recency/order bias.
    """
    eligible = [c for c in candidates if c.qa_id not in excluded_qa_ids]
    tier1 = sorted((c for c in eligible if c.tier == 1), key=lambda c: c.qa_id)
    tier2 = sorted((c for c in eligible if c.tier == 2), key=lambda c: c.qa_id)

    selected: list[ExampleCandidate] = []
    seen_types: set[str] = set()
    seen_stop_reasons: set[str] = set()

    def _greedy_pick(pool: list[ExampleCandidate]) -> None:
        remaining = list(pool)
        while remaining and len(selected) < max_examples:
            def _novelty(c: ExampleCandidate) -> int:
                return (c.question_type not in seen_types) + (c.stop_reason not in seen_stop_reasons)

            remaining.sort(key=lambda c: (-_novelty(c), c.qa_id))
            pick = remaining.pop(0)
            selected.append(pick)
            seen_types.add(pick.question_type)
            seen_stop_reasons.add(pick.stop_reason)

    _greedy_pick(tier1)
    _greedy_pick(tier2)
    return [c.qa_id for c in selected]
