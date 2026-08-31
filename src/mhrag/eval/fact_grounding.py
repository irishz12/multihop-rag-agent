"""EVALUATOR-ONLY: retrieval-side fact-level grounding.

Answers exactly one question: **"did the required gold facts reach the
generation context?"** This is NOT "did the generated answer actually use
or correctly express those facts" — that is a separate, harder question
(answer grounding / entailment) explicitly NOT implemented here, per the
Phase 5 design's deliberate split between evidence-retrieval grounding and
answer grounding. Every place this module's output is reported, it must
be labeled "retrieval-side fact-level grounding," never "grounding
evaluation" or "full grounding evaluation" unqualified.

FACT MATCHING — EXACT/NORMALIZED SUBSTRING ONLY, NO SEMANTIC APPROACH:
`Evidence.fact` is empirically verified extractive, not paraphrased — the
Phase 5 audit found 722/722 gold facts (100%) are exact substrings of
their cited corpus document's body text, and Phase 5A's chunk-survival
check (results/fact_grounding_chunk_survival.json) found all 722 survive
intact inside exactly one production chunk (zero cross-chunk, zero
absent). Given that, a semantic/embedding/LLM-judge approach would solve
a paraphrase problem this dataset does not have while introducing a real,
already-demonstrated failure mode this project paid to discover the hard
way (Task Success's response_structure heuristic: 35-85% false-positive
rates on real text before being downgraded to experimental). Exact
substring matching is therefore not a shortcut here — it is the
methodologically correct choice for this specific, extractive dataset.

SINGLE-CHUNK CHECK ONLY, NO CROSS-CHUNK RECONSTRUCTION: because Phase 5A
found zero facts require cross-chunk reconstruction in this corpus, this
module checks each individual context chunk's text, never a concatenation
of adjacent chunks. This is a deliberate scope simplification justified
by already-gathered empirical evidence, not an oversight — it also avoids
a genuine correctness risk a naive concatenation approach would introduce
(a coincidental match spanning an arbitrary chunk boundary that exists in
neither chunk alone). If this module is ever applied to a different
corpus, this assumption must be re-verified first (rerun
scripts/validate_fact_chunk_survival.py's approach for that corpus).

NORMALIZATION: whitespace-collapse, Unicode NFKC + curly-quote/dash
unification, and casefold ONLY — never token-level, never strips or
reorders words, never removes articles/punctuation. This is deliberately
NOT the same function as `mhrag.eval.answer_metrics.normalize_answer_text`
(which strips punctuation and articles — appropriate for comparing a
short verdict token, wrong for a verbatim multi-sentence quote, where
removing punctuation could create a spurious match across what were two
separate sentences).

GOLD DATA USAGE: this module reads `Evidence.fact`/`Evidence.url` exactly
the way `mhrag.eval.judge`/`mhrag.eval.task_success` already read gold
answer text — evaluator-only. Like every other module in `mhrag.eval`,
this module must NEVER be imported by `mhrag.agent`, `mhrag.generation`,
`mhrag.routing`, or `mhrag.adaptive` — see
tests/test_fact_grounding_no_gold_leakage.py.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_TABLE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_fact_text(text: str) -> str:
    """Whitespace-collapse + Unicode NFKC + curly-quote/dash unification +
    casefold. Deliberately conservative — see module docstring for why
    this is NOT `mhrag.eval.answer_metrics.normalize_answer_text`."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_QUOTE_TABLE)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold()


def fact_grounded(fact: str, context_chunk_texts: list[str]) -> bool:
    """True iff the normalized `fact` is a substring of at least one
    normalized chunk in `context_chunk_texts` — the chunks that ACTUALLY
    reached the generation context (post token-budget truncation via
    `mhrag.generation.context.assemble_context`, never raw retrieval
    output — see scripts/compute_fact_grounding.py for how callers must
    obtain `context_chunk_texts`)."""
    fact_norm = normalize_fact_text(fact)
    if not fact_norm:
        return False
    return any(fact_norm in normalize_fact_text(chunk_text) for chunk_text in context_chunk_texts)


@dataclass(frozen=True, slots=True)
class GoldFact:
    """One evaluator-only gold fact for a question — `fact` text and the
    `doc_id` (derived from `Evidence.url`) it was quoted from. Callers
    build this from `mhrag.data.schema.QARecord.evidence_list`."""

    fact: str
    doc_id: str


@dataclass(frozen=True, slots=True)
class QuestionFactGroundingResult:
    """Per-question retrieval-side fact-grounding result. Every field
    here is derived ONLY from (a) the gold fact list and (b) the chunks
    that reached the generation context — never from the generated
    answer text, never from the judge grade (see module docstring's
    scope boundary)."""

    qa_id: str
    question_type: str
    n_gold_facts: int
    n_grounded_facts: int
    fact_grounded_rate: float | None  # None only if n_gold_facts == 0 (should not occur for non-null questions)
    per_fact_grounded: tuple[bool, ...]  # same order as the gold facts passed in
    gold_fact_doc_ids: tuple[str, ...]


def compute_question_fact_grounding(
    qa_id: str,
    question_type: str,
    gold_facts: list[GoldFact],
    context_chunk_texts: list[str],
) -> QuestionFactGroundingResult:
    """Pure function: no I/O, no model call, no randomness — same inputs
    always produce the same output (see
    tests/test_fact_grounding.py::test_deterministic_repeatability)."""
    per_fact = tuple(fact_grounded(gf.fact, context_chunk_texts) for gf in gold_facts)
    n_gold = len(gold_facts)
    n_grounded = sum(per_fact)
    rate = (n_grounded / n_gold) if n_gold > 0 else None
    return QuestionFactGroundingResult(
        qa_id=qa_id,
        question_type=question_type,
        n_gold_facts=n_gold,
        n_grounded_facts=n_grounded,
        fact_grounded_rate=rate,
        per_fact_grounded=per_fact,
        gold_fact_doc_ids=tuple(gf.doc_id for gf in gold_facts),
    )
