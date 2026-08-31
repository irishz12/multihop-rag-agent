"""Budget sweep: run the FROZEN Phase 7 agentic loop across a fixed
question set, once per candidate `max_context_tokens` value, changing
NOTHING else — every swept `AgenticConfig` differs from the base config
in exactly that one field (see `build_swept_configs`, and
tests/test_calibration_sweep.py::test_swept_configs_differ_only_in_budget).

Gold-evidence evaluation (`evaluate_against_gold`) is the ONLY function in
this module that touches `record.evidence_list` / `record.answer` /
`record.question_type` — and it is only ever called AFTER
`run_agentic_retrieval` has already returned a finished `AgenticTrace`.
Nothing here feeds gold data back into the agent; `run_agentic_retrieval`
itself is called with only `record.query`, exactly as in Phase 7.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mhrag.agent.loop import AgenticConfig, AgenticTrace, HopRunner, run_agentic_retrieval
from mhrag.data.schema import QARecord
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.metrics import collapse_to_unique_documents, complete_evidence_at_k, recall_at_k


def build_swept_configs(base_config: AgenticConfig, token_budgets: list[int]) -> dict[int, AgenticConfig]:
    """One AgenticConfig per candidate budget, each identical to
    `base_config` except `max_context_tokens`."""
    return {budget: dataclasses.replace(base_config, max_context_tokens=budget) for budget in token_budgets}


def new_unique_docs_per_hop(trace: AgenticTrace) -> tuple[int, ...]:
    """For each hop, how many documents NOT already in the pool from an
    earlier hop were introduced by that hop's genuinely-new chunks (chunk-
    level duplicates, already excluded from `new_chunk_ids` by
    `mhrag.agent.evidence.merge_evidence`, don't count here either way —
    this additionally catches a *new* chunk that belongs to an
    *already-seen* document, which is "new evidence" at the chunk level
    but not at the document level)."""
    doc_by_chunk = {c.chunk_id: c.doc_id for c in trace.evidence_pool}
    seen_docs: set[str] = set()
    gains: list[int] = []
    for hop in trace.hops:
        new_docs_this_hop = {doc_by_chunk[cid] for cid in hop.new_chunk_ids}
        gains.append(len(new_docs_this_hop - seen_docs))
        seen_docs |= new_docs_this_hop
    return tuple(gains)


@dataclass(frozen=True, slots=True)
class EvidenceEvaluation:
    gold_doc_ids: frozenset[str]
    recall: float
    complete_evidence: bool
    new_unique_docs_per_hop: tuple[int, ...]


def evaluate_against_gold(record: QARecord, trace: AgenticTrace) -> EvidenceEvaluation:
    """Evaluator-only — call strictly AFTER `run_agentic_retrieval`
    returns. Uses `record.evidence_list`; never touches `record.answer` or
    `record.question_type`, and never mutates or returns anything that
    feeds back into the agent."""
    gold = gold_doc_ids(record)
    unique_doc_ids = collapse_to_unique_documents(list(trace.evidence_pool))
    recall = recall_at_k(unique_doc_ids, gold, k=len(unique_doc_ids))
    complete = bool(complete_evidence_at_k(unique_doc_ids, gold, k=len(unique_doc_ids)))
    return EvidenceEvaluation(
        gold_doc_ids=gold,
        recall=recall,
        complete_evidence=complete,
        new_unique_docs_per_hop=new_unique_docs_per_hop(trace),
    )


@dataclass(frozen=True, slots=True)
class CalibrationQueryResult:
    qa_id: str
    question_type: str
    hop_count: int
    trace: AgenticTrace
    evaluation: EvidenceEvaluation


def run_calibration_query(
    record: QARecord,
    qa_id: str,
    hop_count: int,
    qdrant_client,
    collection_name: str,
    embedding_model,
    bm25_model,
    reranker,
    controller_client,
    generation_client,
    config: AgenticConfig,
    hop_runner: HopRunner | None = None,
) -> CalibrationQueryResult:
    """Run one calibration question through the unmodified agentic loop
    (only `record.query` is passed to it — no ground truth), then evaluate
    the resulting trace against gold evidence.

    `hop_runner` is forwarded to `run_agentic_retrieval` unchanged — pass a
    fake to test this function offline (see tests/test_calibration_sweep.py);
    production callers leave it `None` for the real retrieval pipeline.
    """
    trace = run_agentic_retrieval(
        record.query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
        controller_client, generation_client, config=config, hop_runner=hop_runner,
    )
    evaluation = evaluate_against_gold(record, trace)
    return CalibrationQueryResult(
        qa_id=qa_id, question_type=record.question_type, hop_count=hop_count,
        trace=trace, evaluation=evaluation,
    )
