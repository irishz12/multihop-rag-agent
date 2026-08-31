"""Oracle route labels — EVALUATOR-ONLY, never seen by the runtime router.

Computed entirely from the frozen Phase 4/4.1/5 retrieval evaluation
artifact (`results/retrieval_eval_development.json`'s `per_query` list,
265 non-null DEVELOPMENT questions) — no new retrieval, no re-run of the
Hybrid/Hybrid+Reranker pipeline, no live Mantle calls. Complete-Evidence@5
is computed via the existing, unmodified
`mhrag.eval.metrics.complete_evidence_at_k(..., k=5)`.

Definition (K=5 — the answer-context depth this project actually uses):

    SIMPLE  = Hybrid RRF alone satisfies Complete-Evidence@5
              (all required gold documents are in Hybrid's top-5 unique docs)
    MEDIUM  = Hybrid RRF fails Complete-Evidence@5, but Hybrid+Reranker
              satisfies Complete-Evidence@5
    COMPLEX = neither Hybrid nor Hybrid+Reranker satisfies Complete-Evidence@5

This is the MINIMUM retrieval complexity needed for each question — not a
prediction, a measurement made with full hindsight of the gold evidence.
These labels exist ONLY for router calibration/evaluation
(`mhrag.routing.split`, `mhrag.routing.tune_thresholds`,
`mhrag.routing.metrics`) — nothing in `mhrag.routing.features`,
`mhrag.routing.heuristic`, `mhrag.routing.glm_router`, or
`mhrag.routing.router` (the runtime path) imports this module; see
tests/test_routing_no_gold_leakage.py for the structural guard.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.eval.metrics import complete_evidence_at_k

ORACLE_K = 5
ROUTE_LABELS = ("SIMPLE", "MEDIUM", "COMPLEX")


@dataclass(frozen=True, slots=True)
class OracleLabel:
    qa_id: str
    question_type: str
    hop_count: int
    route: str  # one of ROUTE_LABELS
    hybrid_complete_evidence_at_5: bool
    hybrid_reranker_complete_evidence_at_5: bool


def compute_oracle_label(per_query_record: dict) -> OracleLabel:
    """`per_query_record` is one entry from
    `results/retrieval_eval_development.json`'s `per_query` list — already
    excludes null_query (that artifact only ever scores the 265 non-null
    development questions; see its `counts` field)."""
    gold = frozenset(per_query_record["gold_doc_ids"])
    if not gold:
        raise ValueError(
            f"per_query record {per_query_record.get('qa_id')!r} has an empty gold_doc_ids "
            "set — null_query records must never reach oracle labeling"
        )

    hybrid_top10 = per_query_record["methods"]["hybrid"]["unique_doc_ids_top10"]
    reranker_top10 = per_query_record["methods"]["hybrid_reranker"]["unique_doc_ids_top10"]

    hybrid_ce5 = bool(complete_evidence_at_k(hybrid_top10, gold, ORACLE_K))
    reranker_ce5 = bool(complete_evidence_at_k(reranker_top10, gold, ORACLE_K))

    if hybrid_ce5:
        route = "SIMPLE"
    elif reranker_ce5:
        route = "MEDIUM"
    else:
        route = "COMPLEX"

    return OracleLabel(
        qa_id=per_query_record["qa_id"],
        question_type=per_query_record["question_type"],
        hop_count=per_query_record["hop_count"],
        route=route,
        hybrid_complete_evidence_at_5=hybrid_ce5,
        hybrid_reranker_complete_evidence_at_5=reranker_ce5,
    )


def compute_oracle_labels(retrieval_eval_artifact: dict) -> list[OracleLabel]:
    """`retrieval_eval_artifact` is the full parsed
    `results/retrieval_eval_development.json`. Returns one `OracleLabel`
    per non-null development question (265), in the artifact's existing
    order."""
    return [compute_oracle_label(pq) for pq in retrieval_eval_artifact["per_query"]]


def label_distribution(labels: list[OracleLabel]) -> dict[str, int]:
    dist = {route: 0 for route in ROUTE_LABELS}
    for label in labels:
        dist[label.route] += 1
    return dist
