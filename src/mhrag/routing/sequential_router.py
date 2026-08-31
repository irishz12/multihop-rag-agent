"""RUNTIME evidence-aware SEQUENTIAL router — Phase 8A.1.

    question -> Hybrid RRF (frozen, k=60, dense_top_k=bm25_top_k=20,
                fused pool depth = RERANK_CANDIDATE_DEPTH=20)
             -> take top-5 -> Evidence Gate 1
                  sufficient? -> SIMPLE, done (Reranker never runs)
                  not sufficient?
                    -> Reranker (frozen bge-reranker-base) over the SAME
                       already-fetched 20-candidate pool -> top-5
                    -> Evidence Gate 2
                         sufficient? -> MEDIUM
                         not sufficient? -> COMPLEX

No heuristic shortcut this phase (Phase 8A.1 spec: "disable the heuristic
shortcut entirely... do not optimize router cost before establishing
router correctness") — EVERY question calls Gate 1 for real; Gate 2 is
skipped only when Gate 1 already returned sufficient=True.

`route_question_sequential`'s signature has no parameter for gold answer,
evidence_list, question_type, oracle route, or Complete-Evidence result —
only `question` plus retrieval/LLM infrastructure — so there is no channel
through which evaluator-only data could reach a routing decision (see
tests/test_routing_no_gold_leakage.py). This module never imports
`mhrag.routing.oracle`, `mhrag.routing.gate_analysis`, or
`mhrag.routing.tune_thresholds`, and (per the "disable the heuristic
shortcut" instruction) never imports `mhrag.routing.heuristic` either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from mhrag.generation.cost import CostEstimate, estimate_cost_usd
from mhrag.generation.mantle_client import MantleClient, MantleUsage
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.rerank import RERANK_CANDIDATE_DEPTH, Reranker, rerank_results
from mhrag.retrieval.rrf import RRF_K, deterministic_hybrid_search
from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.evidence_gate import GateResult, call_evidence_gate
from mhrag.routing.evidence_gate_prompts import GateChunkInput

TOP_K = 5  # answer-context depth — matches the oracle's CE@5 definition exactly


def _to_gate_chunks(results: list[RetrievalResult]) -> list[GateChunkInput]:
    return [
        GateChunkInput(chunk_id=r.chunk_id, title=r.title, text=r.text, rank=r.rank, score=r.score)
        for r in results
    ]


@dataclass(frozen=True, slots=True)
class SequentialRouteResult:
    route: str  # "SIMPLE" | "MEDIUM" | "COMPLEX"
    hybrid_top5: tuple[RetrievalResult, ...]
    reranked_top5: tuple[RetrievalResult, ...] | None  # None if Gate 1 was sufficient (Reranker never ran)
    gate1_result: GateResult
    gate2_result: GateResult | None
    num_glm_calls: int
    glm_input_tokens: int
    glm_output_tokens: int
    glm_cost: CostEstimate | None
    retrieval_latency_ms: float
    reranking_latency_ms: float
    gate_latency_ms: float
    total_latency_ms: float


def route_question_sequential(
    question: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    reranker: Reranker,
    glm_client: MantleClient,
    dense_top_k: int = RERANK_CANDIDATE_DEPTH,
    bm25_top_k: int = RERANK_CANDIDATE_DEPTH,
    glm_input_price_per_million: float | None = None,
    glm_output_price_per_million: float | None = None,
    fused_candidates: list[RetrievalResult] | None = None,
) -> SequentialRouteResult:
    """Route one question through the evidence-aware sequential gate.

    `fused_candidates`, if given (the depth-`RERANK_CANDIDATE_DEPTH` fused
    Hybrid RRF pool), is used instead of running `deterministic_hybrid_
    search` live — lets tests exercise this function's control flow
    entirely offline (a fake `reranker.score()` plus a fake `glm_client`
    are enough — see tests/test_routing_sequential_router.py) and lets
    evaluation scripts reuse an already-fetched pool. Production callers
    leave it `None` so the real frozen pipeline runs.
    """
    t_start = time.monotonic()

    t0 = time.monotonic()
    if fused_candidates is None:
        fused_candidates = deterministic_hybrid_search(
            question, qdrant_client, collection_name, embedding_model, bm25_model,
            dense_top_k=dense_top_k, bm25_top_k=bm25_top_k, final_top_k=RERANK_CANDIDATE_DEPTH, k=RRF_K,
        )
    retrieval_latency_ms = (time.monotonic() - t0) * 1000

    hybrid_top5 = fused_candidates[:TOP_K]

    t_gate = time.monotonic()
    gate1_result = call_evidence_gate(glm_client, question, _to_gate_chunks(hybrid_top5))
    gate_latency_ms = (time.monotonic() - t_gate) * 1000

    if gate1_result.decision.sufficient:
        route = "SIMPLE"
        reranked_top5 = None
        reranking_latency_ms = 0.0
        gate2_result = None
    else:
        t1 = time.monotonic()
        reranked_top5 = rerank_results(question, fused_candidates, reranker, top_k=TOP_K)
        reranking_latency_ms = (time.monotonic() - t1) * 1000

        t_gate2 = time.monotonic()
        gate2_result = call_evidence_gate(glm_client, question, _to_gate_chunks(reranked_top5))
        gate_latency_ms += (time.monotonic() - t_gate2) * 1000

        route = "MEDIUM" if gate2_result.decision.sufficient else "COMPLEX"

    gate_results = [gate1_result] + ([gate2_result] if gate2_result is not None else [])
    num_glm_calls = len(gate_results)
    glm_input_tokens = sum(g.mantle_response.usage.input_tokens or 0 for g in gate_results)
    glm_output_tokens = sum(g.mantle_response.usage.output_tokens or 0 for g in gate_results)

    glm_cost = None
    if glm_input_price_per_million is not None and glm_output_price_per_million is not None:
        glm_cost = estimate_cost_usd(
            MantleUsage(glm_input_tokens, glm_output_tokens, None),
            glm_input_price_per_million, glm_output_price_per_million,
        )

    return SequentialRouteResult(
        route=route,
        hybrid_top5=tuple(hybrid_top5),
        reranked_top5=tuple(reranked_top5) if reranked_top5 is not None else None,
        gate1_result=gate1_result,
        gate2_result=gate2_result,
        num_glm_calls=num_glm_calls,
        glm_input_tokens=glm_input_tokens,
        glm_output_tokens=glm_output_tokens,
        glm_cost=glm_cost,
        retrieval_latency_ms=retrieval_latency_ms,
        reranking_latency_ms=reranking_latency_ms,
        gate_latency_ms=gate_latency_ms,
        total_latency_ms=(time.monotonic() - t_start) * 1000,
    )
