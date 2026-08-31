"""RUNTIME learned two-stage router — Phase 8A.2.

Same sequential control flow as Phase 8A.1's
`mhrag.routing.sequential_router` (Hybrid RRF -> Stage 1 decision ->
Reranker only if escalating -> Stage 2 decision), but Stage 1/2 are frozen
`LinearModel`s (`mhrag.routing.learned_router`) instead of GLM Evidence
Gates — NO LLM call anywhere in this module, so routing is $0 marginal
cost and sub-millisecond decision latency on top of the already-frozen
retrieval/reranking pipeline.

`route_question_learned`'s signature has no parameter for gold answer,
evidence_list, question_type, oracle route, or Complete-Evidence result —
only `question` plus retrieval infrastructure and the two frozen
`LinearModel`s — so there is no channel through which evaluator-only data
could reach a routing decision (see tests/test_routing_no_gold_leakage.py).
This module never imports `mhrag.routing.oracle`,
`mhrag.routing.learned_router_training`, `mhrag.routing.gate_analysis`,
`mhrag.routing.tune_thresholds`, or `mhrag.routing.heuristic`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.rerank import RERANK_CANDIDATE_DEPTH, Reranker, rerank_results
from mhrag.retrieval.rrf import RRF_K, rrf_fuse
from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.features import RouterFeatures, extract_query_features, extract_retrieval_signals
from mhrag.routing.learned_features import stage1_feature_vector, stage2_feature_vector
from mhrag.routing.learned_router import LinearModel, predict_sufficient
from mhrag.routing.rerank_features import RerankSignals, extract_rerank_signals

TOP_K = 5


@dataclass(frozen=True, slots=True)
class LearnedRouteResult:
    route: str  # "SIMPLE" | "MEDIUM" | "COMPLEX"
    stage1_probability: float
    stage2_probability: float | None  # None if Stage 2 never ran
    hybrid_top5: tuple[RetrievalResult, ...]
    reranked_top5: tuple[RetrievalResult, ...] | None
    retrieval_latency_ms: float
    reranking_latency_ms: float
    decision_latency_ms: float  # pure-arithmetic model inference time
    total_latency_ms: float


def route_question_learned(
    question: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    reranker: Reranker,
    stage1_model: LinearModel,
    stage2_model: LinearModel,
    dense_top_k: int = RERANK_CANDIDATE_DEPTH,
    bm25_top_k: int = RERANK_CANDIDATE_DEPTH,
    fused_candidates: list[RetrievalResult] | None = None,
    router_features: RouterFeatures | None = None,
) -> LearnedRouteResult:
    """`fused_candidates` (the depth-`RERANK_CANDIDATE_DEPTH` fused Hybrid
    RRF pool) and `router_features` (Stage 1's query+retrieval features),
    if BOTH given, are used instead of running the live retrieval pipeline
    — lets tests exercise this function's control flow entirely offline
    with a fake `reranker.score()` (see
    tests/test_routing_learned_sequential_router.py). Production callers
    leave both `None` so the real frozen pipeline
    (`dense_search`+`bm25_search`+`rrf_fuse`, matching
    `mhrag.routing.features.compute_router_features`'s own composition)
    runs for real, computing genuine dense/BM25 agreement signals rather
    than degenerate ones.
    """
    t_start = time.monotonic()

    t0 = time.monotonic()
    if fused_candidates is None:
        dense_results = dense_search(question, qdrant_client, collection_name, embedding_model, top_k=dense_top_k)
        bm25_results = bm25_search(question, qdrant_client, collection_name, bm25_model, top_k=bm25_top_k)
        fused_candidates = rrf_fuse(dense_results, bm25_results, k=RRF_K, final_top_k=RERANK_CANDIDATE_DEPTH)
        if router_features is None:
            router_features = RouterFeatures(
                query=extract_query_features(question),
                retrieval=extract_retrieval_signals(dense_results, bm25_results, fused_candidates[:10]),
            )
    elif router_features is None:
        # fused_candidates was injected but router_features was not (test-only path) — degrade
        # gracefully rather than requiring a live call just to fill in the gap.
        router_features = RouterFeatures(
            query=extract_query_features(question),
            retrieval=extract_retrieval_signals(fused_candidates[:10], fused_candidates[:10], fused_candidates[:10]),
        )
    retrieval_latency_ms = (time.monotonic() - t0) * 1000

    hybrid_top5 = fused_candidates[:TOP_K]

    t_decision = time.monotonic()
    stage1_sufficient, stage1_probability = predict_sufficient(
        stage1_model, stage1_feature_vector(router_features)
    )
    decision_latency_ms = (time.monotonic() - t_decision) * 1000

    if stage1_sufficient:
        return LearnedRouteResult(
            route="SIMPLE", stage1_probability=stage1_probability, stage2_probability=None,
            hybrid_top5=tuple(hybrid_top5), reranked_top5=None,
            retrieval_latency_ms=retrieval_latency_ms, reranking_latency_ms=0.0,
            decision_latency_ms=decision_latency_ms, total_latency_ms=(time.monotonic() - t_start) * 1000,
        )

    t1 = time.monotonic()
    reranked_top5 = rerank_results(question, fused_candidates, reranker, top_k=TOP_K)
    reranking_latency_ms = (time.monotonic() - t1) * 1000

    rerank_signals: RerankSignals = extract_rerank_signals(fused_candidates, reranked_top5)

    t_decision2 = time.monotonic()
    stage2_sufficient, stage2_probability = predict_sufficient(
        stage2_model, stage2_feature_vector(router_features, rerank_signals)
    )
    decision_latency_ms += (time.monotonic() - t_decision2) * 1000

    route = "MEDIUM" if stage2_sufficient else "COMPLEX"
    return LearnedRouteResult(
        route=route, stage1_probability=stage1_probability, stage2_probability=stage2_probability,
        hybrid_top5=tuple(hybrid_top5), reranked_top5=tuple(reranked_top5),
        retrieval_latency_ms=retrieval_latency_ms, reranking_latency_ms=reranking_latency_ms,
        decision_latency_ms=decision_latency_ms, total_latency_ms=(time.monotonic() - t_start) * 1000,
    )
