"""RUNTIME Adaptive RAG pipeline — Phase 8B.

    Question
      -> Hybrid (Dense + BM25 + deterministic RRF, k=60)   [retrieval call 1]
      -> Stage 1 (frozen Phase 8A.2 LinearModel, tau1=0.63)
           SIMPLE     -> Hybrid top-5 -> Qwen answer
           escalate
             -> Reranker (BAAI/bge-reranker-base)          [reranker call 1]
             -> Stage 2 (frozen Phase 8A.2 LinearModel, tau2=0.70)
                  MEDIUM  -> reranked top-5 -> Qwen answer
                  COMPLEX -> bounded Agentic retrieval (mhrag.agent.loop,
                             max_hops=3, max_context_tokens=4500) -> Qwen
                             answer

Routing is $0 marginal cost, sub-millisecond (`mhrag.routing.learned_router`
— pure arithmetic, no LLM call; see Phase 8A.2). The agent controller
(`zai.glm-4.7-flash`) and the final-answer model (`qwen.qwen3-next-80b-
a3b-instruct`) are called ONLY where the execution diagram above actually
calls them — the controller only for COMPLEX questions, the final-answer
model exactly once for every question, with the SAME prompt/model/pricing
regardless of route (`mhrag.generation.answer.generate_answer`, unchanged
from Phase 6/7 — this module never forks that function per route).

CRITICAL (COMPLEX reuse): the Hybrid+Reranker result already computed to
make the Stage 2 decision is reused as the agentic loop's hop 1 — the
initial retrieval/reranking is never repeated. `_reuse_first_hop_runner`
wraps the loop's own `hop_runner` injection point (`mhrag.agent.loop.
run_agentic_retrieval`) so hop 1 returns the already-computed, already-
timed `reranked_top5` on its first call with no new retrieval/reranking
call — it still counts as `num_retrieval_calls`/`num_reranker_calls` = 1
(the loop's own `hops` list includes it), it just does not pay for it a
second time. Every hop after the first calls the REAL frozen retrieval
pipeline (`mhrag.agent.loop.build_default_hop_runner`), unmodified.

`run_adaptive_pipeline`'s signature has no parameter for gold answer,
evidence_list, question_type, oracle route, or Complete-Evidence result —
only `question` plus retrieval/router/LLM infrastructure — so there is no
channel through which evaluator-only data could reach a routing or
generation decision (see tests/test_adaptive_pipeline_no_gold_leakage.py).
This module never imports `mhrag.routing.oracle`,
`mhrag.routing.learned_router_training`, `mhrag.routing.gate_analysis`,
`mhrag.routing.tune_thresholds`, or `mhrag.routing.heuristic`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mhrag.agent.loop import AgenticConfig, AgenticTrace, HopRunner, build_default_hop_runner, run_agentic_retrieval
from mhrag.generation.answer import GenerationResult, generate_answer
from mhrag.generation.context import approximate_token_count
from mhrag.generation.mantle_client import MantleClient
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
from mhrag.routing.rerank_features import extract_rerank_signals

TOP_K = 5


@dataclass(frozen=True, slots=True)
class AdaptiveTrace:
    question: str
    route: str  # "SIMPLE" | "MEDIUM" | "COMPLEX"
    stage1_probability: float
    stage2_probability: float | None  # None if Stage 1 was already sufficient (SIMPLE)
    stop_reason: str  # "route_simple" | "route_medium" | one of mhrag.agent.loop.STOP_REASONS (COMPLEX)

    num_retrieval_calls: int
    num_reranker_calls: int
    num_agent_hops: int  # 0 unless route == COMPLEX (equals num_retrieval_calls there)
    num_controller_calls: int  # GLM controller calls — 0 unless route == COMPLEX
    num_generation_calls: int  # always 1 — exactly one Qwen call per question, every route

    glm_input_tokens: int
    glm_output_tokens: int
    glm_cost_usd: float | None
    qwen_input_tokens: int
    qwen_output_tokens: int
    qwen_cost_usd: float | None
    total_cost_usd: float | None

    retrieval_latency_ms: float
    reranking_latency_ms: float
    controller_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    answer: str
    evidence_pool: tuple[RetrievalResult, ...]  # every chunk available at the final-answer decision
    chunks_used_for_generation: tuple[RetrievalResult, ...]  # what was actually sent to Qwen (post budget)
    unique_docs_used: int

    agentic_trace: AgenticTrace | None  # full bounded-agentic sub-trace, COMPLEX only, else None


def _reuse_first_hop_runner(
    cached_results: list[RetrievalResult],
    cached_retrieval_latency_ms: float,
    cached_reranking_latency_ms: float,
    fallback_hop_runner: HopRunner,
) -> HopRunner:
    """Wraps `fallback_hop_runner` (the real frozen pipeline) so the FIRST
    call returns the already-computed `cached_results` with the latencies
    already paid for them — no new retrieval/reranking call happens — and
    every call after that goes to `fallback_hop_runner` for real. This is
    the mechanism that makes COMPLEX's hop 1 free (no repeated retrieval)
    while still counting as one retrieval/reranker call, and still letting
    the agent ask genuinely new follow-up queries from hop 2 onward."""
    state = {"used": False}

    def run(query: str) -> tuple[list[RetrievalResult], float, float]:
        if not state["used"]:
            state["used"] = True
            return list(cached_results), cached_retrieval_latency_ms, cached_reranking_latency_ms
        return fallback_hop_runner(query)

    return run


def run_adaptive_pipeline(
    question: str,
    qdrant_client,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    reranker: Reranker,
    stage1_model: LinearModel,
    stage2_model: LinearModel,
    controller_client: MantleClient,
    generation_client: MantleClient,
    agentic_config: AgenticConfig = AgenticConfig(),
    dense_top_k: int = RERANK_CANDIDATE_DEPTH,
    bm25_top_k: int = RERANK_CANDIDATE_DEPTH,
    fused_candidates: list[RetrievalResult] | None = None,
    router_features: RouterFeatures | None = None,
    agent_followup_hop_runner: HopRunner | None = None,
) -> AdaptiveTrace:
    """Route one question through the Adaptive pipeline and generate a
    final answer. `question` is the ONLY thing from ground truth ever used.

    `fused_candidates`/`router_features`, if BOTH given, replace the live
    Hybrid retrieval call — lets tests exercise Stage 1/2 routing and the
    SIMPLE/MEDIUM generation paths entirely offline (see
    tests/test_adaptive_pipeline.py). `agent_followup_hop_runner`, if
    given, replaces the REAL frozen pipeline used for COMPLEX's hop 2+
    (hop 1 always reuses the already-computed Hybrid+Reranker result,
    live or injected) — production callers leave all three `None`.
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
        # fused_candidates injected but router_features was not (test-only path).
        router_features = RouterFeatures(
            query=extract_query_features(question),
            retrieval=extract_retrieval_signals(fused_candidates[:10], fused_candidates[:10], fused_candidates[:10]),
        )
    retrieval_latency_ms = (time.monotonic() - t0) * 1000

    hybrid_top5 = list(fused_candidates[:TOP_K])

    stage1_sufficient, stage1_probability = predict_sufficient(stage1_model, stage1_feature_vector(router_features))

    if stage1_sufficient:
        generation = generate_answer(
            question, hybrid_top5, generation_client, approximate_token_count,
            top_k=TOP_K, max_context_tokens=agentic_config.max_context_tokens,
            input_price_per_million=agentic_config.qwen_input_price_per_million,
            output_price_per_million=agentic_config.qwen_output_price_per_million,
            prompt_version=agentic_config.generation_prompt_version,
        )
        return _trace_from_direct_generation(
            question=question, route="SIMPLE", stop_reason="route_simple",
            stage1_probability=stage1_probability, stage2_probability=None,
            num_retrieval_calls=1, num_reranker_calls=0,
            retrieval_latency_ms=retrieval_latency_ms, reranking_latency_ms=0.0,
            generation=generation, evidence_pool=hybrid_top5, t_start=t_start,
        )

    t1 = time.monotonic()
    reranked_top5 = rerank_results(question, fused_candidates, reranker, top_k=TOP_K)
    reranking_latency_ms = (time.monotonic() - t1) * 1000

    rerank_signals = extract_rerank_signals(fused_candidates, reranked_top5)
    stage2_sufficient, stage2_probability = predict_sufficient(
        stage2_model, stage2_feature_vector(router_features, rerank_signals)
    )

    if stage2_sufficient:
        generation = generate_answer(
            question, reranked_top5, generation_client, approximate_token_count,
            top_k=TOP_K, max_context_tokens=agentic_config.max_context_tokens,
            input_price_per_million=agentic_config.qwen_input_price_per_million,
            output_price_per_million=agentic_config.qwen_output_price_per_million,
            prompt_version=agentic_config.generation_prompt_version,
        )
        return _trace_from_direct_generation(
            question=question, route="MEDIUM", stop_reason="route_medium",
            stage1_probability=stage1_probability, stage2_probability=stage2_probability,
            num_retrieval_calls=1, num_reranker_calls=1,
            retrieval_latency_ms=retrieval_latency_ms, reranking_latency_ms=reranking_latency_ms,
            generation=generation, evidence_pool=reranked_top5, t_start=t_start,
        )

    # --- COMPLEX: bounded agentic retrieval, reusing the Hybrid+Reranker result above as hop 1 ---
    fallback_runner = agent_followup_hop_runner or build_default_hop_runner(
        qdrant_client, collection_name, embedding_model, bm25_model, reranker, agentic_config.hop_top_k
    )
    hop_runner = _reuse_first_hop_runner(
        cached_results=reranked_top5,
        cached_retrieval_latency_ms=retrieval_latency_ms,
        cached_reranking_latency_ms=reranking_latency_ms,
        fallback_hop_runner=fallback_runner,
    )
    agentic_trace = run_agentic_retrieval(
        question, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
        controller_client, generation_client, config=agentic_config, hop_runner=hop_runner,
    )

    return AdaptiveTrace(
        question=question, route="COMPLEX",
        stage1_probability=stage1_probability, stage2_probability=stage2_probability,
        stop_reason=agentic_trace.stop_reason,
        num_retrieval_calls=agentic_trace.num_retrieval_calls,
        num_reranker_calls=agentic_trace.num_retrieval_calls,  # every hop pairs one retrieval with one rerank
        num_agent_hops=agentic_trace.num_retrieval_calls,
        num_controller_calls=agentic_trace.num_controller_calls,
        num_generation_calls=agentic_trace.num_generation_calls,
        glm_input_tokens=agentic_trace.cost.glm_input_tokens,
        glm_output_tokens=agentic_trace.cost.glm_output_tokens,
        glm_cost_usd=agentic_trace.cost.glm_cost_usd,
        qwen_input_tokens=agentic_trace.cost.qwen_input_tokens,
        qwen_output_tokens=agentic_trace.cost.qwen_output_tokens,
        qwen_cost_usd=agentic_trace.cost.qwen_cost_usd,
        total_cost_usd=agentic_trace.cost.total_cost_usd,
        retrieval_latency_ms=agentic_trace.retrieval_latency_ms,
        reranking_latency_ms=agentic_trace.reranking_latency_ms,
        controller_latency_ms=agentic_trace.controller_latency_ms,
        generation_latency_ms=agentic_trace.generation_latency_ms,
        total_latency_ms=(time.monotonic() - t_start) * 1000,
        answer=agentic_trace.final_generation.answer,
        evidence_pool=agentic_trace.evidence_pool,
        chunks_used_for_generation=agentic_trace.final_generation.context.chunks_included,
        unique_docs_used=agentic_trace.unique_documents_retrieved,
        agentic_trace=agentic_trace,
    )


def _trace_from_direct_generation(
    question: str,
    route: str,
    stop_reason: str,
    stage1_probability: float,
    stage2_probability: float | None,
    num_retrieval_calls: int,
    num_reranker_calls: int,
    retrieval_latency_ms: float,
    reranking_latency_ms: float,
    generation: GenerationResult,
    evidence_pool: list[RetrievalResult],
    t_start: float,
) -> AdaptiveTrace:
    """Shared trace assembly for SIMPLE/MEDIUM — both skip the agentic loop
    entirely (no controller call), going straight from the frozen router's
    decision to the SAME Qwen final-generation call COMPLEX and
    Agentic Multi-Hop RAG use."""
    usage = generation.mantle_response.usage
    return AdaptiveTrace(
        question=question, route=route,
        stage1_probability=stage1_probability, stage2_probability=stage2_probability,
        stop_reason=stop_reason,
        num_retrieval_calls=num_retrieval_calls, num_reranker_calls=num_reranker_calls,
        num_agent_hops=0, num_controller_calls=0, num_generation_calls=1,
        glm_input_tokens=0, glm_output_tokens=0, glm_cost_usd=None,
        qwen_input_tokens=usage.input_tokens or 0, qwen_output_tokens=usage.output_tokens or 0,
        qwen_cost_usd=generation.cost.total_cost_usd, total_cost_usd=generation.cost.total_cost_usd,
        retrieval_latency_ms=retrieval_latency_ms, reranking_latency_ms=reranking_latency_ms,
        controller_latency_ms=0.0, generation_latency_ms=generation.mantle_response.llm_latency_ms,
        total_latency_ms=(time.monotonic() - t_start) * 1000,
        answer=generation.answer, evidence_pool=tuple(evidence_pool),
        chunks_used_for_generation=generation.context.chunks_included,
        unique_docs_used=len({r.doc_id for r in evidence_pool}),
        agentic_trace=None,
    )
