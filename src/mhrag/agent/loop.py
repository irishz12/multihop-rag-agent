"""Agentic Multi-Hop RAG: the bounded agent loop.

    Original Question
       -> Hybrid + RRF + Reranker           (hop 1 = initial retrieval)
       -> Agent Controller (GLM 4.7 Flash)
       -> sufficient?
            YES -> final answer (Qwen, mhrag.generation.answer — unchanged from Phase 6)
            NO  -> focused follow-up query -> Hybrid + RRF + Reranker (hop N)
                   -> merge/dedupe evidence -> Agent Controller -> repeat

HARD LIMITS, enforced in code (never only in a prompt):
  - `config.max_hops` retrieval calls total (default 3: hop 1 + up to 2
    follow-ups) — the loop is a `for hop in range(1, max_hops + 1)`, so it
    is structurally impossible to exceed this regardless of what the
    controller returns.
  - `config.max_context_tokens` — checked against the merged evidence pool
    after every hop; exceeding it stops the loop (stop_reason=
    "token_budget") before any further retrieval is attempted.
  - `config.timeout_seconds` — wall-clock, checked at the start of every
    hop; exceeding it stops the loop (stop_reason="timeout").
  - duplicate follow-up queries are rejected before retrieval
    (stop_reason="duplicate_query") — case/whitespace-insensitive.

Retrieval, every hop, is the FROZEN pipeline from Phases 2-5, called
directly and unmodified: `dense_search` + `bm25_search` -> `rrf_fuse` (k=60)
-> `rerank_results` (bge-reranker-base). Nothing in this module changes
chunking, embeddings, BM25, RRF, the reranker, or candidate depths — it
only decides HOW MANY TIMES to call that pipeline and how to merge what
comes back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from mhrag.agent.controller import ControllerResult, call_controller
from mhrag.agent.evidence import merge_evidence
from mhrag.generation.answer import GenerationResult, generate_answer
from mhrag.generation.context import approximate_token_count
from mhrag.generation.cost import estimate_cost_usd
from mhrag.generation.mantle_client import MantleClient, MantleUsage
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.rerank import RERANK_CANDIDATE_DEPTH, Reranker, rerank_results
from mhrag.retrieval.rrf import RRF_K, rrf_fuse
from mhrag.retrieval.schema import RetrievalResult

STOP_REASONS = frozenset(
    {
        "evidence_sufficient",
        "max_hops",
        "token_budget",
        "timeout",
        "duplicate_query",
        "controller_failure",
    }
)


@dataclass(frozen=True)
class AgenticConfig:
    max_hops: int = 3
    hop_top_k: int = 5  # chunks returned per hop after reranking (unchanged production depth)
    max_evidence_chunks: int = 15  # safety ceiling on chunks considered for final context
    max_context_tokens: int = 3000
    timeout_seconds: float = 30.0
    controller_prompt_version: str = "v1"
    generation_prompt_version: str = "v1"
    glm_input_price_per_million: float = 0.08
    glm_output_price_per_million: float = 0.48
    qwen_input_price_per_million: float = 0.18
    qwen_output_price_per_million: float = 1.41


@dataclass(frozen=True, slots=True)
class HopRecord:
    hop_number: int
    query: str
    chunk_results: tuple[RetrievalResult, ...]
    new_chunk_ids: tuple[str, ...]
    duplicate_chunk_ids: tuple[str, ...]
    retrieval_latency_ms: float
    reranking_latency_ms: float


@dataclass(frozen=True, slots=True)
class AgenticCostSummary:
    glm_input_tokens: int
    glm_output_tokens: int
    glm_cost_usd: float | None
    qwen_input_tokens: int
    qwen_output_tokens: int
    qwen_cost_usd: float | None
    total_cost_usd: float | None


@dataclass(frozen=True, slots=True)
class AgenticTrace:
    question: str
    hops: tuple[HopRecord, ...]
    controller_results: tuple[ControllerResult, ...]
    final_generation: GenerationResult
    stop_reason: str
    num_retrieval_calls: int
    num_controller_calls: int
    num_generation_calls: int
    retrieval_latency_ms: float
    reranking_latency_ms: float
    controller_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    cost: AgenticCostSummary
    unique_chunks_retrieved: int
    unique_documents_retrieved: int
    duplicate_chunks_removed: int
    chunks_passed_to_final_generation: int
    chunks_dropped_for_budget: int
    evidence_pool: tuple[RetrievalResult, ...]


HopRunner = Callable[[str], tuple[list[RetrievalResult], float, float]]
"""A hop runner takes a query and returns (reranked_results,
retrieval_latency_ms, reranking_latency_ms) — the frozen retrieval
pipeline's output for one hop. Injectable so tests can exercise the loop's
control flow (hop limits, stop reasons, dedup, budget, timeout) with
scripted fake results, with no live Qdrant/embedding/reranker model
involved (see tests/test_agent_loop.py)."""


def build_default_hop_runner(
    qdrant_client,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    reranker: Reranker,
    hop_top_k: int,
) -> HopRunner:
    """Build the real, production hop runner: one call through the frozen
    Dense+BM25->RRF(k=60)->reranker pipeline, unmodified."""

    def run(query: str) -> tuple[list[RetrievalResult], float, float]:
        t0 = time.monotonic()
        dense_results = dense_search(
            query, qdrant_client, collection_name, embedding_model, top_k=RERANK_CANDIDATE_DEPTH
        )
        bm25_results = bm25_search(
            query, qdrant_client, collection_name, bm25_model, top_k=RERANK_CANDIDATE_DEPTH
        )
        fused = rrf_fuse(dense_results, bm25_results, k=RRF_K, final_top_k=RERANK_CANDIDATE_DEPTH)
        retrieval_latency_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        reranked = rerank_results(query, fused, reranker, top_k=hop_top_k)
        reranking_latency_ms = (time.monotonic() - t0) * 1000

        return reranked, retrieval_latency_ms, reranking_latency_ms

    return run


def run_agentic_retrieval(
    question: str,
    qdrant_client,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    reranker: Reranker,
    controller_client: MantleClient,
    generation_client: MantleClient,
    config: AgenticConfig = AgenticConfig(),
    hop_runner: HopRunner | None = None,
) -> AgenticTrace:
    """Run the bounded agentic retrieval loop for one question and generate
    a final answer. `question` is the ONLY thing from ground truth ever
    used — see module docstrings in mhrag.agent.controller and
    mhrag.generation.answer for the structural guarantee that gold answer/
    evidence_list/question_type never enter any prompt built here.

    `hop_runner` defaults to the real frozen retrieval pipeline (built from
    `qdrant_client`/`embedding_model`/`bm25_model`/`reranker`); pass a fake
    to test the loop's control flow in isolation (see
    tests/test_agent_loop.py).
    """
    if hop_runner is None:
        hop_runner = build_default_hop_runner(
            qdrant_client, collection_name, embedding_model, bm25_model, reranker, config.hop_top_k
        )

    t_loop_start = time.monotonic()

    evidence_pool: list[RetrievalResult] = []
    hops: list[HopRecord] = []
    controller_results: list[ControllerResult] = []
    seen_queries: set[str] = set()
    current_query = question
    stop_reason: str | None = None

    for hop_number in range(1, config.max_hops + 1):
        if time.monotonic() - t_loop_start > config.timeout_seconds:
            stop_reason = "timeout"
            break

        normalized_query = current_query.strip().lower()
        if normalized_query in seen_queries:
            stop_reason = "duplicate_query"
            break
        seen_queries.add(normalized_query)

        hop_results, retrieval_latency_ms, reranking_latency_ms = hop_runner(current_query)

        merge_result = merge_evidence(evidence_pool, hop_results)
        evidence_pool = list(merge_result.pool)

        hops.append(
            HopRecord(
                hop_number=hop_number,
                query=current_query,
                chunk_results=tuple(hop_results),
                new_chunk_ids=merge_result.new_chunk_ids,
                duplicate_chunk_ids=merge_result.duplicate_chunk_ids,
                retrieval_latency_ms=retrieval_latency_ms,
                reranking_latency_ms=reranking_latency_ms,
            )
        )

        approx_tokens = sum(approximate_token_count(r.text) for r in evidence_pool)
        if approx_tokens > config.max_context_tokens:
            stop_reason = "token_budget"
            break

        controller_result = call_controller(
            controller_client,
            question,
            evidence_pool,
            [h.query for h in hops],
            prompt_version=config.controller_prompt_version,
        )
        controller_results.append(controller_result)

        if controller_result.fallback_used:
            stop_reason = "controller_failure"
            break

        if controller_result.decision.sufficient:
            stop_reason = "evidence_sufficient"
            break

        if hop_number >= config.max_hops:
            stop_reason = "max_hops"
            break

        next_query = controller_result.decision.next_query
        if not next_query or not next_query.strip():
            stop_reason = "controller_failure"
            break
        current_query = next_query

    if stop_reason is None:  # defensive — every loop exit above sets it explicitly
        stop_reason = "max_hops"

    generation = generate_answer(
        question,
        evidence_pool[: config.max_evidence_chunks],
        generation_client,
        approximate_token_count,
        top_k=config.max_evidence_chunks,
        max_context_tokens=config.max_context_tokens,
        input_price_per_million=config.qwen_input_price_per_million,
        output_price_per_million=config.qwen_output_price_per_million,
        prompt_version=config.generation_prompt_version,
    )

    glm_input_tokens = sum(c.mantle_response.usage.input_tokens or 0 for c in controller_results)
    glm_output_tokens = sum(c.mantle_response.usage.output_tokens or 0 for c in controller_results)
    glm_cost = estimate_cost_usd(
        MantleUsage(glm_input_tokens, glm_output_tokens, None),
        config.glm_input_price_per_million,
        config.glm_output_price_per_million,
    )
    qwen_cost = generation.cost
    total_cost = (
        glm_cost.total_cost_usd + qwen_cost.total_cost_usd
        if glm_cost.total_cost_usd is not None and qwen_cost.total_cost_usd is not None
        else None
    )
    cost_summary = AgenticCostSummary(
        glm_input_tokens=glm_input_tokens,
        glm_output_tokens=glm_output_tokens,
        glm_cost_usd=glm_cost.total_cost_usd,
        qwen_input_tokens=generation.mantle_response.usage.input_tokens or 0,
        qwen_output_tokens=generation.mantle_response.usage.output_tokens or 0,
        qwen_cost_usd=qwen_cost.total_cost_usd,
        total_cost_usd=total_cost,
    )

    return AgenticTrace(
        question=question,
        hops=tuple(hops),
        controller_results=tuple(controller_results),
        final_generation=generation,
        stop_reason=stop_reason,
        num_retrieval_calls=len(hops),
        num_controller_calls=len(controller_results),
        num_generation_calls=1,
        retrieval_latency_ms=sum(h.retrieval_latency_ms for h in hops),
        reranking_latency_ms=sum(h.reranking_latency_ms for h in hops),
        controller_latency_ms=sum(c.mantle_response.llm_latency_ms for c in controller_results),
        generation_latency_ms=generation.mantle_response.llm_latency_ms,
        total_latency_ms=(time.monotonic() - t_loop_start) * 1000,
        cost=cost_summary,
        unique_chunks_retrieved=len(evidence_pool),
        unique_documents_retrieved=len({r.doc_id for r in evidence_pool}),
        duplicate_chunks_removed=sum(len(h.duplicate_chunk_ids) for h in hops),
        chunks_passed_to_final_generation=len(generation.context.chunks_included),
        chunks_dropped_for_budget=len(generation.context.chunks_dropped),
        evidence_pool=tuple(evidence_pool),
    )
