"""Live portfolio demo API — a thin FastAPI wrapper around the EXISTING,
UNMODIFIED Agentic Multi-Hop RAG pipeline.

    Next.js frontend -> POST /api/ask -> mhrag.agent.loop.run_agentic_retrieval
        -> existing Qdrant collection -> existing Bedrock Mantle generation
        -> structured JSON response

This module contains ZERO retrieval, reranking, agent-controller, or
generation logic of its own — it only constructs the same infrastructure
objects every other script in this project already constructs
(`EmbeddingModel`, `Bm25Model`, `Reranker`, the Qdrant client, two
`MantleClient`s) from the SAME config files (`configs/retrieval.yaml`,
`configs/mantle.yaml`, `configs/agent.yaml` — untouched, unmodified), then
calls `run_agentic_retrieval` exactly as `scripts/agentic_smoke_check.py`
and every Phase 9 benchmark script already do. Every field in the response
below is read directly off the `AgenticTrace` that function already
returns — nothing here recomputes cost, latency, or evidence counts.

Models are loaded ONCE at process startup (`lifespan`), not per request —
loading the embedding/BM25/reranker models takes several seconds each, so
reloading them per question would make the demo unusably slow and would
still be pure infrastructure, not a pipeline change.

Security:
  - `$OPENAI_API_KEY` / `$MANTLE_BASE_URL` are read server-side only, via
    the SAME environment-variable mechanism `mhrag.generation.mantle_client`
    already uses — never returned in any response, never logged.
  - CORS is restricted to an explicit allow-list (`DEMO_ALLOWED_ORIGINS`),
    not `*`.
  - A simple in-memory sliding-window rate limiter caps requests per
    client IP — no external cache/queue, appropriate for a single-process
    portfolio demo.
  - Input is trimmed, rejected if empty, and capped at
    `MAX_QUESTION_LENGTH` characters before it ever reaches retrieval.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import anyio
import anyio.to_thread
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.config import PROJECT_ROOT, load_config
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker

load_dotenv(PROJECT_ROOT / ".env")

MAX_QUESTION_LENGTH = 500
REQUEST_TIMEOUT_SECONDS = 90.0  # generous outer bound; the pipeline's own
# AgenticConfig.timeout_seconds (30s, from configs/agent.yaml) already
# stops the retrieval loop and returns a best-effort answer well before this
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60.0

_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("DEMO_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


class HopInfo(BaseModel):
    hop_number: int
    query: str
    new_chunks: int


class AskResponse(BaseModel):
    answer: str
    hops: list[HopInfo]
    retrieval_calls: int
    controller_calls: int
    documents_used: list[str]
    latency_ms: float
    estimated_cost_usd: float | None
    stop_reason: str


class ErrorResponse(BaseModel):
    error: str


class Pipeline:
    """Everything `run_agentic_retrieval` needs, constructed once at
    startup from the project's own frozen config files — the exact same
    objects `scripts/agentic_smoke_check.py` builds, never a new model
    choice or a new config value."""

    def __init__(self) -> None:
        retrieval_config = load_config("configs/retrieval.yaml")
        mantle_config = load_config("configs/mantle.yaml")
        agent_config_yaml = load_config("configs/agent.yaml")

        self.embedding_model = EmbeddingModel(
            model_name=retrieval_config["embedding"]["model_name"],
            device=retrieval_config["embedding"].get("device"),
            normalize=retrieval_config["embedding"]["normalize"],
            query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
            batch_size=retrieval_config["embedding"]["batch_size"],
        )
        self.bm25_model = Bm25Model(model_name=retrieval_config["bm25"]["model_name"])
        self.reranker = Reranker(
            model_name=retrieval_config["reranker"]["model_name"],
            device=retrieval_config["reranker"].get("device"),
            batch_size=retrieval_config["reranker"]["batch_size"],
        )
        self.qdrant_client = get_client(retrieval_config["qdrant"]["url"])
        self.collection_name = retrieval_config["qdrant"]["collection_name"]

        try:
            self.controller_client = MantleClient(
                model_id=agent_config_yaml["controller"]["model_id"],
                base_url_env=mantle_config["client"]["base_url_env"],
                default_base_url=mantle_config["client"]["default_base_url"],
                api_key_env=mantle_config["client"]["api_key_env"],
                timeout_seconds=mantle_config["client"]["timeout_seconds"],
                temperature=agent_config_yaml["controller"]["temperature"],
                max_output_tokens=agent_config_yaml["controller"]["max_output_tokens"],
                max_retries=mantle_config["client"]["max_retries"],
                retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
            )
            self.generation_client = MantleClient(
                model_id=mantle_config["generation"]["model_id"],
                base_url_env=mantle_config["client"]["base_url_env"],
                default_base_url=mantle_config["client"]["default_base_url"],
                api_key_env=mantle_config["client"]["api_key_env"],
                timeout_seconds=mantle_config["client"]["timeout_seconds"],
                temperature=mantle_config["generation"]["temperature"],
                max_output_tokens=mantle_config["generation"]["max_output_tokens"],
                max_retries=mantle_config["client"]["max_retries"],
                retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
            )
        except MantleConfigError as exc:
            raise RuntimeError(
                f"Cannot start the live demo API: {exc}. Set $OPENAI_API_KEY (see .env.example)."
            ) from exc

        loop_cfg = agent_config_yaml["loop"]
        agent_pricing = agent_config_yaml["pricing"]
        qwen_pricing = mantle_config["pricing"]
        self.agentic_config = AgenticConfig(
            max_hops=loop_cfg["max_hops"],
            hop_top_k=loop_cfg["hop_top_k"],
            max_evidence_chunks=loop_cfg["max_evidence_chunks"],
            max_context_tokens=loop_cfg["max_context_tokens"],
            timeout_seconds=loop_cfg["timeout_seconds"],
            controller_prompt_version=agent_config_yaml["controller"]["prompt_version"],
            generation_prompt_version=mantle_config["generation"]["prompt_version"],
            glm_input_price_per_million=agent_pricing["input_per_million_tokens"],
            glm_output_price_per_million=agent_pricing["output_per_million_tokens"],
            qwen_input_price_per_million=qwen_pricing["input_per_million_tokens"],
            qwen_output_price_per_million=qwen_pricing["output_per_million_tokens"],
        )

    def ask(self, question: str) -> AskResponse:
        trace = run_agentic_retrieval(
            question,
            self.qdrant_client,
            self.collection_name,
            self.embedding_model,
            self.bm25_model,
            self.reranker,
            self.controller_client,
            self.generation_client,
            config=self.agentic_config,
        )

        seen_titles: list[str] = []
        seen_doc_ids: set[str] = set()
        for chunk in trace.final_generation.context.chunks_included:
            if chunk.doc_id not in seen_doc_ids:
                seen_doc_ids.add(chunk.doc_id)
                # ContextChunk has no title field — resolve it from the evidence pool.
                title = next(
                    (r.title for r in trace.evidence_pool if r.doc_id == chunk.doc_id),
                    chunk.doc_id,
                )
                seen_titles.append(title)

        return AskResponse(
            answer=trace.final_generation.answer,
            hops=[
                HopInfo(hop_number=h.hop_number, query=h.query, new_chunks=len(h.new_chunk_ids))
                for h in trace.hops
            ],
            retrieval_calls=trace.num_retrieval_calls,
            controller_calls=trace.num_controller_calls,
            documents_used=seen_titles,
            latency_ms=trace.total_latency_ms,
            estimated_cost_usd=trace.cost.total_cost_usd,
            stop_reason=trace.stop_reason,
        )


class RateLimiter:
    """Sliding-window request cap per client IP, in-process memory only —
    no Redis, no external service, appropriate for a single-instance demo.
    Resets on server restart; that's an accepted trade-off, not a bug."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        hits = self._hits[client_id]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


rate_limiter = RateLimiter(RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline()
    yield


logger = logging.getLogger("demo")

app = FastAPI(title="Agentic Multi-Hop RAG — Live Demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse, responses={429: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=422, detail=f"Question exceeds {MAX_QUESTION_LENGTH} characters.")

    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — max {RATE_LIMIT_MAX_REQUESTS} questions per "
            f"{int(RATE_LIMIT_WINDOW_SECONDS)}s. Try again shortly.",
        )

    try:
        with anyio.fail_after(REQUEST_TIMEOUT_SECONDS):
            return await anyio.to_thread.run_sync(request.app.state.pipeline.ask, question)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="The pipeline took too long to respond.") from exc
    except Exception as exc:  # pragma: no cover - defensive: never leak internals to the client
        logger.exception("Pipeline call failed for a live demo question")
        raise HTTPException(status_code=502, detail=f"Pipeline error: {type(exc).__name__}") from exc
