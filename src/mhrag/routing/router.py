"""RUNTIME two-stage router orchestrator.

    question -> compute_router_features (frozen Hybrid RRF baseline)
             -> Stage A: classify_heuristic
                  confident?  -> done, no LLM call
                  not confident? -> Stage B: call_glm_router (GLM 4.7 Flash)

`route_question`'s signature has no parameter for gold answer,
evidence_list, question_type, or an oracle route label — only `question`
plus retrieval/LLM infrastructure — so there is no channel through which
evaluator-only data could reach a routing decision (see
tests/test_routing_no_gold_leakage.py). This module never imports
`mhrag.routing.oracle` or `mhrag.routing.tune_thresholds`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from mhrag.generation.cost import CostEstimate, estimate_cost_usd
from mhrag.generation.mantle_client import MantleClient
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.routing.features import RouterFeatures, compute_router_features
from mhrag.routing.glm_router import GlmRouterResult, call_glm_router
from mhrag.routing.heuristic import HeuristicThresholds, HeuristicVerdict, classify_heuristic


@dataclass(frozen=True, slots=True)
class RouteResult:
    route: str
    stage_used: str  # "heuristic" | "glm" | "glm_fallback"
    heuristic_verdict: HeuristicVerdict
    glm_result: GlmRouterResult | None
    features: RouterFeatures
    feature_latency_ms: float
    glm_latency_ms: float
    total_latency_ms: float
    glm_cost: CostEstimate | None


def route_question(
    question: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    glm_client: MantleClient,
    thresholds: HeuristicThresholds,
    glm_input_price_per_million: float | None = None,
    glm_output_price_per_million: float | None = None,
    features: RouterFeatures | None = None,
) -> RouteResult:
    """Route one question through Stage A then, only if ambiguous, Stage B.

    `features`, if given, is used instead of computing them live — lets
    callers reuse an already-built `results/router_dataset.json` entry
    (see scripts/run_router_validation.py) instead of re-querying Qdrant
    for every validation run, and lets tests exercise this function
    entirely offline with a fake `RouterFeatures` (see
    tests/test_routing_router.py). Production callers with no precomputed
    features leave it `None` so `compute_router_features` runs for real.
    """
    t_start = time.monotonic()

    t_feat = time.monotonic()
    if features is None:
        features = compute_router_features(question, qdrant_client, collection_name, embedding_model, bm25_model)
    feature_latency_ms = (time.monotonic() - t_feat) * 1000

    verdict = classify_heuristic(features, thresholds)

    if verdict.confident:
        assert verdict.route is not None
        return RouteResult(
            route=verdict.route,
            stage_used="heuristic",
            heuristic_verdict=verdict,
            glm_result=None,
            features=features,
            feature_latency_ms=feature_latency_ms,
            glm_latency_ms=0.0,
            total_latency_ms=(time.monotonic() - t_start) * 1000,
            glm_cost=None,
        )

    t_glm = time.monotonic()
    glm_result = call_glm_router(glm_client, question, features)
    glm_latency_ms = (time.monotonic() - t_glm) * 1000

    glm_cost = None
    if glm_input_price_per_million is not None and glm_output_price_per_million is not None:
        glm_cost = estimate_cost_usd(
            glm_result.mantle_response.usage, glm_input_price_per_million, glm_output_price_per_million
        )

    return RouteResult(
        route=glm_result.decision.route,
        stage_used="glm_fallback" if glm_result.fallback_used else "glm",
        heuristic_verdict=verdict,
        glm_result=glm_result,
        features=features,
        feature_latency_ms=feature_latency_ms,
        glm_latency_ms=glm_latency_ms,
        total_latency_ms=(time.monotonic() - t_start) * 1000,
        glm_cost=glm_cost,
    )
