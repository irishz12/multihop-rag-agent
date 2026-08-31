"""Learned two-stage router tests — entirely offline: injected
`fused_candidates=`/`router_features=` bypass live Qdrant/embedding/BM25,
a fake reranker (same `.score()` duck-type as tests/test_rerank.py) stands
in for the cross-encoder. NO Mantle client anywhere — this router makes no
LLM calls at all.
"""

from __future__ import annotations

import inspect

import numpy as np

from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.learned_router import LinearModel
from mhrag.routing.learned_sequential_router import route_question_learned


class _FakeReranker:
    def __init__(self, score_by_text: dict[str, float]):
        self._score_by_text = score_by_text
        self.score_calls: list[list[str]] = []

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        self.score_calls.append(list(texts))
        return np.array([self._score_by_text.get(t, 0.0) for t in texts])


def _result(chunk_id, rank, score, text="text") -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=score, method="hybrid", chunk_id=chunk_id, doc_id=f"doc-{chunk_id}",
        title=f"Source {chunk_id}", url=f"https://example.com/{chunk_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text=text, position=0,
    )


def _fused_pool(n=20) -> list[RetrievalResult]:
    return [_result(f"c{i}", i + 1, 1.0 - i * 0.01, text=f"chunk text {i}") for i in range(n)]


def _router_features() -> RouterFeatures:
    query = QueryFeatures(
        query_length_words=6, query_length_chars=40, comparison_marker_count=0, has_comparison_marker=False,
        temporal_marker_count=0, has_temporal_marker=False, conjunction_count=0, has_conjunction_marker=False,
        quoted_span_count=0, numeric_date_indicator_count=0,
    )
    retrieval = RetrievalSignals(
        hybrid_top1_score=0.05, hybrid_top5_mean_score=0.04, score_gap_top1_top2=0.005, score_gap_top1_top5=0.01,
        dense_bm25_jaccard_top10=0.5, consensus_fraction_top5=0.5, num_unique_docs_top5=4, num_unique_docs_top10=8,
        mean_abs_rank_diff_common_docs=1.0,
    )
    return RouterFeatures(query=query, retrieval=retrieval)


def _always_model(threshold: float, sufficient: bool, n_features: int) -> LinearModel:
    """A trivial LinearModel that ignores every feature (coef=0) and
    always predicts the same probability via the intercept alone —
    convenient for control-flow tests."""
    import math

    # sigmoid(intercept) = 0.9 if sufficient else 0.1
    target_p = 0.9 if sufficient else 0.1
    intercept = math.log(target_p / (1 - target_p))
    return LinearModel(
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        scaler_mean=tuple(0.0 for _ in range(n_features)), scaler_scale=tuple(1.0 for _ in range(n_features)),
        coef=tuple(0.0 for _ in range(n_features)), intercept=intercept, threshold=threshold,
    )


STAGE1_N = 19
STAGE2_N = 26


# --- Gate 2 (Stage 2) runs only after Stage 1 escalates -------------------------------------


def test_stage1_sufficient_routes_simple_without_reranking_or_stage2():
    stage1 = _always_model(threshold=0.5, sufficient=True, n_features=STAGE1_N)
    stage2 = _always_model(threshold=0.5, sufficient=True, n_features=STAGE2_N)
    reranker = _FakeReranker({})
    result = route_question_learned(
        "q", None, "c", None, None, reranker, stage1, stage2,
        fused_candidates=_fused_pool(), router_features=_router_features(),
    )
    assert result.route == "SIMPLE"
    assert reranker.score_calls == []  # reranker never invoked
    assert result.reranked_top5 is None
    assert result.stage2_probability is None


def test_stage1_insufficient_triggers_reranker_and_stage2():
    stage1 = _always_model(threshold=0.5, sufficient=False, n_features=STAGE1_N)
    stage2 = _always_model(threshold=0.5, sufficient=True, n_features=STAGE2_N)
    reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
    result = route_question_learned(
        "q", None, "c", None, None, reranker, stage1, stage2,
        fused_candidates=_fused_pool(), router_features=_router_features(),
    )
    assert len(reranker.score_calls) == 1
    assert result.route == "MEDIUM"
    assert result.reranked_top5 is not None
    assert len(result.reranked_top5) == 5
    assert result.stage2_probability is not None


def test_both_stages_insufficient_routes_complex():
    stage1 = _always_model(threshold=0.5, sufficient=False, n_features=STAGE1_N)
    stage2 = _always_model(threshold=0.5, sufficient=False, n_features=STAGE2_N)
    reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
    result = route_question_learned(
        "q", None, "c", None, None, reranker, stage1, stage2,
        fused_candidates=_fused_pool(), router_features=_router_features(),
    )
    assert result.route == "COMPLEX"


# --- no LLM cost, near-zero decision latency ------------------------------------------------


def test_no_mantle_client_parameter_anywhere_in_signature():
    params = list(inspect.signature(route_question_learned).parameters)
    assert not any("mantle" in p.lower() or "glm" in p.lower() for p in params)


def test_decision_latency_is_fast_pure_arithmetic():
    stage1 = _always_model(threshold=0.5, sufficient=True, n_features=STAGE1_N)
    stage2 = _always_model(threshold=0.5, sufficient=True, n_features=STAGE2_N)
    reranker = _FakeReranker({})
    result = route_question_learned(
        "q", None, "c", None, None, reranker, stage1, stage2,
        fused_candidates=_fused_pool(), router_features=_router_features(),
    )
    assert result.decision_latency_ms < 50  # pure dot-product+sigmoid, no network call


# --- determinism ----------------------------------------------------------------------------


def test_repeated_routing_deterministic():
    stage1 = _always_model(threshold=0.5, sufficient=False, n_features=STAGE1_N)
    stage2 = _always_model(threshold=0.5, sufficient=True, n_features=STAGE2_N)
    outcomes = []
    for _ in range(3):
        reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
        result = route_question_learned(
            "q", None, "c", None, None, reranker, stage1, stage2,
            fused_candidates=_fused_pool(), router_features=_router_features(),
        )
        outcomes.append((result.route, result.stage1_probability, result.stage2_probability))
    assert all(o == outcomes[0] for o in outcomes)


# --- structural: no gold, no evaluator-only imports ------------------------------------------


def test_route_question_learned_signature_has_no_gold_parameter():
    params = list(inspect.signature(route_question_learned).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label", "complete_evidence"}
    assert not (forbidden & set(params))


def test_learned_sequential_router_never_imports_evaluator_only_modules():
    import re
    from pathlib import Path

    path = Path(__file__).parent.parent / "src" / "mhrag" / "routing" / "learned_sequential_router.py"
    source = path.read_text()
    assert not re.search(
        r"^\s*(import|from)\s+.*\b(oracle|gate_analysis|tune_thresholds|heuristic|learned_router_training)\b",
        source, re.MULTILINE,
    )
