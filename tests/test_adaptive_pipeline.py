"""Adaptive pipeline (Phase 8B) control-flow tests — entirely offline:
injected `fused_candidates=`/`router_features=` bypass live Qdrant/
embedding/BM25, a fake reranker (same `.score()` duck-type as
tests/test_rerank.py / tests/test_routing_learned_sequential_router.py)
stands in for the cross-encoder, and fake Mantle clients (same injection
pattern as tests/test_agent_loop.py) stand in for the GLM controller and
Qwen final-generation calls. No live service involved.
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np

from mhrag.adaptive.pipeline import run_adaptive_pipeline
from mhrag.agent.loop import AgenticConfig
from mhrag.generation.mantle_client import MantleClient
from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.learned_router import LinearModel

# --- fakes (same pattern as tests/test_agent_loop.py) ----------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, total_tokens=15):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeChatCompletion:
    def __init__(self, content, usage=None):
        self.choices = [_FakeChoice(content)]
        self.usage = usage or _FakeUsage()


class _ScriptedCompletions:
    def __init__(self, actions):
        self._actions = list(actions)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self._actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _mantle_client(actions) -> MantleClient:
    fake = _FakeOpenAIClient(_ScriptedCompletions(actions))
    return MantleClient(model_id="test-model", client=fake, max_retries=1, retry_base_delay_seconds=0.0)


def _controller_json(sufficient: bool, next_query: str | None, reason: str = "r") -> str:
    return json.dumps({"sufficient": sufficient, "next_query": next_query, "reason": reason})


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
    target_p = 0.9 if sufficient else 0.1
    intercept = math.log(target_p / (1 - target_p))
    return LinearModel(
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        scaler_mean=tuple(0.0 for _ in range(n_features)), scaler_scale=tuple(1.0 for _ in range(n_features)),
        coef=tuple(0.0 for _ in range(n_features)), intercept=intercept, threshold=threshold,
    )


STAGE1_N = 19
STAGE2_N = 26
_RERANK_SCORES = {f"chunk text {i}": float(20 - i) for i in range(20)}


def _config(**overrides) -> AgenticConfig:
    defaults = dict(
        max_hops=3, hop_top_k=5, max_evidence_chunks=15, max_context_tokens=4500, timeout_seconds=30.0,
        qwen_input_price_per_million=0.18, qwen_output_price_per_million=1.41,
        glm_input_price_per_million=0.08, glm_output_price_per_million=0.48,
    )
    defaults.update(overrides)
    return AgenticConfig(**defaults)


def _run(stage1_sufficient, stage2_sufficient, controller_actions=None, generation_action=None,
          agent_followup_hop_runner=None, config=None):
    stage1 = _always_model(threshold=0.5, sufficient=stage1_sufficient, n_features=STAGE1_N)
    stage2 = _always_model(threshold=0.5, sufficient=stage2_sufficient, n_features=STAGE2_N)
    reranker = _FakeReranker(_RERANK_SCORES)
    controller_client = _mantle_client(controller_actions or [_FakeChatCompletion(_controller_json(True, None))])
    generation_client = _mantle_client([generation_action or _FakeChatCompletion("final answer text")])
    trace = run_adaptive_pipeline(
        "What year and who?", None, "collection", None, None, reranker, stage1, stage2,
        controller_client, generation_client, agentic_config=config or _config(),
        fused_candidates=_fused_pool(), router_features=_router_features(),
        agent_followup_hop_runner=agent_followup_hop_runner,
    )
    return trace, reranker


# --- SIMPLE route ------------------------------------------------------------------------


def test_simple_route_skips_reranker_and_controller():
    trace, reranker = _run(stage1_sufficient=True, stage2_sufficient=True)
    assert trace.route == "SIMPLE"
    assert trace.stop_reason == "route_simple"
    assert reranker.score_calls == []
    assert trace.num_retrieval_calls == 1
    assert trace.num_reranker_calls == 0
    assert trace.num_agent_hops == 0
    assert trace.num_controller_calls == 0
    assert trace.glm_input_tokens == 0 and trace.glm_output_tokens == 0
    assert trace.glm_cost_usd is None
    assert trace.qwen_input_tokens > 0
    assert trace.answer == "final answer text"
    assert trace.agentic_trace is None
    assert trace.stage2_probability is None


# --- MEDIUM route --------------------------------------------------------------------------


def test_medium_route_calls_reranker_once_no_controller():
    trace, reranker = _run(stage1_sufficient=False, stage2_sufficient=True)
    assert trace.route == "MEDIUM"
    assert trace.stop_reason == "route_medium"
    assert len(reranker.score_calls) == 1
    assert trace.num_retrieval_calls == 1
    assert trace.num_reranker_calls == 1
    assert trace.num_controller_calls == 0
    assert trace.agentic_trace is None
    assert trace.stage2_probability is not None


# --- COMPLEX route: hop-1 reuse ------------------------------------------------------------


def test_complex_route_reuses_hop1_no_repeated_retrieval_or_rerank():
    """Controller says sufficient on the very first (reused) hop — proves
    hop 1 costs exactly one reranker call, not two."""
    trace, reranker = _run(
        stage1_sufficient=False, stage2_sufficient=False,
        controller_actions=[_FakeChatCompletion(_controller_json(True, None))],
    )
    assert trace.route == "COMPLEX"
    assert trace.stop_reason == "evidence_sufficient"
    assert trace.num_retrieval_calls == 1
    assert trace.num_reranker_calls == 1
    assert trace.num_agent_hops == 1
    assert trace.num_controller_calls == 1
    assert len(reranker.score_calls) == 1  # the Stage-2 rerank, never repeated for hop 1
    assert trace.agentic_trace is not None


def test_complex_route_second_hop_uses_real_fallback_runner_not_cache():
    """Controller escalates once, so hop 2 must run a genuinely NEW
    retrieval via the injected follow-up hop runner (not the cached hop-1
    result again)."""
    followup_calls: list[str] = []

    def followup_runner(query: str):
        followup_calls.append(query)
        return [_result("new1", 1, 5.0, text="new chunk")], 12.0, 6.0

    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "follow up query")),
        _FakeChatCompletion(_controller_json(True, None)),
    ]
    trace, reranker = _run(
        stage1_sufficient=False, stage2_sufficient=False,
        controller_actions=controller_actions, agent_followup_hop_runner=followup_runner,
    )
    assert trace.route == "COMPLEX"
    assert trace.num_retrieval_calls == 2
    assert trace.num_reranker_calls == 2
    assert followup_calls == ["follow up query"]
    assert len(reranker.score_calls) == 1  # hop 1 still never re-reranked; hop 2 used the fake follow-up runner


def test_complex_route_max_hops_enforced():
    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "q2")),
        _FakeChatCompletion(_controller_json(False, "q3")),
        _FakeChatCompletion(_controller_json(False, "q4")),  # would ask for a 4th hop if allowed
    ]
    calls = []

    def followup_runner(query: str):
        calls.append(query)
        return [_result(f"n{len(calls)}", 1, 5.0, text=f"chunk {len(calls)}")], 1.0, 1.0

    trace, _ = _run(
        stage1_sufficient=False, stage2_sufficient=False,
        controller_actions=controller_actions, agent_followup_hop_runner=followup_runner,
    )
    assert trace.num_retrieval_calls == 3
    assert trace.stop_reason == "max_hops"


# --- router probabilities always populated appropriately ------------------------------------


def test_stage1_probability_always_present_stage2_only_when_computed():
    simple_trace, _ = _run(stage1_sufficient=True, stage2_sufficient=True)
    assert 0.0 <= simple_trace.stage1_probability <= 1.0
    assert simple_trace.stage2_probability is None

    medium_trace, _ = _run(stage1_sufficient=False, stage2_sufficient=True)
    assert 0.0 <= medium_trace.stage1_probability <= 1.0
    assert medium_trace.stage2_probability is not None


# --- same Qwen model/prompt/pricing across routes --------------------------------------------


def test_qwen_cost_computed_from_configured_pricing_for_every_route():
    config = _config(qwen_input_price_per_million=1000.0, qwen_output_price_per_million=1000.0)
    usage = _FakeUsage(prompt_tokens=100, completion_tokens=50)
    trace, _ = _run(
        stage1_sufficient=True, stage2_sufficient=True,
        generation_action=_FakeChatCompletion("answer", usage=usage), config=config,
    )
    expected_cost = (100 / 1_000_000) * 1000.0 + (50 / 1_000_000) * 1000.0
    assert trace.qwen_cost_usd == expected_cost


# --- determinism -----------------------------------------------------------------------------


def test_repeated_routing_deterministic():
    outcomes = []
    for _ in range(3):
        trace, _ = _run(stage1_sufficient=False, stage2_sufficient=True)
        outcomes.append((trace.route, trace.stage1_probability, trace.stage2_probability))
    assert all(o == outcomes[0] for o in outcomes)


# --- structural: no gold parameter, no evaluator-only imports ---------------------------------


def test_run_adaptive_pipeline_signature_has_no_gold_parameter():
    params = list(inspect.signature(run_adaptive_pipeline).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label", "complete_evidence"}
    assert not (forbidden & set(params))


def test_adaptive_pipeline_never_imports_evaluator_only_modules():
    import re
    from pathlib import Path

    path = Path(__file__).parent.parent / "src" / "mhrag" / "adaptive" / "pipeline.py"
    source = path.read_text()
    assert not re.search(
        r"^\s*(import|from)\s+.*\b(oracle|gate_analysis|tune_thresholds|heuristic|learned_router_training)\b",
        source, re.MULTILINE,
    )
