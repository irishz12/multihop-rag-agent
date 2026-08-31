"""Two-stage router orchestrator tests — entirely offline: injected
`features=` bypasses live Qdrant/embedding/BM25, and a fake Mantle client
(same pattern as tests/test_routing_glm_router.py) stands in for GLM.
Proves: GLM invoked only for ambiguous cases, repeated routing is
deterministic with fixed responses, router cost is tracked, and the
signature has no gold parameter.
"""

from __future__ import annotations

import inspect
import json

from mhrag.generation.mantle_client import MantleClient
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.heuristic import HeuristicThresholds
from mhrag.routing.router import route_question

THRESHOLDS = HeuristicThresholds(
    simple_min_top1_score=0.03, simple_min_agreement=0.5,
    complex_max_top1_score=0.015, complex_max_agreement=0.2,
    medium_min_top1_score=0.02,
)


class _FakeUsage:
    def __init__(self, prompt_tokens=20, completion_tokens=10, total_tokens=30):
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
        return self._actions.pop(0)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _mantle_client(actions):
    completions = _ScriptedCompletions(actions)
    client = MantleClient(
        model_id="zai.glm-4.7-flash", client=_FakeOpenAIClient(completions), max_retries=1,
        retry_base_delay_seconds=0.0,
    )
    return client, completions


def _features(top1_score=0.05, agreement=0.8, has_comparison=False) -> RouterFeatures:
    query = QueryFeatures(
        query_length_words=6, query_length_chars=40,
        comparison_marker_count=1 if has_comparison else 0, has_comparison_marker=has_comparison,
        temporal_marker_count=0, has_temporal_marker=False,
        conjunction_count=0, has_conjunction_marker=False,
        quoted_span_count=0, numeric_date_indicator_count=0,
    )
    retrieval = RetrievalSignals(
        hybrid_top1_score=top1_score, hybrid_top5_mean_score=top1_score * 0.8,
        score_gap_top1_top2=0.005, score_gap_top1_top5=0.01,
        dense_bm25_jaccard_top10=agreement, consensus_fraction_top5=agreement,
        num_unique_docs_top5=4, num_unique_docs_top10=8, mean_abs_rank_diff_common_docs=1.0,
    )
    return RouterFeatures(query=query, retrieval=retrieval)


# --- GLM invoked only for ambiguous cases -------------------------------------------------


def test_confident_heuristic_never_calls_glm():
    client, completions = _mantle_client([_FakeChatCompletion("SHOULD NOT BE CALLED")])
    result = route_question(
        "q", None, "c", None, None, client, THRESHOLDS,
        features=_features(top1_score=0.05, agreement=0.8),  # confident SIMPLE
    )
    assert result.stage_used == "heuristic"
    assert result.route == "SIMPLE"
    assert len(completions.calls) == 0
    assert result.glm_result is None
    assert result.glm_cost is None


def test_ambiguous_heuristic_calls_glm_exactly_once():
    client, completions = _mantle_client([_FakeChatCompletion(json.dumps({"route": "MEDIUM", "reason": "r"}))])
    # top1_score below medium bar (0.02), agreement above complex bar (0.2) -> no rule matches.
    result = route_question(
        "q", None, "c", None, None, client, THRESHOLDS,
        features=_features(top1_score=0.005, agreement=0.3),
    )
    assert result.heuristic_verdict.confident is False
    assert result.stage_used == "glm"
    assert result.route == "MEDIUM"
    assert len(completions.calls) == 1


def test_glm_fallback_stage_used_recorded_distinctly_on_malformed_response():
    client, _ = _mantle_client([_FakeChatCompletion("not json")])
    result = route_question(
        "q", None, "c", None, None, client, THRESHOLDS,
        features=_features(top1_score=0.005, agreement=0.3),
    )
    assert result.stage_used == "glm_fallback"
    assert result.route == "COMPLEX"


# --- cost tracked -------------------------------------------------------------------------


def test_heuristic_only_route_has_no_glm_cost():
    client, _ = _mantle_client([])
    result = route_question(
        "q", None, "c", None, None, client, THRESHOLDS,
        features=_features(top1_score=0.05, agreement=0.8),
        glm_input_price_per_million=0.08, glm_output_price_per_million=0.48,
    )
    assert result.glm_cost is None


def test_glm_route_has_cost_computed_from_usage():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"route": "COMPLEX", "reason": "r"}))])
    result = route_question(
        "q", None, "c", None, None, client, THRESHOLDS,
        features=_features(top1_score=0.005, agreement=0.3),
        glm_input_price_per_million=0.08, glm_output_price_per_million=0.48,
    )
    assert result.glm_cost is not None
    assert result.glm_cost.total_cost_usd is not None
    assert result.glm_cost.total_cost_usd > 0


def test_glm_cost_none_when_pricing_not_provided():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"route": "COMPLEX", "reason": "r"}))])
    result = route_question(
        "q", None, "c", None, None, client, THRESHOLDS,
        features=_features(top1_score=0.005, agreement=0.3),
    )
    assert result.glm_cost is None


# --- determinism ----------------------------------------------------------------------------


def test_repeated_routing_deterministic_with_fixed_responses():
    features = _features(top1_score=0.005, agreement=0.3)
    results = []
    for _ in range(3):
        client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"route": "MEDIUM", "reason": "r"}))])
        result = route_question("q", None, "c", None, None, client, THRESHOLDS, features=features)
        results.append((result.route, result.stage_used, result.heuristic_verdict))
    assert all(r == results[0] for r in results)


def test_repeated_heuristic_only_routing_deterministic():
    features = _features(top1_score=0.05, agreement=0.8)
    results = []
    for _ in range(3):
        client, _ = _mantle_client([])
        result = route_question("q", None, "c", None, None, client, THRESHOLDS, features=features)
        results.append((result.route, result.stage_used))
    assert all(r == results[0] for r in results)


# --- structural: no gold parameter -----------------------------------------------------------


def test_route_question_signature_has_no_gold_parameter():
    params = list(inspect.signature(route_question).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label"}
    assert not (forbidden & set(params))


def test_router_module_never_imports_oracle_or_tune_thresholds():
    import re
    from pathlib import Path

    path = Path(__file__).parent.parent / "src" / "mhrag" / "routing" / "router.py"
    source = path.read_text()
    assert not re.search(r"^\s*(import|from)\s+.*\b(oracle|tune_thresholds)\b", source, re.MULTILINE)
