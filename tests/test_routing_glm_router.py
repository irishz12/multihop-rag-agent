"""Stage B (GLM router) tests — entirely offline, injected fake Mantle
client (same pattern as tests/test_agent_controller.py). Covers structured-
output parsing, malformed-response fallback (fails toward COMPLEX, not
SIMPLE), and that gold fields never reach the router prompt.
"""

from __future__ import annotations

import inspect
import json

from mhrag.generation.mantle_client import MantleClient
from mhrag.routing.features import QueryFeatures, RetrievalSignals, RouterFeatures
from mhrag.routing.glm_router import FALLBACK_ROUTE, call_glm_router
from mhrag.routing.prompts import build_router_prompt


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


def _mantle_client(actions):
    completions = _ScriptedCompletions(actions)
    client = MantleClient(
        model_id="zai.glm-4.7-flash", client=_FakeOpenAIClient(completions), max_retries=1,
        retry_base_delay_seconds=0.0,
    )
    return client, completions


def _features(has_comparison=False, top1_score=0.02) -> RouterFeatures:
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
        dense_bm25_jaccard_top10=0.4, consensus_fraction_top5=0.4,
        num_unique_docs_top5=4, num_unique_docs_top10=8, mean_abs_rank_diff_common_docs=1.0,
    )
    return RouterFeatures(query=query, retrieval=retrieval)


def _route_json(route, reason="r"):
    return json.dumps({"route": route, "reason": reason})


# --- happy path --------------------------------------------------------------------------


def test_valid_response_parsed_correctly():
    client, _ = _mantle_client([_FakeChatCompletion(_route_json("MEDIUM", "moderate signal"))])
    result = call_glm_router(client, "some question", _features())
    assert result.decision.route == "MEDIUM"
    assert result.decision.reason == "moderate signal"
    assert result.fallback_used is False


def test_all_three_routes_accepted():
    for route in ("SIMPLE", "MEDIUM", "COMPLEX"):
        client, _ = _mantle_client([_FakeChatCompletion(_route_json(route))])
        result = call_glm_router(client, "q", _features())
        assert result.decision.route == route
        assert result.fallback_used is False


# --- malformed / failure handling -----------------------------------------------------------


def test_malformed_json_falls_back_to_complex():
    client, _ = _mantle_client([_FakeChatCompletion("not json at all")])
    result = call_glm_router(client, "q", _features())
    assert result.fallback_used is True
    assert result.decision.route == FALLBACK_ROUTE == "COMPLEX"


def test_invalid_route_value_falls_back_to_complex():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"route": "VERY_HARD", "reason": "r"}))])
    result = call_glm_router(client, "q", _features())
    assert result.fallback_used is True
    assert result.decision.route == "COMPLEX"


def test_missing_route_field_falls_back_to_complex():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"reason": "r"}))])
    result = call_glm_router(client, "q", _features())
    assert result.fallback_used is True
    assert result.decision.route == "COMPLEX"


def test_non_dict_json_falls_back_to_complex():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps(["SIMPLE"]))])
    result = call_glm_router(client, "q", _features())
    assert result.fallback_used is True
    assert result.decision.route == "COMPLEX"


def test_missing_reason_defaults_to_empty_string_not_a_failure():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"route": "SIMPLE"}))])
    result = call_glm_router(client, "q", _features())
    assert result.fallback_used is False
    assert result.decision.route == "SIMPLE"
    assert result.decision.reason == ""


# --- gold-leakage structural checks -----------------------------------------------------------


def test_call_glm_router_signature_has_no_gold_parameter():
    params = list(inspect.signature(call_glm_router).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label"}
    assert not (forbidden & set(params))


def test_prompt_never_contains_gold_markers():
    """Even though only `question` + `features` can be passed, double-check
    the built prompt text itself never contains anything resembling a gold
    marker leaking through some other path."""
    system_prompt, user_prompt = build_router_prompt("What year was Company X founded?", _features())
    for marker in ("GOLD_ANSWER", "evidence_list", "question_type", "oracle"):
        assert marker not in system_prompt
        assert marker not in user_prompt


def test_sent_request_uses_structured_output_and_only_question_text_features():
    client, completions = _mantle_client([_FakeChatCompletion(_route_json("SIMPLE"))])
    call_glm_router(client, "What year was Company X founded?", _features())
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    sent_text = str(call["messages"])
    assert "What year was Company X founded?" in sent_text
    for marker in ("GOLD_ANSWER", "evidence_list", "question_type"):
        assert marker not in sent_text
