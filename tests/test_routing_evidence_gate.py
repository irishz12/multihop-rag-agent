"""Evidence Sufficiency Gate tests — entirely offline, injected fake
Mantle client (same pattern as tests/test_agent_controller.py /
tests/test_routing_glm_router.py). Covers structured-output parsing,
deterministic chunk-id validation, the conservative missing_information
override, and that every failure mode escalates (never silently
sufficient=True).
"""

from __future__ import annotations

import inspect
import json

from mhrag.generation.mantle_client import MantleClient
from mhrag.routing.evidence_gate import call_evidence_gate
from mhrag.routing.evidence_gate_prompts import GateChunkInput, build_gate_prompt


class _FakeUsage:
    def __init__(self, prompt_tokens=30, completion_tokens=15, total_tokens=45):
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


def _chunks(n=3) -> list[GateChunkInput]:
    return [GateChunkInput(chunk_id=f"c{i}", title=f"Source {i}", text=f"chunk text {i}", rank=i + 1, score=0.9 - i * 0.1) for i in range(n)]


def _gate_json(sufficient, supporting=None, missing=None, reason="r"):
    return json.dumps({
        "sufficient": sufficient,
        "supporting_chunk_ids": supporting or [],
        "missing_information": missing or [],
        "reason": reason,
    })


# --- happy path --------------------------------------------------------------------------


def test_sufficient_with_valid_supporting_ids_parsed_correctly():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["c0", "c1"]))])
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is False
    assert result.decision.sufficient is True
    assert result.decision.raw_sufficient is True
    assert result.decision.supporting_chunk_ids == ("c0", "c1")
    assert result.decision.conservative_override is False


def test_insufficient_with_missing_information_parsed_correctly():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(False, missing=["the exact date"]))])
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is False
    assert result.decision.sufficient is False
    assert result.decision.missing_information == ("the exact date",)


# --- supplied chunk IDs validated -----------------------------------------------------------


def test_supporting_chunk_ids_referencing_given_chunks_accepted():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["c0", "c2"]))])
    result = call_evidence_gate(client, "q", _chunks(3))
    assert result.fallback_used is False
    assert set(result.decision.supporting_chunk_ids) <= {"c0", "c1", "c2"}


def test_unknown_chunk_id_reference_escalates():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["c0", "HALLUCINATED_ID"]))])
    result = call_evidence_gate(client, "q", _chunks(3))
    assert result.fallback_used is True
    assert result.decision.sufficient is False


def test_all_unknown_chunk_ids_escalates():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["not-a-real-id"]))])
    result = call_evidence_gate(client, "q", _chunks(3))
    assert result.fallback_used is True
    assert result.decision.sufficient is False


# --- conservative missing_information override ------------------------------------------


def test_sufficient_true_with_nonempty_missing_information_escalates_conservatively():
    client, _ = _mantle_client(
        [_FakeChatCompletion(_gate_json(True, supporting=["c0"], missing=["one more source"]))]
    )
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is False  # this parses fine — it's a semantic override, not a parse failure
    assert result.decision.raw_sufficient is True
    assert result.decision.sufficient is False  # forced insufficient
    assert result.decision.conservative_override is True


def test_sufficient_true_with_empty_missing_information_not_overridden():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["c0"], missing=[]))])
    result = call_evidence_gate(client, "q", _chunks())
    assert result.decision.sufficient is True
    assert result.decision.conservative_override is False


# --- malformed output escalates ------------------------------------------------------------


def test_malformed_json_escalates():
    client, _ = _mantle_client([_FakeChatCompletion("not json at all")])
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is True
    assert result.decision.sufficient is False


def test_missing_required_field_escalates():
    client, _ = _mantle_client([_FakeChatCompletion(json.dumps({"sufficient": True}))])
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is True
    assert result.decision.sufficient is False


def test_wrong_type_for_sufficient_escalates():
    client, _ = _mantle_client(
        [_FakeChatCompletion(json.dumps({"sufficient": "yes", "supporting_chunk_ids": [], "missing_information": [], "reason": "r"}))]
    )
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is True
    assert result.decision.sufficient is False


def test_non_string_item_in_supporting_chunk_ids_escalates():
    client, _ = _mantle_client(
        [_FakeChatCompletion(json.dumps({"sufficient": True, "supporting_chunk_ids": [123], "missing_information": [], "reason": "r"}))]
    )
    result = call_evidence_gate(client, "q", _chunks())
    assert result.fallback_used is True
    assert result.decision.sufficient is False


def test_call_failure_escalates():
    # Simulate call failure via a client whose complete() reports success=False.
    import httpx
    import openai

    class _AlwaysFailCompletions:
        def create(self, **kwargs):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://example.com"))

    failing_client = MantleClient(
        model_id="zai.glm-4.7-flash",
        client=type("C", (), {"chat": type("Chat", (), {"completions": _AlwaysFailCompletions()})()})(),
        max_retries=0, retry_base_delay_seconds=0.0,
    )
    result = call_evidence_gate(failing_client, "q", _chunks())
    assert result.fallback_used is True
    assert result.decision.sufficient is False
    assert result.mantle_response.success is False


# --- structural: no follow-up query field, no gold ---------------------------------------


def test_gate_schema_has_no_follow_up_query_field():
    from mhrag.routing.evidence_gate import GATE_JSON_SCHEMA

    assert "next_query" not in GATE_JSON_SCHEMA["properties"]
    assert "follow_up_query" not in GATE_JSON_SCHEMA["properties"]
    assert "query" not in GATE_JSON_SCHEMA["properties"]


def test_call_evidence_gate_signature_has_no_gold_parameter():
    params = list(inspect.signature(call_evidence_gate).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label", "complete_evidence"}
    assert not (forbidden & set(params))


def test_gate_prompt_never_contains_gold_markers():
    system_prompt, user_prompt = build_gate_prompt("What year was Company X founded?", _chunks())
    for marker in ("GOLD_ANSWER", "evidence_list", "question_type", "oracle", "complete_evidence"):
        assert marker not in system_prompt
        assert marker not in user_prompt


def test_sent_request_contains_only_question_and_chunk_fields():
    client, completions = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["c0"]))])
    call_evidence_gate(client, "What year was Company X founded?", _chunks())
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    sent_text = str(call["messages"])
    assert "What year was Company X founded?" in sent_text
    assert "c0" in sent_text and "chunk text 0" in sent_text
    for marker in ("GOLD_ANSWER", "evidence_list", "question_type"):
        assert marker not in sent_text
