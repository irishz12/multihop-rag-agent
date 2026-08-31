"""Agent controller tests — entirely offline, using an injected fake Mantle
client (same fake pattern as tests/test_mantle_client.py). Covers
structured-output parsing, malformed-JSON fallback, call-failure fallback,
and that gold answer/evidence/question_type never reach the controller
prompt.
"""

from __future__ import annotations

import inspect

import httpx
import openai

from mhrag.agent.controller import (
    ControllerDecision,
    call_controller,
)
from mhrag.generation.mantle_client import MantleClient
from mhrag.retrieval.schema import RetrievalResult


# --- fakes (mirrors tests/test_mantle_client.py's pattern) ----------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens=5, completion_tokens=5, total_tokens=10):
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


def _client_with(actions) -> tuple[MantleClient, _ScriptedCompletions]:
    completions = _ScriptedCompletions(actions)
    fake = _FakeOpenAIClient(completions)
    client = MantleClient(model_id="glm-test", client=fake, max_retries=1, retry_base_delay_seconds=0.0)
    return client, completions


def _auth_error():
    req = httpx.Request("POST", "https://example.com")
    resp = httpx.Response(status_code=401, request=req)
    return openai.AuthenticationError("bad key", response=resp, body=None)


def _result(chunk_id: str, doc_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=1.0, method="hybrid_reranked", chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text=text, position=0,
    )


# --- structured-output parsing ---------------------------------------------------------


def test_valid_structured_response_parses_insufficient():
    client, _ = _client_with(
        [_FakeChatCompletion('{"sufficient": false, "next_query": "who is the CEO?", "reason": "missing CEO"}')]
    )
    result = call_controller(client, "q", [_result("a", "doc-a", "some evidence")], [])
    assert result.fallback_used is False
    assert result.decision == ControllerDecision(sufficient=False, next_query="who is the CEO?", reason="missing CEO")


def test_valid_structured_response_parses_sufficient():
    client, _ = _client_with(
        [_FakeChatCompletion('{"sufficient": true, "next_query": null, "reason": "fully answered"}')]
    )
    result = call_controller(client, "q", [_result("a", "doc-a", "evidence")], [])
    assert result.fallback_used is False
    assert result.decision.sufficient is True
    assert result.decision.next_query is None


def test_sufficient_true_normalizes_stray_next_query_to_none():
    """Defensive: even if the model violates its own schema contract (next_query
    non-null while sufficient=true), the parser normalizes rather than
    propagating an inconsistent decision."""
    client, _ = _client_with(
        [_FakeChatCompletion('{"sufficient": true, "next_query": "some query", "reason": "done"}')]
    )
    result = call_controller(client, "q", [], [])
    assert result.decision.sufficient is True
    assert result.decision.next_query is None


# --- malformed JSON / missing fields -> deterministic fallback ------------------------


def test_malformed_json_falls_back():
    client, _ = _client_with([_FakeChatCompletion("this is not json at all")])
    result = call_controller(client, "q", [], [])
    assert result.fallback_used is True
    assert result.decision.sufficient is True  # fallback stops the loop
    assert result.decision.next_query is None
    assert "unparseable" in result.decision.reason or "invalid" in result.decision.reason


def test_missing_optional_fields_default_leniently_not_a_fallback():
    """"reason"/"next_query" absent (rather than explicitly null) is
    tolerated — only a malformed/wrong-typed "sufficient" triggers the
    fallback path; the parser isn't needlessly strict about optional
    fields it can safely default."""
    client, _ = _client_with([_FakeChatCompletion('{"sufficient": false}')])
    result = call_controller(client, "q", [], [])
    assert result.fallback_used is False
    assert result.decision.sufficient is False
    assert result.decision.reason == ""


def test_json_with_wrong_type_for_sufficient_falls_back():
    client, _ = _client_with([_FakeChatCompletion('{"sufficient": "yes", "next_query": null, "reason": "x"}')])
    result = call_controller(client, "q", [], [])
    assert result.fallback_used is True


def test_json_array_instead_of_object_falls_back():
    client, _ = _client_with([_FakeChatCompletion('["sufficient", false]')])
    result = call_controller(client, "q", [], [])
    assert result.fallback_used is True


def test_next_query_wrong_type_falls_back():
    client, _ = _client_with([_FakeChatCompletion('{"sufficient": false, "next_query": 42, "reason": "x"}')])
    result = call_controller(client, "q", [], [])
    assert result.fallback_used is True


# --- call failure -> deterministic fallback --------------------------------------------


def test_controller_call_failure_falls_back():
    client, _ = _client_with([_auth_error()])
    result = call_controller(client, "q", [], [])
    assert result.fallback_used is True
    assert result.decision.sufficient is True
    assert "controller call failed" in result.decision.reason
    assert result.mantle_response.success is False


# --- request shape: json_schema response_format is actually requested -----------------


def test_call_controller_requests_json_schema_response_format():
    client, completions = _client_with(
        [_FakeChatCompletion('{"sufficient": true, "next_query": null, "reason": "ok"}')]
    )
    call_controller(client, "q", [], [])
    sent = completions.calls[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True


# --- gold answer/evidence/question_type never enter the controller prompt -------------


def test_call_controller_signature_has_no_ground_truth_parameter():
    params = set(inspect.signature(call_controller).parameters)
    for forbidden in ("answer", "gold_answer", "evidence_list", "question_type", "expected_documents"):
        assert forbidden not in params


def test_controller_prompt_never_contains_gold_fields():
    from mhrag.data.schema import Evidence, QARecord

    record = QARecord(
        query="What year was the company founded?",
        answer="GOLD_ANSWER_MARKER_24680",
        question_type="inference_query",
        evidence_list=(
            Evidence(
                title="t", author=None, url="https://example.com/doc-1", source="s", category="c",
                published_at="2024-01-01T00:00:00+00:00", fact="GOLD_EVIDENCE_FACT_MARKER_11223",
            ),
        ),
    )
    client, completions = _client_with(
        [_FakeChatCompletion('{"sufficient": true, "next_query": null, "reason": "ok"}')]
    )
    retrieved = [_result("c1", "doc-1", "ordinary retrieved chunk text")]

    call_controller(client, record.query, retrieved, [])

    sent_text = str(completions.calls[0]["messages"])
    assert record.answer not in sent_text
    assert record.evidence_list[0].fact not in sent_text
    assert record.question_type not in sent_text
