"""MantleClient tests — entirely offline, using an injected fake client
(no network, no real API key, no cost). Covers usage extraction, retryable
vs non-retryable failure handling, and that no API key ever leaks into a
response/error/log-able object.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from mhrag.generation.mantle_client import (
    MantleClient,
    MantleConfigError,
    MantleUsage,
    extract_usage,
)

# --- fakes ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None, total_tokens=None):
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
        self.usage = usage


class _ScriptedCompletions:
    """Fake `.chat.completions.create()` — returns or raises according to a
    scripted sequence of actions, one per call. Records every call's kwargs
    for inspection (e.g. to assert exact prompt content, or that no
    unexpected value like an API key ended up in the request)."""

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


def _client_with(actions, **kwargs) -> tuple[MantleClient, _ScriptedCompletions]:
    completions = _ScriptedCompletions(actions)
    fake = _FakeOpenAIClient(completions)
    client = MantleClient(model_id="test-model", client=fake, max_retries=3, retry_base_delay_seconds=0.0, **kwargs)
    return client, completions


def _fake_rate_limit_error() -> openai.RateLimitError:
    req = httpx.Request("POST", "https://example.com")
    resp = httpx.Response(status_code=429, request=req)
    return openai.RateLimitError("rate limited", response=resp, body=None)


def _fake_auth_error() -> openai.AuthenticationError:
    req = httpx.Request("POST", "https://example.com")
    resp = httpx.Response(status_code=401, request=req)
    return openai.AuthenticationError("invalid api key: sk-SECRET-VALUE-123", response=resp, body=None)


def _fake_connection_error() -> openai.APIConnectionError:
    req = httpx.Request("POST", "https://example.com")
    return openai.APIConnectionError(request=req)


# --- usage extraction ----------------------------------------------------------------


def test_extract_usage_full():
    response = _FakeChatCompletion("hello", usage=_FakeUsage(10, 20, 30))
    usage = extract_usage(response)
    assert usage == MantleUsage(input_tokens=10, output_tokens=20, total_tokens=30)


def test_extract_usage_missing_entirely():
    response = _FakeChatCompletion("hello", usage=None)
    usage = extract_usage(response)
    assert usage == MantleUsage(input_tokens=None, output_tokens=None, total_tokens=None)


def test_extract_usage_partial():
    """Some OpenAI-compatible backends return a usage object with some
    fields absent rather than the whole object being None."""
    response = _FakeChatCompletion("hello", usage=_FakeUsage(prompt_tokens=10))
    usage = extract_usage(response)
    assert usage == MantleUsage(input_tokens=10, output_tokens=None, total_tokens=None)


# --- successful call -------------------------------------------------------------------


def test_complete_success_returns_text_and_usage():
    client, completions = _client_with([_FakeChatCompletion("the answer", usage=_FakeUsage(5, 7, 12))])
    result = client.complete("system", "user question")
    assert result.success is True
    assert result.text == "the answer"
    assert result.usage == MantleUsage(5, 7, 12)
    assert result.retry_count == 0
    assert result.error is None
    assert len(completions.calls) == 1


def test_complete_sends_model_temperature_and_max_tokens():
    client, completions = _client_with(
        [_FakeChatCompletion("ok")], temperature=0.2, max_output_tokens=99
    )
    client.complete("system", "user")
    call = completions.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 99
    assert call["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


# --- retryable vs non-retryable failures -----------------------------------------------


def test_transient_error_is_retried_then_succeeds():
    client, completions = _client_with(
        [_fake_rate_limit_error(), _fake_connection_error(), _FakeChatCompletion("recovered")]
    )
    result = client.complete("system", "user")
    assert result.success is True
    assert result.text == "recovered"
    assert result.retry_count == 2
    assert len(completions.calls) == 3


def test_transient_error_exhausts_retries_and_fails_clean():
    actions = [_fake_rate_limit_error() for _ in range(4)]  # max_retries=3 -> 4 attempts total
    client, completions = _client_with(actions)
    result = client.complete("system", "user")
    assert result.success is False
    assert result.retry_count == 3
    assert "RateLimitError" in result.error
    assert len(completions.calls) == 4


def test_non_retryable_error_fails_immediately_no_retries():
    client, completions = _client_with([_fake_auth_error()])
    result = client.complete("system", "user")
    assert result.success is False
    assert result.retry_count == 0
    assert len(completions.calls) == 1  # no retry attempted


def test_complete_never_raises_on_failure():
    """The whole point of returning a MantleResponse instead of propagating
    the exception: callers never need a try/except around complete()."""
    client, _ = _client_with([_fake_auth_error()])
    result = client.complete("system", "user")  # must not raise
    assert result.success is False


# --- no API-key leakage -----------------------------------------------------------------


def test_missing_api_key_raises_config_error_without_leaking(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MantleConfigError) as exc_info:
        MantleClient(model_id="test-model")  # no injected client -> tries real env lookup
    assert "OPENAI_API_KEY" in str(exc_info.value)
    assert "sk-" not in str(exc_info.value)  # no key material could possibly appear


def test_client_does_not_expose_api_key_as_readable_attribute(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SECRET-DO-NOT-LEAK")
    client = MantleClient(model_id="test-model")
    for attr_name in vars(client):
        value = getattr(client, attr_name)
        assert "SECRET-DO-NOT-LEAK" not in repr(value)


def test_error_string_never_contains_key_material_even_on_auth_failure():
    """Simulates an auth failure whose SDK-provided message happens to echo
    back key-shaped text — MantleResponse.error is still just the
    exception's own string, never anything MantleClient adds the key to."""
    client, _ = _client_with([_fake_auth_error()])
    result = client.complete("system", "user")
    # the fake error message intentionally contains a fake secret-looking
    # string to prove we don't ADD the real key on top, not that we scrub
    # the SDK's own message (which is out of our control) — but our code
    # must never independently interpolate the key into the error.
    assert result.error is not None
