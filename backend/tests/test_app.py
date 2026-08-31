"""Live demo API tests — entirely offline: the real `Pipeline` (which loads
embedding/BM25/reranker models and requires a live Mantle key) is replaced
with a fake exposing the same `.ask(question) -> AskResponse` contract, so
these tests never load a model, touch Qdrant, or call Mantle.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import AskResponse, HopInfo, RateLimiter


class _FakePipeline:
    def __init__(self):
        self.received_questions: list[str] = []

    def ask(self, question: str) -> AskResponse:
        self.received_questions.append(question)
        return AskResponse(
            answer=f"Answer to: {question}",
            hops=[HopInfo(hop_number=1, query=question, new_chunks=3)],
            retrieval_calls=1,
            controller_calls=1,
            documents_used=["Example Source A", "Example Source B"],
            latency_ms=1234.5,
            estimated_cost_usd=0.00051,
            stop_reason="evidence_sufficient",
        )


class _FailingPipeline:
    def ask(self, question: str) -> AskResponse:
        raise RuntimeError("secret-internal-detail: sk-should-never-leak")


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake = _FakePipeline()
    monkeypatch.setattr(app_module, "Pipeline", lambda: fake)
    app_module.rate_limiter._hits.clear()
    with TestClient(app_module.app) as client:
        yield client, fake


@pytest.fixture
def client_with_failing_pipeline(monkeypatch):
    monkeypatch.setattr(app_module, "Pipeline", lambda: _FailingPipeline())
    app_module.rate_limiter._hits.clear()
    with TestClient(app_module.app) as client:
        yield client


# --- basic behavior --------------------------------------------------------------------------


def test_healthz_returns_ok(fake_pipeline):
    client, _ = fake_pipeline
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_structured_response(fake_pipeline):
    client, _ = fake_pipeline
    response = client.post("/api/ask", json={"question": "Who won the match?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "answer", "hops", "retrieval_calls", "controller_calls",
        "documents_used", "latency_ms", "estimated_cost_usd", "stop_reason",
    }
    assert body["answer"] == "Answer to: Who won the match?"
    assert body["hops"] == [{"hop_number": 1, "query": "Who won the match?", "new_chunks": 3}]
    assert body["retrieval_calls"] == 1
    assert body["documents_used"] == ["Example Source A", "Example Source B"]
    assert body["stop_reason"] == "evidence_sufficient"


def test_ask_trims_whitespace_before_processing(fake_pipeline):
    client, fake = fake_pipeline
    response = client.post("/api/ask", json={"question": "  Who won?  "})
    assert response.status_code == 200
    assert fake.received_questions == ["Who won?"]


# --- validation --------------------------------------------------------------------------------


def test_ask_rejects_empty_question(fake_pipeline):
    client, _ = fake_pipeline
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_whitespace_only_question(fake_pipeline):
    client, _ = fake_pipeline
    response = client.post("/api/ask", json={"question": "     "})
    assert response.status_code == 422


def test_ask_rejects_question_exceeding_max_length(fake_pipeline):
    client, _ = fake_pipeline
    too_long = "a" * (app_module.MAX_QUESTION_LENGTH + 1)
    response = client.post("/api/ask", json={"question": too_long})
    assert response.status_code == 422


def test_ask_accepts_question_at_exact_max_length(fake_pipeline):
    client, _ = fake_pipeline
    exactly_max = "a" * app_module.MAX_QUESTION_LENGTH
    response = client.post("/api/ask", json={"question": exactly_max})
    assert response.status_code == 200


def test_ask_rejects_missing_question_field(fake_pipeline):
    client, _ = fake_pipeline
    response = client.post("/api/ask", json={})
    assert response.status_code == 422


# --- rate limiting -------------------------------------------------------------------------------


def test_rate_limit_blocks_after_max_requests(fake_pipeline):
    client, _ = fake_pipeline
    for _ in range(app_module.RATE_LIMIT_MAX_REQUESTS):
        response = client.post("/api/ask", json={"question": "q"})
        assert response.status_code == 200
    blocked = client.post("/api/ask", json={"question": "q"})
    assert blocked.status_code == 429


def test_rate_limiter_allows_again_after_window_elapses():
    limiter = RateLimiter(max_requests=2, window_seconds=0.05)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False
    time.sleep(0.06)
    assert limiter.allow("client-a") is True


def test_rate_limiter_tracks_clients_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is False


# --- error handling never leaks internals -----------------------------------------------------


def test_pipeline_failure_returns_clean_error_without_leaking_internals(client_with_failing_pipeline):
    response = client_with_failing_pipeline.post("/api/ask", json={"question": "test"})
    assert response.status_code == 502
    body = response.json()
    assert "secret-internal-detail" not in str(body)
    assert "sk-" not in str(body)


# --- CORS ------------------------------------------------------------------------------------------


def test_cors_allows_configured_origin(fake_pipeline):
    client, _ = fake_pipeline
    response = client.options(
        "/api/ask",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unlisted_origin(fake_pipeline):
    client, _ = fake_pipeline
    response = client.options(
        "/api/ask",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_never_configured_as_wildcard():
    assert "*" not in app_module._ALLOWED_ORIGINS
