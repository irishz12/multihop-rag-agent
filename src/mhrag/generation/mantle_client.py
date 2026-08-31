"""Reusable Amazon Bedrock Mantle client.

One abstraction for every LLM call in this project, wrapping the OpenAI
Python SDK against Mantle's OpenAI-compatible endpoint — per the approved
architecture ("use the OpenAI Python SDK against Bedrock Mantle... do not
add a second LLM framework"). No other LLM client library is used anywhere
in this codebase.

Endpoint and authentication come ONLY from the environment:
  - base URL: `$MANTLE_BASE_URL`, falling back to a documented default
    (ap-south-1) if unset — never hardcoded as the only option, always
    overridable per environment.
  - API key: `$OPENAI_API_KEY` — REQUIRED for any real call. Read once at
    client construction via `os.environ`, handed directly to the OpenAI SDK
    constructor, and never stored on `MantleClient` as a readable attribute,
    never included in `MantleResponse`, never logged, printed, or written to
    any file by this module or any caller that only uses the types here.

Model ID, temperature, max output tokens, timeout, and retry policy are all
constructor parameters — see `configs/mantle.yaml` for the values actually
used (kept in config, not hardcoded, so they can change without a code
edit).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

DEFAULT_BASE_URL = "https://bedrock-mantle.ap-south-1.api.aws/v1"
DEFAULT_BASE_URL_ENV = "MANTLE_BASE_URL"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

# Transient errors only — retried with exponential backoff. Verified against
# the installed `openai` SDK's exception hierarchy: these four are distinct
# leaves (APIConnectionError/APITimeoutError under APIConnectionError;
# RateLimitError/InternalServerError under APIStatusError), never a
# supertype of a non-retryable error like AuthenticationError or
# BadRequestError, so this tuple can be checked first without risk of
# accidentally swallowing a non-transient failure.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class MantleConfigError(RuntimeError):
    """Raised when required Mantle configuration (the API key) is missing.
    Never includes any secret value in its message."""


@dataclass(frozen=True, slots=True)
class MantleUsage:
    """Structured usage extraction — tolerant of a missing/partial `usage`
    field (some OpenAI-compatible backends omit it or return partial data)."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class MantleResponse:
    """Everything a caller needs for tracing/cost/debugging — never the
    request headers, API key, or raw SDK response object."""

    text: str
    model: str
    usage: MantleUsage
    llm_latency_ms: float
    total_latency_ms: float
    retry_count: int
    success: bool
    error: str | None = None


def extract_usage(response: Any) -> MantleUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return MantleUsage(input_tokens=None, output_tokens=None, total_tokens=None)
    return MantleUsage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _load_base_url(base_url_env: str, default_base_url: str) -> str:
    import os

    return os.environ.get(base_url_env) or default_base_url


def _load_api_key(api_key_env: str) -> str:
    import os

    key = os.environ.get(api_key_env)
    if not key:
        raise MantleConfigError(
            f"Missing required environment variable {api_key_env!r} — Bedrock Mantle "
            "authentication requires it (see .env.example). Never hardcode this key "
            "in code or config."
        )
    return key


class MantleClient:
    """One reusable client for every Mantle call — construct once, reuse
    across queries.

    `client` (an already-constructed OpenAI-SDK-compatible object exposing
    `.chat.completions.create(...)`) can be injected directly, bypassing
    environment/API-key loading entirely — this is how tests exercise
    `complete()`'s retry/usage-extraction/error-handling logic with a fake,
    without ever touching a real key or the network. Production code should
    leave it `None` so the real OpenAI SDK client is constructed from
    `$MANTLE_BASE_URL`/`$OPENAI_API_KEY`.
    """

    def __init__(
        self,
        model_id: str,
        base_url_env: str = DEFAULT_BASE_URL_ENV,
        default_base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds

        if client is not None:
            self._client = client
        else:
            base_url = _load_base_url(base_url_env, default_base_url)
            api_key = _load_api_key(api_key_env)
            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> MantleResponse:
        """Single chat completion call with retry-on-transient-error and
        structured usage/latency/error extraction.

        `response_format` is optional and additive — omitted entirely from
        the request when `None` (the default), so every Phase 6 call site
        (final-answer generation) sends an identical request to before this
        parameter existed. Passed straight through to the SDK when given
        (e.g. `{"type": "json_schema", "json_schema": {...}}` — see
        `mhrag.agent.controller`, verified working against GLM 4.7 Flash
        through Mantle).

        Never raises for a Mantle-side failure (transient, exhausted-retry,
        or non-retryable) — always returns a `MantleResponse`, with
        `success=False` and `error` set on failure, so callers (evaluation
        harnesses, smoke checks) always have something uniform to log/trace
        rather than needing a try/except around every call site.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra_kwargs = {"response_format": response_format} if response_format is not None else {}

        t_start = time.monotonic()
        last_error: Exception | None = None

        # `attempt` is 0 for the original call, 1..max_retries for each
        # retry — so `retry_count=attempt` always reports the number of
        # RETRIES actually used (0 if it succeeded first try), never
        # conflated with the total attempt count (= retry_count + 1).
        for attempt in range(self.max_retries + 1):
            t_call = time.monotonic()
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    **extra_kwargs,
                )
            except RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay_seconds * (2**attempt))
                    continue
                break  # retries exhausted
            except Exception as exc:  # non-retryable — fail clean, don't raise
                return MantleResponse(
                    text="",
                    model=self.model_id,
                    usage=MantleUsage(None, None, None),
                    llm_latency_ms=(time.monotonic() - t_call) * 1000,
                    total_latency_ms=(time.monotonic() - t_start) * 1000,
                    retry_count=attempt,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

            llm_latency_ms = (time.monotonic() - t_call) * 1000
            total_latency_ms = (time.monotonic() - t_start) * 1000
            text = response.choices[0].message.content or ""
            return MantleResponse(
                text=text,
                model=self.model_id,
                usage=extract_usage(response),
                llm_latency_ms=llm_latency_ms,
                total_latency_ms=total_latency_ms,
                retry_count=attempt,
                success=True,
            )

        # Retries exhausted on a transient error.
        return MantleResponse(
            text="",
            model=self.model_id,
            usage=MantleUsage(None, None, None),
            llm_latency_ms=0.0,
            total_latency_ms=(time.monotonic() - t_start) * 1000,
            retry_count=self.max_retries,
            success=False,
            error=(
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "unknown retryable failure"
            ),
        )
