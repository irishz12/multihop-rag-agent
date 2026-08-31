"""LIVE Mantle connectivity test — makes ONE real API call and incurs real
(tiny) cost.

Skipped by DEFAULT so normal `pytest` never spends money or requires
network/credentials — opt in explicitly:

    RUN_LIVE_MANTLE_TESTS=1 pytest tests/test_mantle_live.py -v -s

Opt-in requires an explicit env var (not just the presence of
`OPENAI_API_KEY`, e.g. in a CI secrets store) — merely having the key
configured somewhere must never be enough to trigger a real spend during
routine test runs.

This is a connectivity check, not an answer-quality test: it only asserts
the call succeeds and returns usage data, mirroring what
scripts/mantle_smoke_check.py does at larger (but still small) scale.
"""

from __future__ import annotations

import os

import pytest

from mhrag.config import load_config
from mhrag.generation.mantle_client import MantleClient, MantleConfigError

RUN_LIVE = os.environ.get("RUN_LIVE_MANTLE_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_LIVE,
    reason="live Mantle test — opt in with RUN_LIVE_MANTLE_TESTS=1 (incurs real cost)",
)
def test_live_mantle_connectivity():
    mantle_config = load_config("configs/mantle.yaml")
    try:
        client = MantleClient(
            model_id=mantle_config["generation"]["model_id"],
            base_url_env=mantle_config["client"]["base_url_env"],
            default_base_url=mantle_config["client"]["default_base_url"],
            api_key_env=mantle_config["client"]["api_key_env"],
            timeout_seconds=mantle_config["client"]["timeout_seconds"],
            temperature=mantle_config["generation"]["temperature"],
            max_output_tokens=16,  # keep this connectivity check cheap
            max_retries=mantle_config["client"]["max_retries"],
            retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
        )
    except MantleConfigError as exc:
        pytest.fail(f"RUN_LIVE_MANTLE_TESTS=1 but Mantle is not configured: {exc}")

    result = client.complete(
        "You are a helpful assistant. Answer in one short sentence.",
        "Say hello and nothing else.",
    )

    assert result.success, f"live Mantle call failed: {result.error}"
    assert result.text
    assert result.usage.input_tokens is not None
    assert result.usage.output_tokens is not None
    assert result.llm_latency_ms > 0

    print(f"\n[live] model={result.model}")
    print(f"[live] text={result.text!r}")
    print(f"[live] usage={result.usage}")
    print(f"[live] llm_latency_ms={result.llm_latency_ms:.0f}")
