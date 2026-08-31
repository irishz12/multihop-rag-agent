"""Agent controller: ONE structured LLM call per hop, deciding both
sufficiency and (if insufficient) the next focused follow-up query —
preferred over separate sufficiency+subquery calls, per the Phase 7 spec.

STRUCTURED-OUTPUT MECHANISM — VERIFIED, NOT ASSUMED. Tested ad hoc against
`zai.glm-4.7-flash` through the installed Mantle/OpenAI-compatible API
before writing this module: all three of
  - `response_format={"type": "json_object"}`
  - `response_format={"type": "json_schema", "json_schema": {..., "strict": True}}`
  - forced tool-calling (`tools=[...]`, `tool_choice={"type":"function",...}`)
returned valid, schema-conformant JSON for a controller-shaped prompt. This
module uses the `json_schema` strict variant as the primary mechanism (the
strongest guarantee: schema-conformant generation, not just "some JSON").

Even so, the response is NEVER trusted blindly: `_parse_and_validate`
re-checks types field by field, and any failure — the call itself failing,
or the returned text not parsing/validating — falls through to a
deterministic fallback (`ControllerResult.fallback_used=True`) rather than
raising or guessing. The orchestrator (`mhrag.agent.loop`) turns that into
stop_reason="controller_failure", never confusing it with a genuine
sufficiency verdict.

The controller sees ONLY the original question, retrieved evidence text,
and previous search queries — `call_controller`'s signature has no
parameter for gold answer, `evidence_list`, `question_type`, or any
evaluation label, so there is no channel to pass them through even by
mistake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mhrag.agent.prompts import CONTROLLER_PROMPT_VERSION, build_controller_prompt
from mhrag.generation.mantle_client import MantleClient, MantleResponse
from mhrag.retrieval.schema import RetrievalResult

CONTROLLER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "next_query": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["sufficient", "next_query", "reason"],
    "additionalProperties": False,
}

CONTROLLER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "controller_decision",
        "schema": CONTROLLER_JSON_SCHEMA,
        "strict": True,
    },
}


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    sufficient: bool
    next_query: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class ControllerResult:
    decision: ControllerDecision
    mantle_response: MantleResponse
    fallback_used: bool  # True if the real call/parse failed and this is the deterministic fallback


def _deterministic_fallback(reason: str) -> ControllerDecision:
    """On any controller failure, deterministically stop the loop —
    `sufficient=True` so the orchestrator proceeds straight to best-effort
    final generation from whatever evidence exists so far, rather than
    looping on a broken controller. The orchestrator records the REAL stop
    reason (`controller_failure`) separately via `ControllerResult.
    fallback_used`, so this is never confused with a genuine sufficiency
    verdict downstream."""
    return ControllerDecision(sufficient=True, next_query=None, reason=reason)


def _parse_and_validate(text: str) -> ControllerDecision | None:
    """Defensive, field-by-field validation — never trusts that `text` is
    well-formed just because structured output was requested. Returns
    `None` on any deviation from the expected shape."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        return None

    next_query = data.get("next_query")
    if next_query is not None and not isinstance(next_query, str):
        return None
    if sufficient:
        next_query = None  # schema contract: null when sufficient — normalize rather than reject

    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""

    return ControllerDecision(sufficient=sufficient, next_query=next_query, reason=reason)


def call_controller(
    client: MantleClient,
    question: str,
    evidence: list[RetrievalResult],
    previous_queries: list[str],
    prompt_version: str = CONTROLLER_PROMPT_VERSION,
) -> ControllerResult:
    evidence_text = "\n\n".join(f"[{i + 1}] {r.text}" for i, r in enumerate(evidence))
    system_prompt, user_prompt = build_controller_prompt(
        question, evidence_text, previous_queries, version=prompt_version
    )

    response = client.complete(system_prompt, user_prompt, response_format=CONTROLLER_RESPONSE_FORMAT)

    if not response.success:
        return ControllerResult(
            decision=_deterministic_fallback(f"controller call failed: {response.error}"),
            mantle_response=response,
            fallback_used=True,
        )

    decision = _parse_and_validate(response.text)
    if decision is None:
        return ControllerResult(
            decision=_deterministic_fallback(
                f"controller returned unparseable/invalid response: {response.text[:200]!r}"
            ),
            mantle_response=response,
            fallback_used=True,
        )

    return ControllerResult(decision=decision, mantle_response=response, fallback_used=False)
