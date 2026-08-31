"""Evidence Sufficiency Gate — RUNTIME, GLM 4.7 Flash via the existing
Mantle client, structured output (same verified `json_schema` strict
mechanism as `mhrag.agent.controller` / `mhrag.routing.glm_router`).

Judges whether the ACTUAL retrieved chunks handed to it are sufficient to
answer the question — nothing else. Every failure mode is conservative,
resolving to `sufficient=False` (escalate), never silently `True`
(under-route):

  1. the Mantle call itself failing
  2. the response not parsing as valid JSON / not matching the schema
  3. `supporting_chunk_ids` referencing a chunk id that was NOT actually
     given to the gate (a hallucinated reference — deterministically
     validated against the real input, never trusted)
  4. `sufficient=true` but `missing_information` is non-empty — an
     internal contradiction in the model's own output, corrected to
     `sufficient=False` (`GateDecision.conservative_override=True`)

`call_evidence_gate`'s signature takes only `client`, `question`, and
`chunks` (`GateChunkInput` — id/title/text/rank/score, the same fields
already public on any `RetrievalResult`) — no parameter for gold answer,
evidence_list, expected documents, oracle route, question_type, or
Complete-Evidence result exists, so there is no channel for evaluator-only
data to reach this call. The gate is never asked for (and the schema has
no field for) a follow-up search query — that stays the Agentic
controller's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mhrag.generation.mantle_client import MantleClient, MantleResponse
from mhrag.routing.evidence_gate_prompts import (
    EVIDENCE_GATE_PROMPT_VERSION,
    GateChunkInput,
    build_gate_prompt,
)

GATE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "supporting_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["sufficient", "supporting_chunk_ids", "missing_information", "reason"],
    "additionalProperties": False,
}

GATE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_gate_decision",
        "schema": GATE_JSON_SCHEMA,
        "strict": True,
    },
}


@dataclass(frozen=True, slots=True)
class GateDecision:
    sufficient: bool  # FINAL verdict, after the conservative missing_information correction
    raw_sufficient: bool  # the model's own claim, before correction
    supporting_chunk_ids: tuple[str, ...]
    missing_information: tuple[str, ...]
    reason: str
    conservative_override: bool  # True if raw_sufficient=True but missing_information was non-empty


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: GateDecision
    mantle_response: MantleResponse
    fallback_used: bool


def _deterministic_fallback(reason: str) -> GateDecision:
    """Uniform conservative fallback for every failure mode — see module
    docstring. Never `sufficient=True`."""
    return GateDecision(
        sufficient=False, raw_sufficient=False, supporting_chunk_ids=(),
        missing_information=(), reason=reason, conservative_override=False,
    )


def _parse_and_validate(text: str, valid_chunk_ids: frozenset[str]) -> GateDecision | None:
    """Defensive, field-by-field validation, PLUS deterministic validation
    that every `supporting_chunk_ids` entry actually refers to a chunk that
    was given to the gate. Returns `None` (caller falls back) on any
    deviation."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    sufficient = data.get("sufficient")
    if not isinstance(sufficient, bool):
        return None

    supporting_chunk_ids = data.get("supporting_chunk_ids")
    if not isinstance(supporting_chunk_ids, list) or not all(isinstance(c, str) for c in supporting_chunk_ids):
        return None
    if not set(supporting_chunk_ids) <= valid_chunk_ids:
        return None  # hallucinated chunk id reference — cannot be trusted

    missing_information = data.get("missing_information")
    if not isinstance(missing_information, list) or not all(isinstance(m, str) for m in missing_information):
        return None

    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""

    conservative_override = sufficient and len(missing_information) > 0
    final_sufficient = False if conservative_override else sufficient

    return GateDecision(
        sufficient=final_sufficient,
        raw_sufficient=sufficient,
        supporting_chunk_ids=tuple(supporting_chunk_ids),
        missing_information=tuple(missing_information),
        reason=reason,
        conservative_override=conservative_override,
    )


def call_evidence_gate(
    client: MantleClient,
    question: str,
    chunks: list[GateChunkInput],
    prompt_version: str = EVIDENCE_GATE_PROMPT_VERSION,
) -> GateResult:
    system_prompt, user_prompt = build_gate_prompt(question, chunks, version=prompt_version)
    response = client.complete(system_prompt, user_prompt, response_format=GATE_RESPONSE_FORMAT)

    if not response.success:
        return GateResult(
            decision=_deterministic_fallback(f"gate call failed: {response.error}"),
            mantle_response=response,
            fallback_used=True,
        )

    valid_chunk_ids = frozenset(c.chunk_id for c in chunks)
    decision = _parse_and_validate(response.text, valid_chunk_ids)
    if decision is None:
        return GateResult(
            decision=_deterministic_fallback(
                f"gate returned unparseable/invalid response: {response.text[:200]!r}"
            ),
            mantle_response=response,
            fallback_used=True,
        )

    return GateResult(decision=decision, mantle_response=response, fallback_used=False)
