"""Stage B: GLM 4.7 Flash router — RUNTIME, called ONLY for cases Stage A
(`mhrag.routing.heuristic`) was not confident about.

Structured-output mechanism mirrors `mhrag.agent.controller` exactly (that
mechanism was verified empirically against `zai.glm-4.7-flash` through
Mantle in Phase 7 — not re-verified here, same client/model, same
`json_schema` strict primary path). The response is never trusted blindly:
`_parse_and_validate` re-checks types field by field, and any failure — the
call itself failing, an invalid route, or unparseable JSON — falls through
to a DETERMINISTIC fallback, `fallback_used=True`.

Fallback route is COMPLEX, not SIMPLE — per the Phase 8A spec's explicit
harm asymmetry ("Under-routing is more harmful to quality"), a broken
router call should fail toward the safer, more-expensive route rather than
risk silently under-serving a hard question.

`call_glm_router`'s signature takes only `client`, `question`, and
`features` (`mhrag.routing.features.RouterFeatures`, itself gold-free) — no
parameter for gold answer, evidence_list, question_type, or oracle route
label exists, so there is no channel to pass them through even by mistake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mhrag.generation.mantle_client import MantleClient, MantleResponse
from mhrag.routing.features import RouterFeatures
from mhrag.routing.prompts import ROUTER_PROMPT_VERSION, build_router_prompt

VALID_ROUTES = ("SIMPLE", "MEDIUM", "COMPLEX")

ROUTER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(VALID_ROUTES)},
        "reason": {"type": "string"},
    },
    "required": ["route", "reason"],
    "additionalProperties": False,
}

ROUTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "router_decision",
        "schema": ROUTER_JSON_SCHEMA,
        "strict": True,
    },
}

FALLBACK_ROUTE = "COMPLEX"  # safer direction — see module docstring


@dataclass(frozen=True, slots=True)
class RouterDecision:
    route: str
    reason: str


@dataclass(frozen=True, slots=True)
class GlmRouterResult:
    decision: RouterDecision
    mantle_response: MantleResponse
    fallback_used: bool


def _deterministic_fallback(reason: str) -> RouterDecision:
    return RouterDecision(route=FALLBACK_ROUTE, reason=reason)


def _parse_and_validate(text: str) -> RouterDecision | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    route = data.get("route")
    if not isinstance(route, str) or route not in VALID_ROUTES:
        return None

    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""

    return RouterDecision(route=route, reason=reason)


def call_glm_router(
    client: MantleClient,
    question: str,
    features: RouterFeatures,
    prompt_version: str = ROUTER_PROMPT_VERSION,
) -> GlmRouterResult:
    system_prompt, user_prompt = build_router_prompt(question, features, version=prompt_version)
    response = client.complete(system_prompt, user_prompt, response_format=ROUTER_RESPONSE_FORMAT)

    if not response.success:
        return GlmRouterResult(
            decision=_deterministic_fallback(f"router call failed: {response.error}"),
            mantle_response=response,
            fallback_used=True,
        )

    decision = _parse_and_validate(response.text)
    if decision is None:
        return GlmRouterResult(
            decision=_deterministic_fallback(
                f"router returned unparseable/invalid response: {response.text[:200]!r}"
            ),
            mantle_response=response,
            fallback_used=True,
        )

    return GlmRouterResult(decision=decision, mantle_response=response, fallback_used=False)
