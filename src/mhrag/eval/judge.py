"""EVALUATOR-ONLY: fixed LLM-as-judge correctness scoring — Phase 9.

ONE structured LLM call per (question, pipeline-answer) pair, grading the
candidate answer against the gold reference answer on a fixed 3-level
rubric (`mhrag.eval.judge_prompts`) — same structured-output mechanism
already verified working through Mantle in `mhrag.agent.controller`
(`response_format={"type": "json_schema", ..., "strict": True}`), reused
here rather than re-verified from scratch.

FROZEN CONFIGURATION (Phase 9 spec: "keep judge configuration frozen"):
  - model: `openai.gpt-oss-120b` (see `configs/judge.yaml`) — deliberately
    a THIRD, distinct model from both `qwen.qwen3-next-80b-a3b-instruct`
    (produces every candidate answer being judged — never grades its own
    output) and `zai.glm-4.7-flash` (the Agentic Multi-Hop RAG/Adaptive RAG
    controller — judging is never done by the same model making routing
    decisions either).
  - temperature: 0.0, prompt version "v1" (`JUDGE_PROMPT_VERSION`).
  - `GRADE_TO_SCORE` mapping (correct=1.0, partially_correct=0.5,
    incorrect=0.0) — fixed, not re-derived per run.

JUDGE INPUT IS DELIBERATELY MINIMAL: `call_judge`'s signature accepts only
`question`, `gold_answer`, `candidate_answer` — no pipeline name, no
predicted route, no retrieval method, no generation-model identifier. The
judge grades an answer on its own merits against the reference, with no
channel through which it could learn (and be biased by) which pipeline or
backend produced the candidate — see
tests/test_eval_judge.py::test_call_judge_signature_has_no_pipeline_identifying_parameter.

Judge cost/latency is tracked entirely separately from pipeline cost
(`JudgeResult.mantle_response`, `JudgeResult.cost`) — NEVER added into a
pipeline's own cost summary; the two are reported side by side, never
merged into one number. `input_price_per_million`/`output_price_per_million`
are optional: pricing for `openai.gpt-oss-120b` was not available at
implementation time, so `call_judge` tracks token usage/latency exactly
either way but leaves `JudgeResult.cost` as `None` (never a guessed
number) until real pricing is supplied — same "cost is never silently
guessed" rule as `mhrag.generation.cost.estimate_cost_usd`.

Never trusts the response blindly: `_parse_and_validate` re-checks the
grade is one of the three allowed literals; any call failure or
unparseable/invalid response falls back to a DETERMINISTIC, CONSERVATIVE
verdict (`grade="incorrect"`, `score=0.0`) rather than raising or silently
crediting a broken judge call as correct — `JudgeResult.fallback_used`
keeps this distinguishable from a genuine "incorrect" verdict downstream.

Gold answer is a required parameter here — by design, this is the one
place in the whole evaluation pipeline gold data is allowed to appear
inside an LLM prompt, and only AFTER a pipeline's own answer has already
been generated with no access to it. Never imported by any RUNTIME module
(`mhrag.routing.*`, `mhrag.adaptive.*`, `mhrag.agent.*`,
`mhrag.generation.*`) — see tests/test_eval_judge_no_runtime_leakage.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mhrag.eval.judge_prompts import JUDGE_PROMPT_VERSION, build_judge_prompt
from mhrag.generation.cost import CostEstimate, estimate_cost_usd
from mhrag.generation.mantle_client import MantleClient, MantleResponse

GRADES = ("correct", "partially_correct", "incorrect")
GRADE_TO_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": list(GRADES)},
        "reason": {"type": "string"},
    },
    "required": ["grade", "reason"],
    "additionalProperties": False,
}

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_verdict",
        "schema": JUDGE_JSON_SCHEMA,
        "strict": True,
    },
}


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    grade: str  # one of GRADES
    score: float  # GRADE_TO_SCORE[grade]
    reason: str


@dataclass(frozen=True, slots=True)
class JudgeResult:
    verdict: JudgeVerdict
    mantle_response: MantleResponse
    cost: CostEstimate | None  # None whenever pricing wasn't supplied — never a guessed number
    fallback_used: bool  # True if the real call/parse failed and this is the deterministic fallback


def _deterministic_fallback(reason: str) -> JudgeVerdict:
    """On any judge failure, fall back to the MOST CONSERVATIVE verdict
    (incorrect / score 0.0) — a broken judge call must never silently
    inflate a pipeline's measured quality."""
    return JudgeVerdict(grade="incorrect", score=0.0, reason=reason)



# VERIFIED, REPRODUCIBLE GLITCH (Phase 9 judge-validation sample, temperature=0,
# discovered BEFORE the full run per "validate judge behavior on a small sample" —
# see results/phase9_judge_validation.json): openai.gpt-oss-120b, through this Mantle
# endpoint's strict json_schema mode, reliably emits a DUPLICATED leading opening
# brace with no matching second closing brace — e.g. '{\n{\n  "grade": "correct"\n
# ,\n  "reason": "..."\n}' (2 opens, 1 close) — never observed from zai.glm-4.7-flash's
# structured output (mhrag.agent.controller). This is NOT a generic "be lenient with
# any malformed JSON" allowance: `_repair_known_double_open_brace` only strips a
# provably-extraneous SECOND leading '{' when the rest still parses as valid,
# schema-conforming JSON; anything else still falls through to the same conservative
# fallback as before.
_DOUBLE_OPEN_BRACE_RE = re.compile(r"^(\s*\{)\s*\{", re.DOTALL)


def _repair_known_double_open_brace(text: str) -> str:
    match = _DOUBLE_OPEN_BRACE_RE.match(text)
    if match is None:
        return text
    return match.group(1) + text[match.end():]


def _parse_and_validate(text: str) -> JudgeVerdict | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            data = json.loads(_repair_known_double_open_brace(text))
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(data, dict):
        return None

    grade = data.get("grade")
    if grade not in GRADES:
        return None

    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""

    return JudgeVerdict(grade=grade, score=GRADE_TO_SCORE[grade], reason=reason)


def call_judge(
    client: MantleClient,
    question: str,
    gold_answer: str,
    candidate_answer: str,
    prompt_version: str = JUDGE_PROMPT_VERSION,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
) -> JudgeResult:
    """`input_price_per_million`/`output_price_per_million` are optional —
    when both are given, `JudgeResult.cost` is computed from the real
    token usage via `mhrag.generation.cost.estimate_cost_usd` (same
    function every other Mantle cost figure in this project goes through);
    when either is `None` (the default — `openai.gpt-oss-120b` pricing was
    not available at implementation time), `JudgeResult.cost` is `None`,
    never a guessed number. Token usage and latency are always tracked
    precisely either way, via `JudgeResult.mantle_response`."""
    system_prompt, user_prompt = build_judge_prompt(question, gold_answer, candidate_answer, version=prompt_version)

    response = client.complete(system_prompt, user_prompt, response_format=JUDGE_RESPONSE_FORMAT)
    cost = (
        estimate_cost_usd(response.usage, input_price_per_million, output_price_per_million)
        if input_price_per_million is not None and output_price_per_million is not None
        else None
    )

    if not response.success:
        return JudgeResult(
            verdict=_deterministic_fallback(f"judge call failed: {response.error}"),
            mantle_response=response,
            cost=cost,
            fallback_used=True,
        )

    verdict = _parse_and_validate(response.text)
    if verdict is None:
        return JudgeResult(
            verdict=_deterministic_fallback(
                f"judge returned unparseable/invalid response: {response.text[:200]!r}"
            ),
            mantle_response=response,
            cost=cost,
            fallback_used=True,
        )

    return JudgeResult(verdict=verdict, mantle_response=response, cost=cost, fallback_used=False)
