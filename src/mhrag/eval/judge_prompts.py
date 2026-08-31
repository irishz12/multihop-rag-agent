"""Versioned, deterministic prompt templates for the Phase 9 LLM-as-judge.

Same determinism/versioning contract as `mhrag.generation.prompts`: no
randomness, no timestamp, plain string substitution only
(question/gold_answer/candidate_answer), and the prompt text is pinned to
`JUDGE_PROMPT_VERSION` — a later change to the rubric text means bumping
the version, never editing `_V1` in place, so every judge verdict on file
stays attributable to the exact rubric that produced it.
"""

from __future__ import annotations

JUDGE_PROMPT_VERSION = "v1"

JUDGE_SYSTEM_PROMPT_V1 = (
    "You are a strict, impartial grading assistant for a question-answering "
    "benchmark. You will be given a QUESTION, a REFERENCE ANSWER (assumed "
    "correct), and a CANDIDATE ANSWER produced by a different system. Judge "
    "ONLY whether the candidate answer is factually consistent with and "
    "conveys the same substantive information as the reference answer, "
    "given the question — not writing style, length, or phrasing.\n\n"
    "Grade as exactly one of:\n"
    "- \"correct\": the candidate conveys the same key fact(s) as the "
    "reference answer, with no contradiction.\n"
    "- \"partially_correct\": the candidate is on-topic and gets part of "
    "the reference answer right, but is incomplete, vague, or only "
    "partially matches.\n"
    "- \"incorrect\": the candidate contradicts the reference answer, "
    "answers a different question, or declines to answer (states the "
    "information is insufficient) when the reference answer is a normal, "
    "answerable fact.\n\n"
    "If the reference answer itself is \"Insufficient information.\" (the "
    "question is intentionally unanswerable from the source material), "
    "grade \"correct\" if the candidate ALSO declines to answer for that "
    "reason, and \"incorrect\" if the candidate states a specific answer "
    "anyway.\n\n"
    "Respond with structured JSON only."
)

JUDGE_USER_PROMPT_TEMPLATE_V1 = (
    "QUESTION:\n{question}\n\n"
    "REFERENCE ANSWER:\n{gold_answer}\n\n"
    "CANDIDATE ANSWER:\n{candidate_answer}\n\n"
    "Grade the candidate answer per the rubric."
)

_VERSIONS = {"v1": (JUDGE_SYSTEM_PROMPT_V1, JUDGE_USER_PROMPT_TEMPLATE_V1)}


def build_judge_prompt(
    question: str, gold_answer: str, candidate_answer: str, version: str = JUDGE_PROMPT_VERSION
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the given judge prompt
    version. Raises ValueError for an unknown version rather than silently
    falling back to a default."""
    if version not in _VERSIONS:
        raise ValueError(f"unknown judge prompt version: {version!r} (known: {sorted(_VERSIONS)})")
    system_prompt, user_template = _VERSIONS[version]
    return system_prompt, user_template.format(
        question=question, gold_answer=gold_answer, candidate_answer=candidate_answer
    )
