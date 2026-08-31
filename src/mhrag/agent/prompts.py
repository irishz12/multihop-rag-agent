"""Versioned, deterministic prompt for the agent controller (GLM 4.7 Flash).

The controller may see ONLY: the original question, retrieved evidence
text (chunk text — never gold), and the search queries already tried
(a short search history). It must NEVER see: gold answer, `evidence_list`,
`question_type`, expected documents, or any evaluation label —
`build_controller_prompt`'s signature has no parameter for any of them, so
there is no channel a caller could use to pass them through even by
mistake (same structural-guarantee pattern as
`mhrag.generation.prompts.build_prompt`).

Deterministic and versioned for the same reason as the Phase 6 generation
prompt: no randomness in the template text itself, and a later change to
the wording must bump `CONTROLLER_PROMPT_VERSION` rather than editing
`_V1` in place.
"""

from __future__ import annotations

CONTROLLER_PROMPT_VERSION = "v1"

CONTROLLER_SYSTEM_PROMPT_V1 = (
    "You are a retrieval sufficiency controller for a multi-hop question-answering "
    "system. You are given the original question, evidence retrieved so far, and the "
    "search queries already tried. Decide whether the evidence is sufficient to fully "
    "answer the question. If it is not sufficient, propose exactly ONE focused "
    "follow-up search query that would retrieve the specific missing information — "
    "it must be a new query, not a repeat of one already tried. Respond with JSON "
    "only, matching this exact schema: "
    '{"sufficient": true or false, "next_query": string or null, "reason": string}. '
    "When sufficient is true, next_query must be null. Keep reason to one short "
    "sentence."
)

CONTROLLER_USER_TEMPLATE_V1 = (
    "Original question: {question}\n\n"
    "Search queries tried so far:\n{search_history}\n\n"
    "Evidence retrieved so far:\n{evidence_text}\n\n"
    "Decide: is this evidence sufficient to fully answer the original question?"
)

_VERSIONS = {"v1": (CONTROLLER_SYSTEM_PROMPT_V1, CONTROLLER_USER_TEMPLATE_V1)}


def build_controller_prompt(
    question: str,
    evidence_text: str,
    previous_queries: list[str],
    version: str = CONTROLLER_PROMPT_VERSION,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the controller call.

    Raises ValueError for an unknown version rather than silently falling
    back to a default.
    """
    if version not in _VERSIONS:
        raise ValueError(f"unknown controller prompt version: {version!r} (known: {sorted(_VERSIONS)})")
    system_prompt, user_template = _VERSIONS[version]
    search_history = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(previous_queries)) or "(none yet)"
    user_prompt = user_template.format(
        question=question,
        search_history=search_history,
        evidence_text=evidence_text or "(no evidence retrieved yet)",
    )
    return system_prompt, user_prompt
