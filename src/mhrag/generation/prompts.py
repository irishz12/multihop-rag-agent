"""Versioned, deterministic prompt templates for evidence-grounded answer
generation.

Deterministic: no randomness, no timestamp, no per-call variation in the
template text itself (the only per-call inputs are `question` and
`context_text`, both plain string substitutions). Versioned: the prompt
text is pinned to `PROMPT_VERSION`; every `GenerationResult` records which
version produced it (see `mhrag.generation.answer`), and a later phase
comparing retrieval pipelines must hold the model AND this prompt version
fixed across all of them for a fair comparison (per the Phase 6 spec) —
changing the prompt text means bumping the version, not editing `_V1`
in place.

Never mentions or includes ground truth: the template has no slot for a
gold answer, evidence list, or question_type, so there is no place for a
caller to accidentally pass ground truth through even if it tried
(`build_prompt`'s signature only accepts `question` and `context_text`).
"""

from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM_PROMPT_V1 = (
    "You are a careful assistant that answers questions using ONLY the "
    "provided context. Do not use any outside knowledge. If the context "
    "does not contain enough information to answer confidently, say "
    "exactly that the available information is insufficient to answer — "
    "do not guess or speculate. Answer in plain prose. Do not include "
    "citations, source numbers, bracketed references, or a list of "
    "sources in your answer."
)

USER_PROMPT_TEMPLATE_V1 = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."

_VERSIONS = {"v1": (SYSTEM_PROMPT_V1, USER_PROMPT_TEMPLATE_V1)}


def build_prompt(question: str, context_text: str, version: str = PROMPT_VERSION) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the given prompt version.

    Raises ValueError for an unknown version rather than silently falling
    back to a default — a caller requesting a specific version must get
    exactly that version or an explicit error.
    """
    if version not in _VERSIONS:
        raise ValueError(f"unknown prompt version: {version!r} (known: {sorted(_VERSIONS)})")
    system_prompt, user_template = _VERSIONS[version]
    return system_prompt, user_template.format(context=context_text, question=question)
