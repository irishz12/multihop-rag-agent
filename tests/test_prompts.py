"""Prompt template tests: determinism, versioning, and that no ground-truth
slot exists for a caller to accidentally fill in."""

from __future__ import annotations

import inspect

import pytest

from mhrag.generation.prompts import PROMPT_VERSION, build_prompt


def test_build_prompt_is_deterministic():
    a = build_prompt("What year?", "Some context text.")
    b = build_prompt("What year?", "Some context text.")
    assert a == b


def test_build_prompt_includes_question_and_context():
    system, user = build_prompt("What year was it founded?", "The company was founded in 1999.")
    assert "What year was it founded?" in user
    assert "The company was founded in 1999." in user


def test_build_prompt_instructs_no_outside_knowledge_and_no_citations():
    system, _ = build_prompt("q", "c")
    assert "outside knowledge" in system.lower()
    assert "citations" in system.lower() or "citation" in system.lower()


def test_build_prompt_instructs_insufficient_information_handling():
    system, _ = build_prompt("q", "c")
    assert "insufficient" in system.lower()


def test_default_prompt_version_matches_module_constant():
    system_default, user_default = build_prompt("q", "c")
    system_explicit, user_explicit = build_prompt("q", "c", version=PROMPT_VERSION)
    assert (system_default, user_default) == (system_explicit, user_explicit)


def test_unknown_prompt_version_raises():
    with pytest.raises(ValueError, match="unknown prompt version"):
        build_prompt("q", "c", version="v999-does-not-exist")


def test_build_prompt_signature_has_no_ground_truth_parameter():
    """Structural guarantee: there is no `answer`, `evidence`, `evidence_list`,
    or `question_type` parameter for a caller to pass ground truth through,
    even by mistake."""
    params = set(inspect.signature(build_prompt).parameters)
    assert params == {"question", "context_text", "version"}
    for forbidden in ("answer", "evidence", "evidence_list", "question_type", "gold"):
        assert forbidden not in params
