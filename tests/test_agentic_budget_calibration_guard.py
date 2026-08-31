"""Structural guard tests for scripts/agentic_budget_calibration.py: it
must never be able to reach final_holdout.json, and it may only ever run
against the 3 fixed candidate token budgets this phase authorizes.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "agentic_budget_calibration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agentic_budget_calibration", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_split_file_constant_is_the_development_split():
    module = _load_module()
    assert module.DEV_SPLIT_FILE == "dev_subset.json"


def test_token_budgets_are_exactly_the_three_authorized_values():
    module = _load_module()
    assert module.TOKEN_BUDGETS == (3000, 4500, 6000)


def test_budget_flag_choices_are_restricted_to_token_budgets():
    """The --budget CLI flag must not accept any value outside the fixed
    3-way comparison — it should be impossible to sweep a 4th budget."""
    source = SCRIPT_PATH.read_text()
    assert "choices=TOKEN_BUDGETS" in source


def test_script_source_never_references_final_holdout_outside_documentation():
    """Same pattern as tests/test_agentic_smoke_check_guard.py /
    tests/test_mantle_smoke_check_guard.py: documentation may explain the
    guarantee in prose; no actual code may reference it."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_script_has_no_split_flag_that_can_select_a_different_split():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_never_dumps_raw_environ():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source


def test_script_only_changes_max_context_tokens_on_the_base_config():
    """Structural: the only dataclasses.replace call in this script may set
    max_context_tokens — every other AgenticConfig field must come from
    configs/agent.yaml / configs/mantle.yaml unmodified."""
    source = SCRIPT_PATH.read_text()
    replace_calls = re.findall(r"dataclasses\.replace\(([^)]*)\)", source)
    assert replace_calls, "expected exactly one dataclasses.replace call"
    for call in replace_calls:
        assert "max_context_tokens" in call
        assert "max_hops" not in call
        assert "hop_top_k" not in call
        assert "timeout_seconds" not in call
        assert "controller_prompt_version" not in call
        assert "generation_prompt_version" not in call
