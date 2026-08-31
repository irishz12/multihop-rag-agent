"""Structural guard tests for scripts/agentic_smoke_check.py: it must never
be able to reach final_holdout.json, and its default question count stays
in the 5-10 range this phase specifies.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "agentic_smoke_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agentic_smoke_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_split_file_constant_is_the_smoke_split():
    module = _load_module()
    assert module.SMOKE_SPLIT_FILE == "smoke_subset.json"


def test_default_indices_count_is_within_5_to_10():
    module = _load_module()
    assert 5 <= len(module.DEFAULT_INDICES) <= 10


def test_default_indices_are_unique():
    module = _load_module()
    assert len(module.DEFAULT_INDICES) == len(set(module.DEFAULT_INDICES))


def test_script_source_never_references_final_holdout_outside_documentation():
    """Same pattern as tests/test_mantle_smoke_check_guard.py /
    tests/test_eval_harness.py: documentation may explain the guarantee in
    prose; no actual code may reference it."""
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
