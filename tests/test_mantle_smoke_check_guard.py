"""Structural guard tests for scripts/mantle_smoke_check.py: it must never
be able to reach final_holdout.json, and it must never be part of normal
(offline, free) pytest collection making a real network call — that
guarantee is tests/test_mantle_live.py's job (opt-in, skipped by default).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "mantle_smoke_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mantle_smoke_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_split_file_constant_is_the_smoke_split():
    module = _load_module()
    assert module.SMOKE_SPLIT_FILE == "smoke_subset.json"


def test_script_source_never_references_final_holdout_outside_documentation():
    """Documentation (docstrings/comments) may explain the guarantee in
    prose; no actual code — no variable value, no path, no CLI choice —
    may reference it. Same pattern as
    tests/test_eval_harness.py::test_script_source_never_references_final_holdout_outside_documentation."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_script_has_no_cli_flag_that_can_select_a_different_split():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_defaults_to_a_small_n():
    """--n must default to something in the 3-5 smoke-question range this
    phase specifies, not the full development or smoke set. Checked via
    source inspection (not by calling main(), which would require live
    models/network)."""
    source = SCRIPT_PATH.read_text()
    match = re.search(r'"--n".*?default=(\d+)', source, re.DOTALL)
    assert match is not None, "expected --n to have an explicit integer default"
    assert 3 <= int(match.group(1)) <= 5


def test_script_never_logs_the_api_key_env_var_name_as_a_value_to_print():
    """The key itself obviously never appears in source (it's read from the
    environment at runtime) — this just double-checks no debug/print
    statement in the script dumps `os.environ` or the client's internals
    wholesale, which could incidentally leak it."""
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source  # config loading goes through mhrag.generation, not raw env dumps
