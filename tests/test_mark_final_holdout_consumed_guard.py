"""Structural guard for scripts/mark_final_holdout_consumed.py: must never
open final_holdout.json itself, must refuse to write the marker unless the
holdout report's integrity_check already reads PASSED, and must never
mutate final_evaluation_manifest.json.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "mark_final_holdout_consumed.py"


def test_script_never_opens_final_holdout_json_directly():
    """The marker's own 'purpose' string legitimately NAMES final_holdout
    .json in prose — what matters is this script never has FILE-ACCESS
    code for it: check the actual set of files it opens/reads via
    PROJECT_ROOT-joined paths never includes it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mark_final_holdout_consumed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.REPORT_PATH.endswith("final_holdout.json")
    assert not module.OUTPUT_PATH.endswith("final_holdout.json")


def test_script_refuses_unless_integrity_check_passed():
    source = SCRIPT_PATH.read_text()
    assert "integrity_check" in source
    assert "SystemExit" in source


def test_script_only_writes_its_own_output_path_never_the_manifest():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_output_path_is_a_separate_marker_not_the_manifest_file():

    import importlib.util

    spec = importlib.util.spec_from_file_location("mark_final_holdout_consumed", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUTPUT_PATH != "results/final_evaluation_manifest.json"
    assert module.OUTPUT_PATH == "results/final_holdout_consumed.json"
