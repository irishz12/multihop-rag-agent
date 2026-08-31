"""Structural guard for scripts/analyze_phase9_holdout.py: purely offline,
must NOT re-open final_holdout.json (gold doc ids come from the already-
persisted phase9_holdout_sample.json), never guesses judge cost, and
performs the promised integrity re-hash check.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "analyze_phase9_holdout.py"


def test_script_never_constructs_a_mantle_client():
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source


def test_script_never_opens_final_holdout_json_directly():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout.json" not in without_comments


def test_script_performs_the_integrity_rehash_check():
    source = SCRIPT_PATH.read_text()
    assert "_verify_manifest_unchanged" in source
    assert "sha1" in source.lower()
    assert "SystemExit" in source


def test_script_only_writes_its_own_output_path():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_script_never_guesses_judge_cost():
    source = SCRIPT_PATH.read_text()
    assert "total_judge_cost = None" in source
