"""Structural guard for scripts/analyze_phase9_sample.py: purely offline
(no Mantle client, no live call possible), never reaches final_holdout,
never writes to any file but its own output.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "analyze_phase9_sample.py"


def test_script_never_constructs_a_mantle_client():
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source


def test_script_source_never_references_final_holdout_outside_documentation():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_script_only_writes_its_own_output_path():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_script_never_dumps_raw_environ():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source


def test_script_never_guesses_judge_cost():
    """Judge pricing is unverified — the aggregation must report cost as
    null, never a guessed number, matching mhrag.eval.judge's contract."""
    source = SCRIPT_PATH.read_text()
    assert '"total_cost_usd"] = None' in source
