"""Structural guard test for scripts/analyze_sequential_router_eval.py: it
reads Phase 8A output files but must never write to them, and never
reaches final_holdout."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "analyze_sequential_router_eval.py"


def test_output_path_is_new_not_a_phase_8a_file():
    source = SCRIPT_PATH.read_text()
    match = re.search(r'OUTPUT_PATH = "([^"]+)"', source)
    assert match is not None
    assert match.group(1) == "results/sequential_router_report.json"


def test_script_never_calls_write_text_on_phase_8a_paths():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_script_source_never_references_final_holdout_outside_documentation():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments
