"""Structural guard test for scripts/run_phase8a_full_dev_eval.py: it
reads Phase 8A's output files but must never write to them."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_phase8a_full_dev_eval.py"


def test_output_path_is_new_not_a_phase_8a_file():
    source = SCRIPT_PATH.read_text()
    match = re.search(r'OUTPUT_PATH = "([^"]+)"', source)
    assert match is not None
    assert match.group(1) == "results/router_full_dev_eval.json"


def test_script_never_calls_write_text_on_phase_8a_paths():
    source = SCRIPT_PATH.read_text()
    # every .write_text( call in this script must be on out_path (derived from OUTPUT_PATH), never
    # on a path built from ROUTER_DATASET_PATH / ROUTER_THRESHOLDS_PATH / ROUTER_VALIDATION_REPORT_PATH
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_script_source_never_references_final_holdout_outside_documentation():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments
