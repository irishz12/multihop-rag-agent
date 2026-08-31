"""Structural guard tests for scripts/run_sequential_router_eval.py: it
must never be able to reach final_holdout.json, and it must never
reference (read OR write) any Phase 8A output file — Phase 8A's results
are preserved as the baseline for comparison, never overwritten.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_sequential_router_eval.py"

PHASE_8A_OUTPUT_FILENAMES = (
    "router_dataset.json", "router_split.json", "router_thresholds.json", "router_validation_report.json",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_sequential_router_eval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_split_file_constant_is_the_development_split():
    module = _load_module()
    assert module.DEV_SPLIT_FILE == "dev_subset.json"


def test_default_output_is_a_new_file_not_a_phase_8a_file():
    module = _load_module()
    assert module.DEFAULT_OUTPUT not in PHASE_8A_OUTPUT_FILENAMES
    assert all(name not in module.DEFAULT_OUTPUT for name in PHASE_8A_OUTPUT_FILENAMES)


def test_script_source_never_references_any_phase_8a_output_filename():
    """Documentation may explain the preservation guarantee in prose (the
    module docstring names the Phase 8A files it must never touch); no
    actual CODE (docstrings/comments stripped) may reference them."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    for name in PHASE_8A_OUTPUT_FILENAMES:
        assert name not in without_comments, f"script must never reference Phase 8A's {name}"


def test_script_source_never_references_final_holdout_outside_documentation():
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


def test_phase_8a_output_files_untouched_by_this_phase(tmp_path):
    """Real-data guard: if the actual Phase 8A result files exist in this
    checkout, verify they parse as valid JSON with their expected Phase 8A
    'purpose' field intact — i.e. nothing in this session has clobbered
    them with Phase 8A.1 content."""
    import json

    root = Path(__file__).parent.parent
    expectations = {
        "router_dataset.json": "router feature dataset",
        "router_split.json": "router_tune / router_validation split",
        "router_thresholds.json": "frozen Stage A heuristic thresholds",
        "router_validation_report.json": "router_validation performance report",
    }
    for filename, expected_substring in expectations.items():
        path = root / "results" / filename
        if not path.exists():
            continue  # not present in this checkout — nothing to verify
        data = json.loads(path.read_text())
        assert expected_substring in data.get("purpose", ""), (
            f"{filename}'s 'purpose' field no longer looks like Phase 8A's — it may have been overwritten"
        )
