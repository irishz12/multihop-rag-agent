"""Structural guard for scripts/task_success_phase4_error_analysis.py:
dev-only, zero new LLM/API calls, never touches final_holdout.json, never
overwrites any existing results/*.json (including Phase 2/3's task-success
reports and Phase 3's error-analysis file), only ever writes its own new
output file.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "task_success_phase4_error_analysis.py"


def _source() -> str:
    return SCRIPT_PATH.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_script_only_writes_to_its_own_out_path():
    write_calls = re.findall(r"(\w+)\.write_text\(", _source())
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"


def test_output_filename_is_new_and_distinct_from_prior_artifacts():
    source = _source()
    assert 'OUTPUT_FILE = "results/task_success_hardening_error_analysis_v2.json"' in source
    for protected in (
        '"results/task_success_report.json"',
        '"results/task_success_report_v2.json"',
        '"results/task_success_report_v3.json"',
        '"results/task_success_hardening_error_analysis.json"',
    ):
        assert protected not in source


def test_script_never_references_holdout_in_code():
    code = _without_docstrings_and_comments(_source())
    for marker in ("final_holdout", "phase9_holdout_"):
        assert marker not in code


def test_script_makes_zero_llm_calls():
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval", "run_adaptive_pipeline"):
        assert forbidden not in source


def test_script_has_no_split_flag_and_never_dumps_environ():
    source = _source()
    assert "--split" not in source
    assert "os.environ" not in source


def test_response_structure_sample_uses_a_fixed_documented_seed():
    source = _source()
    assert "SAMPLE_SEED = 2029" in source
    assert "random.Random(SAMPLE_SEED)" in source
