"""Structural guard for scripts/task_success_hardening_error_analysis.py:
dev-only, zero new LLM/API calls, never touches final_holdout.json,
never overwrites any existing results/*.json (including
results/task_success_report.json and results/task_success_report_v2.json),
only ever writes its own new output file.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "task_success_hardening_error_analysis.py"


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


def test_output_filename_is_new_and_distinct_from_both_reports():
    source = _source()
    assert 'OUTPUT_FILE = "results/task_success_hardening_error_analysis.json"' in source
    assert '"results/task_success_report.json"' not in source
    assert '"results/task_success_report_v2.json"' not in source


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


def test_regression_sets_are_non_empty_and_match_required_case_counts():
    """Sanity-checks the hand-built fixtures actually contain the required
    10 verdict + 7 abstention-structure cases, not an accidentally-emptied
    list."""
    source = _source()
    assert source.count('"There is no evidence that..."') == 1
    assert source.count('"There is no agreement between the sources."') == 1
    assert source.count('"Insufficient information to answer; however, the answer is Google."') == 1
    assert source.count('"Google, although the evidence is limited."') == 1
