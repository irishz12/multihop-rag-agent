"""Structural guard for scripts/compute_task_success.py: dev-only, zero
new LLM/API calls, never touches final_holdout.json or any
phase9_holdout_* artifact, never overwrites an existing results/*.json,
only ever writes its own new results/task_success_report.json. Mirrors
this session's earlier guard-test pattern
(tests/test_context_matched_ablation_guard.py).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT_PATH = ROOT / "scripts" / "compute_task_success.py"

# Every results/*.json filename known to exist before this Task Success work — none may ever
# be a write target of this script.
PROTECTED_EXISTING_FILENAMES = (
    "phase9_sample_report.json", "phase9_holdout_report.json", "phase9_sample.json",
    "phase9_dense_raw.json", "phase9_hybrid_raw.json", "phase9_hybrid_reranker_raw.json",
    "phase9_always_agentic_raw.json", "phase9_adaptive_raw.json",
    "phase9_judge_dense.json", "phase9_judge_hybrid.json", "phase9_judge_hybrid_reranker.json",
    "phase9_judge_always_agentic.json", "phase9_judge_adaptive.json", "phase9_judge_validation.json",
    "phase9_holdout_sample.json", "phase9_holdout_always_agentic_raw.json", "phase9_holdout_adaptive_raw.json",
    "phase9_holdout_judge_always_agentic.json", "phase9_holdout_judge_adaptive.json",
    "learned_router_model.json", "final_holdout_consumed.json", "final_evaluation_manifest.json",
    "phase9_hybrid_reranker_matched_raw.json", "phase9_judge_hybrid_reranker_matched.json",
    "context_matched_ablation_report.json",
    "phase9_hybrid_reranker_matched_full_raw.json", "phase9_judge_hybrid_reranker_matched_full.json",
    "phase9_judge_hybrid_reranker_extended73.json", "phase9_judge_always_agentic_extended73.json",
    "context_matched_ablation_full_report.json",
    # Phase 3 hardening: task_success_report.json is now itself a protected, pre-existing artifact
    # (Phase 2's original) — the hardened script must write to a NEW filename, never overwrite it.
    "task_success_report.json",
    # Phase 4 hardening: task_success_report_v2.json (Phase 3's) is now ALSO protected.
    "task_success_report_v2.json",
    "task_success_hardening_error_analysis.json",
)

_HOLDOUT_MARKERS = ("final_holdout", "phase9_holdout_")


def _source() -> str:
    return SCRIPT_PATH.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_output_filename_is_genuinely_new():
    assert "task_success_report_v3.json" not in PROTECTED_EXISTING_FILENAMES
    # and BOTH earlier reports (Phase 2's original, Phase 3's v2) are explicitly protected,
    # per the "don't overwrite the previous artifact" requirement
    assert "task_success_report.json" in PROTECTED_EXISTING_FILENAMES
    assert "task_success_report_v2.json" in PROTECTED_EXISTING_FILENAMES
    assert 'OUTPUT_FILE = "results/task_success_report_v3.json"' in _source()


def test_script_only_writes_to_its_own_out_path():
    write_calls = re.findall(r"(\w+)\.write_text\(", _source())
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"


def test_script_never_references_holdout_in_code():
    code = _without_docstrings_and_comments(_source())
    for marker in _HOLDOUT_MARKERS:
        assert marker not in code, f"compute_task_success.py's CODE must never reference {marker!r}"


def test_script_has_no_split_flag():
    assert "--split" not in _source()


def test_script_never_dumps_raw_environ():
    assert "os.environ" not in _source()


def test_script_makes_zero_llm_calls():
    """No Mantle client, no judge call, no generation call anywhere —
    this script only reads already-persisted JSON and applies pure
    offline functions."""
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval", "run_adaptive_pipeline"):
        assert forbidden not in source, f"compute_task_success.py must make zero new LLM/API calls — found {forbidden!r}"


def test_script_dev_split_file_constant():
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in _source()


def test_script_reads_only_already_known_development_artifacts():
    """Every *_FILE constant this script reads must be one of the
    already-established development-split artifacts from this session's
    earlier work — never a holdout file, never an unexpected new input."""
    source = _source()
    expected_read_constants = (
        'BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"',
        'AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"',
        'MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"',
    )
    for constant in expected_read_constants:
        assert constant in source
