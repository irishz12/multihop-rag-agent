"""Structural guard tests for the AUDIT ABLATION scripts
(scripts/run_phase9_context_matched_ablation.py,
scripts/run_context_matched_judge.py, scripts/analyze_context_matched_ablation.py):
must never be able to reach final_holdout.json, must never write to any
existing results/*.json artifact, and (for the judge script) must never
pass a pipeline-identifying value into call_judge. Same pattern as
tests/test_phase9_benchmark_guard.py / tests/test_phase9_judge_guard.py,
scoped to the new dev-only ablation scripts.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ABLATION_SCRIPT = ROOT / "scripts" / "run_phase9_context_matched_ablation.py"
JUDGE_SCRIPT = ROOT / "scripts" / "run_context_matched_judge.py"
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyze_context_matched_ablation.py"

NEW_SCRIPTS = (ABLATION_SCRIPT, JUDGE_SCRIPT, ANALYSIS_SCRIPT)

# Every results/*.json filename that existed BEFORE this ablation was added —
# frozen list, must never be a write target of any new script.
EXISTING_RESULT_FILENAMES = (
    "benchmark_manifest.json", "dev_subset.json", "retrieval_eval_development.json",
    "retrieval_eval_development_superseded_qdrant_native_rrf.json",
    "phase9_sample_report.json", "phase9_holdout_report.json", "phase9_sample.json",
    "phase9_dense_raw.json", "phase9_hybrid_raw.json", "phase9_hybrid_reranker_raw.json",
    "phase9_always_agentic_raw.json", "phase9_adaptive_raw.json",
    "phase9_judge_dense.json", "phase9_judge_hybrid.json", "phase9_judge_hybrid_reranker.json",
    "phase9_judge_always_agentic.json", "phase9_judge_adaptive.json", "phase9_judge_validation.json",
    "phase9_holdout_sample.json", "phase9_holdout_always_agentic_raw.json", "phase9_holdout_adaptive_raw.json",
    "phase9_holdout_judge_always_agentic.json", "phase9_holdout_judge_adaptive.json",
    "learned_router_model.json", "final_holdout_consumed.json", "final_evaluation_manifest.json",
    "router_dataset.json", "router_split.json", "router_thresholds.json", "router_validation_report.json",
    "sequential_router_eval_raw.json", "router_full_dev_eval.json", "sequential_router_report.json",
    "learned_router_dataset.json", "learned_router_report.json", "adaptive_smoke_comparison.json",
    "agentic_budget_calibration_3000.json", "agentic_budget_calibration_4500.json",
    "agentic_budget_calibration_6000.json", "agentic_budget_calibration_decision.json",
    "agentic_smoke_check.json", "mantle_smoke_check.json",
)

NEW_OUTPUT_FILENAMES = (
    "phase9_hybrid_reranker_matched_raw.json",
    "phase9_judge_hybrid_reranker_matched.json",
    "context_matched_ablation_report.json",
)


def _source(path: Path) -> str:
    return path.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_new_ablation_output_filenames_are_genuinely_new():
    """The three filenames this ablation introduces must not collide with
    any filename that already existed in results/ before this ablation."""
    for name in NEW_OUTPUT_FILENAMES:
        assert name not in EXISTING_RESULT_FILENAMES, f"{name} collides with a pre-existing results/ artifact"


def test_new_scripts_never_reference_final_holdout():
    for path in NEW_SCRIPTS:
        code = _without_docstrings_and_comments(_source(path))
        assert "final_holdout" not in code, f"{path.name} must never reference final_holdout in executable code"


def test_new_scripts_have_no_split_flag():
    for path in NEW_SCRIPTS:
        assert "--split" not in _source(path), f"{path.name} must not accept a --split flag"


def test_new_scripts_never_dump_raw_environ():
    for path in NEW_SCRIPTS:
        assert "os.environ" not in _source(path), f"{path.name} must never dump os.environ"


def test_ablation_and_judge_scripts_dev_split_file_constant():
    source = _source(ABLATION_SCRIPT)
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in source, "ablation script must read only the DEVELOPMENT split"


def test_ablation_script_only_writes_to_its_own_out_path():
    source = _source(ABLATION_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"expected the only write_text() call to target out_path, found: {write_calls}"
    assert 'OUTPUT_FILE = "results/phase9_hybrid_reranker_matched_raw.json"' in source


def test_judge_script_only_writes_to_its_own_out_path():
    source = _source(JUDGE_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"expected the only write_text() call to target out_path, found: {write_calls}"
    assert 'OUTPUT_FILE = "results/phase9_judge_hybrid_reranker_matched.json"' in source


def test_judge_script_excludes_null_query_from_judge_scoring():
    source = _source(JUDGE_SCRIPT)
    assert 'r["question_type"] != "null_query"' in source


def test_judge_script_never_passes_pipeline_identifying_args_to_call_judge():
    source = _source(JUDGE_SCRIPT)
    calls = re.findall(r"call_judge\(([^)]*)\)", source, flags=re.DOTALL)
    assert calls, "expected at least one call_judge(...) invocation"
    forbidden_kwargs = ("pipeline=", "route=", "predicted_route=", "retrieval_method=", "model=")
    for call_args in calls:
        for forbidden in forbidden_kwargs:
            assert forbidden not in call_args, f"call_judge() must never receive {forbidden}"


def test_analysis_script_is_offline_no_mantle_client_import():
    """The analysis script must never construct a MantleClient or import
    call_judge/generate_answer — it only reads already-written JSON."""
    source = _source(ANALYSIS_SCRIPT)
    assert "MantleClient" not in source
    assert "call_judge" not in source
    assert "generate_answer" not in source


def test_analysis_script_only_writes_its_own_report():
    source = _source(ANALYSIS_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"expected the only write_text() call to target out_path, found: {write_calls}"
    assert 'OUTPUT_FILE = "results/context_matched_ablation_report.json"' in source


def test_ablation_script_reads_the_frozen_50_question_sample_not_all_300():
    """Must restrict to results/phase9_sample.json's qa_ids, never process
    the full 300-question development population the way run_phase9_benchmark.py does."""
    source = _source(ABLATION_SCRIPT)
    assert 'SAMPLE_FILE = "results/phase9_sample.json"' in source
    assert "slice_records = [by_qa_id[q] for q in sample_qa_ids]" in source
