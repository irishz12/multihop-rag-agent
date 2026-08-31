"""Structural guard tests for the FULL-POPULATION SCALE-UP audit ablation
scripts (scripts/run_phase9_context_matched_ablation_full.py,
scripts/run_context_matched_judge_full.py,
scripts/run_extended_baseline_agentic_judge.py,
scripts/analyze_context_matched_ablation_full.py): must never reach
final_holdout.json, must never write to any existing results/*.json
artifact (including the two frozen judge files this scale-up reads to
compute eligibility/union), and must never process the null_query stratum
or the full 300-question development population — only the eligible
(non-null, Agentic-covered) subset.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
ABLATION_SCRIPT = ROOT / "scripts" / "run_phase9_context_matched_ablation_full.py"
JUDGE_SCRIPT = ROOT / "scripts" / "run_context_matched_judge_full.py"
EXTENDED_JUDGE_SCRIPT = ROOT / "scripts" / "run_extended_baseline_agentic_judge.py"
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyze_context_matched_ablation_full.py"

NEW_SCRIPTS = (ABLATION_SCRIPT, JUDGE_SCRIPT, EXTENDED_JUDGE_SCRIPT, ANALYSIS_SCRIPT)

# Every results/*.json filename that existed BEFORE this scale-up was added (the pre-existing
# frozen corpus PLUS the original 50-question ablation's own three artifacts) — none of these
# may ever be a write target of any new script here.
PROTECTED_EXISTING_FILENAMES = (
    "phase9_sample_report.json", "phase9_holdout_report.json", "phase9_sample.json",
    "phase9_dense_raw.json", "phase9_hybrid_raw.json", "phase9_hybrid_reranker_raw.json",
    "phase9_always_agentic_raw.json", "phase9_adaptive_raw.json",
    "phase9_judge_dense.json", "phase9_judge_hybrid.json", "phase9_judge_hybrid_reranker.json",
    "phase9_judge_always_agentic.json", "phase9_judge_adaptive.json", "phase9_judge_validation.json",
    "phase9_holdout_sample.json", "phase9_holdout_always_agentic_raw.json", "phase9_holdout_adaptive_raw.json",
    "phase9_holdout_judge_always_agentic.json", "phase9_holdout_judge_adaptive.json",
    "learned_router_model.json", "final_holdout_consumed.json", "final_evaluation_manifest.json",
    # the original 50-question ablation's own outputs — must remain untouched by the scale-up
    "phase9_hybrid_reranker_matched_raw.json", "phase9_judge_hybrid_reranker_matched.json",
    "context_matched_ablation_report.json",
)

NEW_OUTPUT_FILENAMES = (
    "phase9_hybrid_reranker_matched_full_raw.json",
    "phase9_judge_hybrid_reranker_matched_full.json",
    "phase9_judge_hybrid_reranker_extended73.json",
    "phase9_judge_always_agentic_extended73.json",
    "context_matched_ablation_full_report.json",
)


def _source(path: Path) -> str:
    return path.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_new_output_filenames_are_genuinely_new():
    for name in NEW_OUTPUT_FILENAMES:
        assert name not in PROTECTED_EXISTING_FILENAMES, f"{name} collides with a protected existing artifact"


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


def test_ablation_full_script_dev_split_file_constant():
    source = _source(ABLATION_SCRIPT)
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in source


def test_ablation_full_script_only_writes_to_its_own_out_path():
    source = _source(ABLATION_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"
    assert 'OUTPUT_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"' in source


def test_ablation_full_script_eligibility_excludes_null_query():
    """Eligibility must be derived by filtering question_type != null_query
    from the Agentic raw file, never by processing the full 300-question
    development population."""
    source = _source(ABLATION_SCRIPT)
    assert 'question_type"] != "null_query"' in source
    assert "load_qa_records(dev_path)" in source  # loads dev split for query text, but eligibility itself comes from the agentic raw file
    assert "range(1, 301)" not in source and "population = all_records" not in source


def test_judge_full_script_only_writes_to_its_own_out_path():
    source = _source(JUDGE_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"
    assert 'OUTPUT_FILE = "results/phase9_judge_hybrid_reranker_matched_full.json"' in source


def test_judge_full_script_excludes_null_query():
    source = _source(JUDGE_SCRIPT)
    assert 'r["question_type"] != "null_query"' in source


def test_judge_full_script_never_passes_pipeline_identifying_args_to_call_judge():
    source = _source(JUDGE_SCRIPT)
    calls = re.findall(r"call_judge\(([^)]*)\)", source, flags=re.DOTALL)
    assert calls
    forbidden_kwargs = ("pipeline=", "route=", "predicted_route=", "retrieval_method=", "model=")
    for call_args in calls:
        for forbidden in forbidden_kwargs:
            assert forbidden not in call_args


def test_extended_judge_script_only_writes_to_its_two_extension_files():
    """CRITICAL: this script must NEVER write to phase9_judge_hybrid_reranker.json
    or phase9_judge_always_agentic.json (the original, frozen judge files) —
    only to the two brand-new *_extended73.json files."""
    source = _source(EXTENDED_JUDGE_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert set(write_calls) == {"out_path"}, f"unexpected write target(s): {set(write_calls)}"
    assert 'results/phase9_judge_hybrid_reranker_extended73.json' in source
    assert 'results/phase9_judge_always_agentic_extended73.json' in source
    # the ORIGINAL judge filenames must appear only as *_file constants opened for reading, never assigned to out_path
    assert '"raw_file":' in source and '"existing_judge_file":' in source and '"extension_output_file":' in source


def test_extended_judge_script_never_passes_pipeline_identifying_args_to_call_judge():
    source = _source(EXTENDED_JUDGE_SCRIPT)
    calls = re.findall(r"call_judge\(([^)]*)\)", source, flags=re.DOTALL)
    assert calls
    forbidden_kwargs = ("pipeline=", "route=", "predicted_route=", "retrieval_method=", "model=")
    for call_args in calls:
        for forbidden in forbidden_kwargs:
            assert forbidden not in call_args


def test_extended_judge_script_computes_a_set_difference_never_rejudges_existing():
    """Must skip qa_ids already present in the existing judge file — never
    re-call the paid judge for a question that's already judged."""
    source = _source(EXTENDED_JUDGE_SCRIPT)
    assert "already_judged_ids" in source
    assert "already_judged_ids" in source.split("to_judge_ids")[0] or "not in already_judged_ids" in source


def test_analysis_full_script_is_offline_no_mantle_client_import():
    source = _source(ANALYSIS_SCRIPT)
    assert "MantleClient" not in source
    assert "call_judge" not in source
    assert "generate_answer" not in source


def test_analysis_full_script_only_writes_its_own_report():
    source = _source(ANALYSIS_SCRIPT)
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"
    assert 'OUTPUT_FILE = "results/context_matched_ablation_full_report.json"' in source


def test_analysis_full_script_reads_original_judge_files_read_only():
    """The two original judge files must appear only as *_load(...) read
    targets (via _judge_union), never as a write path."""
    source = _source(ANALYSIS_SCRIPT)
    assert 'BASELINE_JUDGE_FILE = "results/phase9_judge_hybrid_reranker.json"' in source
    assert 'AGENTIC_JUDGE_FILE = "results/phase9_judge_always_agentic.json"' in source
    # neither constant name is ever the argument to .write_text(
    write_lines = [line for line in source.splitlines() if ".write_text(" in line]
    for line in write_lines:
        assert "BASELINE_JUDGE_FILE" not in line
        assert "AGENTIC_JUDGE_FILE" not in line
