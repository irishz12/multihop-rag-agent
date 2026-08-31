"""Structural guard tests for scripts/run_phase9_judge.py: must never
reach final_holdout.json, must never write to any prior-phase output file
other than its own checkpoints, and must never pass a pipeline-identifying
value into `call_judge`.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_phase9_judge.py"

PRIOR_PHASE_OUTPUT_FILENAMES = (
    "router_dataset.json", "router_split.json", "router_thresholds.json", "router_validation_report.json",
    "sequential_router_eval_raw.json", "router_full_dev_eval.json", "sequential_router_report.json",
    "learned_router_dataset.json", "learned_router_model.json", "learned_router_report.json",
)


def test_script_source_never_references_any_prior_phase_output_filename():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    for name in PRIOR_PHASE_OUTPUT_FILENAMES:
        pattern = re.compile(r"(?<![\w./])" + re.escape(name))
        assert not pattern.search(without_comments), f"script must never reference prior-phase output {name}"


def test_script_source_never_references_final_holdout_outside_documentation():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_script_has_no_split_flag():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_never_dumps_raw_environ():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source


def test_script_only_writes_its_own_checkpoint_files():
    """Every .write_text() call in this script must target one of its OWN
    checkpoint artifacts (out_path) — never a prior-phase file."""
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert set(write_calls) == {"out_path"}, f"unexpected write target(s): {set(write_calls) - {'out_path'}}"


def test_script_excludes_null_query_from_judge_scoring():
    """null_query is scored via abstention correctness offline, not the
    open-ended judge — the pipeline-scoring path must filter it out."""
    source = SCRIPT_PATH.read_text()
    assert 'r["question_type"] != "null_query"' in source


def test_script_never_passes_pipeline_identifying_args_to_call_judge():
    """Every call_judge(...) invocation in this script must use only
    positional (question, gold_answer, candidate_answer)-style arguments
    plus prompt_version/pricing kwargs — never a pipeline/route kwarg."""
    source = SCRIPT_PATH.read_text()
    calls = re.findall(r"call_judge\(([^)]*)\)", source, flags=re.DOTALL)
    assert calls, "expected at least one call_judge(...) invocation"
    forbidden_kwargs = ("pipeline=", "route=", "predicted_route=", "retrieval_method=", "model=")
    for call_args in calls:
        for forbidden in forbidden_kwargs:
            assert forbidden not in call_args, f"call_judge() must never receive {forbidden}"
