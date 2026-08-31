"""Structural guard tests for scripts/run_phase9_benchmark.py: it must
never be able to reach final_holdout.json, must never WRITE to any
prior-phase output file, and must only READ (never overwrite) the frozen
Phase 8A.2 router model artifact the adaptive pipeline depends on.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_phase9_benchmark.py"

# learned_router_model.json is deliberately excluded — legitimately READ (adaptive pipeline).
PRIOR_PHASE_OUTPUT_FILENAMES = (
    "router_dataset.json", "router_split.json", "router_thresholds.json", "router_validation_report.json",
    "sequential_router_eval_raw.json", "router_full_dev_eval.json", "sequential_router_report.json",
    "learned_router_dataset.json", "learned_router_report.json", "adaptive_smoke_comparison.json",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase9_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_split_file_constant_is_the_development_split():
    module = _load_module()
    assert module.DEV_SPLIT_FILE == "dev_subset.json"


def test_router_model_path_points_at_the_frozen_phase_8a2_artifact():
    module = _load_module()
    assert module.LEARNED_ROUTER_MODEL_PATH == "results/learned_router_model.json"


def test_output_filenames_for_every_pipeline_are_new_files():
    module = _load_module()
    for pipeline in module.PIPELINES:
        basename = f"phase9_{pipeline}_raw.json"
        assert basename not in PRIOR_PHASE_OUTPUT_FILENAMES
        assert basename != "learned_router_model.json"


def test_script_only_writes_to_out_path_never_to_the_router_model_path():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"], f"expected the only write_text() call to target out_path, found: {write_calls}"


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


def test_script_includes_null_query_never_filters_it_out():
    """Unlike every earlier router-only script, this one must process ALL
    300 development questions, not just the 265 non-null ones — assert the
    actual CODE (docstrings may explain this in prose) never filters by
    question_type."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "null_query" not in without_comments
    assert "non_null" not in without_comments


def test_prior_phase_output_files_untouched_by_this_phase():
    import json

    root = Path(__file__).parent.parent
    expectations = {
        "router_dataset.json": "router feature dataset",
        "router_split.json": "router_tune / router_validation split",
        "router_thresholds.json": "frozen Stage A heuristic thresholds",
        "router_validation_report.json": "router_validation performance report",
        "sequential_router_eval_raw.json": "evidence-aware sequential router",
        "router_full_dev_eval.json": "Phase 8A direct classifier predictions",
        "sequential_router_report.json": "Phase 8A.1",
        "learned_router_dataset.json": "learned-router training dataset",
        "learned_router_model.json": "frozen deployable LinearModels",
        "learned_router_report.json": "Phase 8A.2 learned two-stage router",
        "adaptive_smoke_comparison.json": "Phase 8B Adaptive vs Always-Agentic",
    }
    for filename, expected_substring in expectations.items():
        path = root / "results" / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert expected_substring in data.get("purpose", ""), (
            f"{filename}'s 'purpose' field no longer looks like its original phase's — it may have been overwritten"
        )
