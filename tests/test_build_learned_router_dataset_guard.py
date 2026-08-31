"""Structural guard tests for scripts/build_learned_router_dataset.py: it
must never be able to reach final_holdout.json, and it must never
reference (read OR write) any Phase 8A or Phase 8A.1 output file — those
results are preserved as the baseline for Phase 8A.2's 3-way comparison,
never overwritten. The one documented exception is
results/retrieval_eval_development.json, which this script (like Phase
8A's build_router_dataset.py before it) legitimately READS to source
oracle route labels without a new retrieval run.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "build_learned_router_dataset.py"

PRIOR_PHASE_OUTPUT_FILENAMES = (
    "router_dataset.json", "router_split.json", "router_thresholds.json", "router_validation_report.json",
    "sequential_router_eval_raw.json", "router_full_dev_eval.json", "sequential_router_report.json",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_learned_router_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_split_file_constant_is_the_development_split():
    module = _load_module()
    assert module.DEV_SPLIT_FILE == "dev_subset.json"


def test_default_output_is_a_new_file_not_a_prior_phase_file():
    module = _load_module()
    default_output_basename = Path(module.DEFAULT_OUTPUT).name
    assert default_output_basename not in PRIOR_PHASE_OUTPUT_FILENAMES


def test_script_source_never_references_any_prior_phase_output_filename():
    """Documentation may explain the preservation guarantee in prose (the
    module docstring names the prior-phase files it must never touch); no
    actual CODE (docstrings/comments stripped) may reference them. The
    frozen retrieval-eval artifact it legitimately reads is a distinct
    filename from every one of these and is unaffected by this check.

    Matches on a filename boundary (not preceded by a word character or
    "/") so e.g. this script's own "learned_router_dataset.json" — which
    ends with the substring "router_dataset.json" — is correctly NOT
    flagged as a reference to Phase 8A's "router_dataset.json"."""
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


def test_script_has_no_split_flag_that_can_select_a_different_split():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_never_dumps_raw_environ():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source


def test_script_never_calls_any_mantle_or_glm_client():
    """This dataset-building script is explicitly LLM-call-free (Stage 1/2
    are learned from retrieval/rerank diagnostics only) — it must never
    construct a Mantle/GLM client."""
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source
    assert "glm" not in source.lower()


def test_prior_phase_output_files_untouched_by_this_phase():
    """Real-data guard: if the actual prior-phase result files exist in
    this checkout, verify they parse as valid JSON with their expected
    'purpose' field intact — i.e. nothing in this session has clobbered
    them with Phase 8A.2 content."""
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
    }
    for filename, expected_substring in expectations.items():
        path = root / "results" / filename
        if not path.exists():
            continue  # not present in this checkout — nothing to verify
        data = json.loads(path.read_text())
        assert expected_substring in data.get("purpose", ""), (
            f"{filename}'s 'purpose' field no longer looks like its original phase's — it may have been overwritten"
        )
