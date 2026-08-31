"""Structural guard for scripts/run_phase9_holdout_benchmark.py: MUST
target final_holdout.json (the deliberate exception), MUST restrict
PIPELINES to only agentic_multi_hop/adaptive_rag (never re-running Dense/
Hybrid/Hybrid+Reranker against holdout), and must only write its own
checkpoint files.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_phase9_holdout_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase9_holdout_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_split_file_constant_is_final_holdout():
    module = _load_module()
    assert module.HOLDOUT_SPLIT_FILE == "final_holdout.json"


def test_pipelines_restricted_to_agentic_multi_hop_and_adaptive_rag_only():
    module = _load_module()
    assert module.PIPELINES == ("agentic_multi_hop", "adaptive_rag")
    assert "dense" not in module.PIPELINES
    assert "hybrid" not in module.PIPELINES
    assert "hybrid_reranker" not in module.PIPELINES


def test_script_never_references_dev_subset_json():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "dev_subset" not in without_comments


def test_script_never_writes_to_the_development_sample_checkpoints():
    """Every write_text() target must be derived from `output_path`
    (holdout-specific), never one of the dev-sample's phase9_{pipeline}_raw
    filenames."""
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_output_path_is_holdout_specific_not_the_dev_sample_filename():
    source = SCRIPT_PATH.read_text()
    assert 'f"results/phase9_holdout_{args.pipeline}_raw.json"' in source


def test_script_requires_the_holdout_sample_selection_to_already_exist():
    source = SCRIPT_PATH.read_text()
    assert "HOLDOUT_SAMPLE_PATH" in source
    assert "SystemExit" in source


def test_script_has_no_split_override_flag():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_never_dumps_raw_environ():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source
