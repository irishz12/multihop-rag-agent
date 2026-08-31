"""Structural guard for scripts/run_phase9_holdout_judge.py: restricted to
agentic_multi_hop/adaptive_rag, reads only already-completed holdout
checkpoints (never final_holdout.json directly), never passes a
pipeline-identifying value into call_judge, and only writes its own
checkpoint files.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_phase9_holdout_judge.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase9_holdout_judge", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipelines_restricted_to_agentic_multi_hop_and_adaptive_rag_only():
    module = _load_module()
    assert module.PIPELINES == ("agentic_multi_hop", "adaptive_rag")


def test_script_never_opens_final_holdout_json_directly():
    """Judge scoring reads the ALREADY-COMPLETED raw checkpoint, never
    final_holdout.json itself — the actual CODE must not reference it
    (docstrings may explain the contrast in prose)."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout.json" not in without_comments


def test_script_excludes_null_query_from_judge_scoring():
    source = SCRIPT_PATH.read_text()
    assert 'r["question_type"] != "null_query"' in source


def test_script_only_writes_its_own_checkpoint_files():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert set(write_calls) == {"out_path"}


def test_output_path_is_holdout_specific():
    source = SCRIPT_PATH.read_text()
    assert 'f"phase9_holdout_judge_{args.pipeline}.json"' in source


def test_script_never_passes_pipeline_identifying_args_to_call_judge():
    source = SCRIPT_PATH.read_text()
    calls = re.findall(r"call_judge\(([^)]*)\)", source, flags=re.DOTALL)
    assert calls
    forbidden_kwargs = ("pipeline=", "route=", "predicted_route=", "retrieval_method=", "model=")
    for call_args in calls:
        for forbidden in forbidden_kwargs:
            assert forbidden not in call_args


def test_script_has_no_split_override_flag():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_never_dumps_raw_environ():
    source = SCRIPT_PATH.read_text()
    assert "os.environ" not in source
