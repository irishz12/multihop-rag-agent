"""Structural guard tests for scripts/train_learned_router.py: it must
never be able to reach final_holdout.json, must never write to any prior
Phase 8A/8A.1 output file, and (being purely OFFLINE) must never construct
any live retrieval/embedding/BM25/reranker/Mantle client.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "train_learned_router.py"

PRIOR_PHASE_OUTPUT_FILENAMES = (
    "router_dataset.json", "router_split.json", "router_thresholds.json", "router_validation_report.json",
    "sequential_router_eval_raw.json", "router_full_dev_eval.json", "sequential_router_report.json",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("train_learned_router", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_paths_are_new_files_not_prior_phase_files():
    module = _load_module()
    for attr in ("MODEL_OUTPUT_PATH", "REPORT_OUTPUT_PATH"):
        basename = Path(getattr(module, attr)).name
        assert basename not in PRIOR_PHASE_OUTPUT_FILENAMES


def test_script_source_never_references_any_prior_phase_output_filename():
    """Filename-boundary match (not preceded by a word character or '/')
    so this script's own "learned_router_model.json"/"learned_router_
    report.json" — which end with substrings shared with prior-phase
    filenames — are correctly not flagged."""
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


def test_script_is_offline_no_live_service_clients():
    """Training is purely offline (reads the already-built dataset JSON) —
    it must never construct a Qdrant client, embedding model, BM25 model,
    reranker, or Mantle/GLM client."""
    source = SCRIPT_PATH.read_text()
    forbidden = ("QdrantClient", "EmbeddingModel", "Bm25Model", "Reranker(", "MantleClient", "get_client(")
    for token in forbidden:
        assert token not in source, f"train_learned_router.py must be offline — found {token!r}"


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
    }
    for filename, expected_substring in expectations.items():
        path = root / "results" / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert expected_substring in data.get("purpose", ""), (
            f"{filename}'s 'purpose' field no longer looks like its original phase's — it may have been overwritten"
        )
