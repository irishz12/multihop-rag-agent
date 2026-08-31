"""Structural guard test for scripts/build_router_dataset.py: it must
never be able to reach final_holdout.json."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "build_router_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_router_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_split_file_constant_is_the_development_split():
    module = _load_module()
    assert module.DEV_SPLIT_FILE == "dev_subset.json"


def test_script_source_never_references_final_holdout_outside_documentation():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_script_has_no_split_flag_that_can_select_a_different_split():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source


def test_script_makes_no_mantle_client():
    """This script computes features only (Qdrant + local embedding/BM25
    models) — it must never construct a MantleClient or import anything
    Mantle-related, so it truly has zero LLM cost."""
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source
    assert "mantle_client" not in source
