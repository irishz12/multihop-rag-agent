"""Structural guard for scripts/select_phase9_holdout_sample.py — the ONE
deliberate exception to this project's "never reach final_holdout" rule.
Asserts the OPPOSITE of every other guard test in this suite: this script
MUST target final_holdout.json, and must NOT be able to read dev_subset.json
instead (which would defeat the whole point of a held-out evaluation).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "select_phase9_holdout_sample.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("select_phase9_holdout_sample", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_split_file_constant_is_final_holdout():
    module = _load_module()
    assert module.HOLDOUT_SPLIT_FILE == "final_holdout.json"


def test_script_never_references_dev_subset_json():
    """This script must be incapable of accidentally scoring the
    DEVELOPMENT split as if it were the holdout — the actual CODE must
    never reference dev_subset.json (docstrings may explain the contrast
    in prose)."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "dev_subset" not in without_comments


def test_script_has_no_split_override_flag():
    """No CLI flag may let a caller redirect this script at a different
    file — HOLDOUT_SPLIT_FILE is the only split it will ever read."""
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source
    assert "add_argument" not in source


def test_script_requires_the_pre_access_manifest_to_already_exist():
    source = SCRIPT_PATH.read_text()
    assert "MANIFEST_PATH" in source
    assert "SystemExit" in source


def test_script_only_writes_its_own_output_path():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_script_never_constructs_a_mantle_client():
    """Selection is offline — no live call needed to pick which qa_ids to evaluate."""
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source


def test_script_reuses_the_same_frozen_seed_as_the_development_sample():
    """Deliberately the SAME seed constant as the dev-sample selector, not
    a new one — no appearance of seed-shopping for a favorable holdout
    sample."""
    from mhrag.eval.phase9_sample import PHASE9_SAMPLE_SEED

    _load_module()
    source = SCRIPT_PATH.read_text()
    assert "PHASE9_SAMPLE_SEED" in source  # imports the shared constant, does not redefine its own
    assert PHASE9_SAMPLE_SEED == 2029
