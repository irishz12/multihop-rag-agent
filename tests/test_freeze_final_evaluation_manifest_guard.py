"""Structural guard for scripts/freeze_final_evaluation_manifest.py: must
never read final_holdout.json or dev_subset.json itself (only config/code/
artifact files), must never construct a Mantle client, and must write only
its own output.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "freeze_final_evaluation_manifest.py"


def test_script_never_constructs_a_mantle_client():
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source


def test_script_never_opens_final_holdout_or_dev_subset_as_a_data_file():
    """The manifest's 'purpose' string legitimately NAMES final_holdout.json
    in prose (explaining the ordering guarantee) — what must never exist is
    actual file-access CODE naming it: a literal 'data/processed/...json'
    style path, or the JSON filename immediately preceded by '/' or a
    quote used as a path argument. Checked via FROZEN_FILES (the only list
    of paths this script actually opens) never containing either name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_final_evaluation_manifest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path in module.FROZEN_FILES:
        assert "final_holdout" not in path
        assert "dev_subset" not in path


def test_script_only_writes_its_own_output_path():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"(\w+)\.write_text\(", source)
    assert write_calls == ["out_path"]


def test_manifest_status_starts_not_yet_accessed():
    source = SCRIPT_PATH.read_text()
    assert '"NOT_YET_ACCESSED"' in source


def test_frozen_manifest_artifact_reflects_not_yet_accessed_at_generation_time():
    """Real-data check: if the manifest file exists in this checkout, its
    status field must literally be the pre-access sentinel — a later
    'consumed' marker lives in a SEPARATE file
    (results/final_holdout_consumed.json), never by mutating this one."""
    import json

    path = Path(__file__).parent.parent / "results" / "final_evaluation_manifest.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["final_holdout_access_status"] == "NOT_YET_ACCESSED"
