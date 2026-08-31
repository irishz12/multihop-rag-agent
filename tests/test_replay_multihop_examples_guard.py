"""Structural guard for scripts/replay_multihop_examples.py — written and
verified BEFORE this script is ever executed against a live API. Covers:

 1. Selected-qa-id restriction: qa_ids come ONLY from
    results/multihop_success_analysis.json's `selected_example_qa_ids`
    field — no CLI flag, no other input path.
 2. Dev-only / no holdout access: hardcoded dev split, no --split flag, no
    holdout marker anywhere in the script.
 3. Replay output isolation: writes ONLY results/multihop_examples_replay.json,
    never any existing results/*.json.
 4. Uses the real, unmodified `run_agentic_retrieval` — never reimplements
    the loop or bypasses its hard limits.
 5. At most 5 qa_ids are ever replayed in one invocation.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "replay_multihop_examples.py"
RESULTS_DIR = Path(__file__).parent.parent / "results"

_PHASE6A_NEW_FILENAMES = {"multihop_success_analysis.json", "multihop_examples_replay.json"}
PROTECTED_EXISTING_FILENAMES = tuple(
    sorted(p.name for p in RESULTS_DIR.glob("*.json") if p.name not in _PHASE6A_NEW_FILENAMES)
)


def _source() -> str:
    return SCRIPT_PATH.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_qa_ids_come_only_from_the_selection_artifact():
    source = _source()
    assert 'SELECTION_FILE = "results/multihop_success_analysis.json"' in source
    assert 'selected_qa_ids = selection["selected_example_qa_ids"]' in source
    assert "argparse" not in source
    assert "--qa-id" not in source and "--qa_ids" not in source


def test_dev_split_hardcoded_no_holdout_marker():
    source = _source()
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in source
    code = _without_docstrings_and_comments(source)
    for marker in ("final_holdout", "phase9_holdout_", "final_evaluation_manifest", "final_holdout_consumed"):
        assert marker not in code


def test_missing_qa_id_raises_instead_of_falling_back():
    source = _source()
    assert "raise SystemExit" in source
    assert "not found in {DEV_SPLIT_FILE}" in source


def test_script_only_writes_to_its_own_out_path():
    write_calls = re.findall(r"(\w+)\.write_text\(", _source())
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"


def test_output_filename_is_new_and_protected_filenames_never_write_targets():
    source = _source()
    assert 'OUTPUT_FILE = "results/multihop_examples_replay.json"' in source
    write_lines = [line.strip() for line in source.splitlines() if ".write_text(" in line]
    for line in write_lines:
        for protected in PROTECTED_EXISTING_FILENAMES:
            assert protected not in line, f"write_text() call must never target {protected!r}: {line}"


def test_script_uses_the_real_unmodified_agentic_loop():
    source = _source()
    assert "from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval" in source
    assert "def run_agentic_retrieval" not in source  # no local reimplementation
    assert "class AgenticConfig" not in source


def test_at_most_five_qa_ids_asserted():
    source = _source()
    assert "assert 1 <= len(selected_qa_ids) <= 5" in source


def test_checkpointed_per_question_like_phase9_pattern():
    """Must write after every single question, not just at the end, so an
    interrupted run never repeats a paid call."""
    source = _source()
    for_loop_idx = source.index("for record in records:")
    write_idx = source.index("out_path.write_text(", for_loop_idx)
    assert write_idx > for_loop_idx, "checkpoint write must happen INSIDE the per-question loop"
