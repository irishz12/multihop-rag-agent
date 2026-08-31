"""Structural guard for scripts/generate_phase9_charts.py: purely offline
visualization tooling — no Mantle client, no retrieval/model code, no
final_holdout access, only reads already-measured artifacts and writes
only into results/charts/.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_phase9_charts.py"


def test_script_never_constructs_a_mantle_client():
    source = SCRIPT_PATH.read_text()
    assert "MantleClient" not in source


def test_script_never_references_final_holdout_json_directly():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout.json" not in without_comments


def test_script_only_writes_into_the_charts_directory():
    source = SCRIPT_PATH.read_text()
    write_calls = re.findall(r"savefig\((\w+)", source)
    assert write_calls, "expected at least one savefig() call"
    for target in write_calls:
        assert target == "CHARTS_DIR"


def test_color_assignment_is_fixed_not_re_cycled_per_chart():
    """The COLORS dict must be a single module-level mapping reused by
    every chart function — never a fresh/re-ordered palette per plot."""
    source = SCRIPT_PATH.read_text()
    assert source.count("COLORS = {") == 1
    assert re.search(r"^LABELS = \{", source, re.MULTILINE)
    assert source.count("\nLABELS = {") == 1
