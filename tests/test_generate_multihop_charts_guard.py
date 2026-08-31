"""Structural guard for scripts/generate_multihop_charts.py: zero LLM
calls, reads only results/multihop_success_analysis.json, writes only its
own two new chart files, never touches any filename generate_phase9_charts.py
already produces."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_multihop_charts.py"

EXISTING_CHART_FILENAMES = (
    "answer_quality.png", "cost_per_query.png", "dev_vs_holdout.png", "evidence_coverage.png",
    "hop_performance.png", "latency_per_query.png", "query_type_performance.png",
)


def _source() -> str:
    return SCRIPT_PATH.read_text()


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_script_makes_zero_llm_calls():
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval", "requests."):
        assert forbidden not in source


def test_script_reads_only_the_offline_analysis_artifact():
    source = _source()
    assert 'INPUT_FILE = "results/multihop_success_analysis.json"' in source
    assert "final_holdout" not in source


def test_script_never_writes_an_existing_chart_filename():
    source = _source()
    write_lines = [line for line in source.splitlines() if "savefig" in line]
    quoted_filenames = {m for line in write_lines for m in re.findall(r'"([^"]+\.png)"', line)}
    for existing in EXISTING_CHART_FILENAMES:
        assert existing not in quoted_filenames, f"must never write over an existing Phase 9 chart: {existing}"
    assert any("multihop_evidence_coverage.png" in line for line in write_lines)
    assert any("multihop_question_type_breakdown.png" in line for line in write_lines)
