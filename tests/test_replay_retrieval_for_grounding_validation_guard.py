"""Structural guard for scripts/replay_retrieval_for_grounding_validation.py:
dev-only, zero NEW LLM/API calls (local retrieval/rerank models only — no
MantleClient, no judge, no generation, no controller), never touches
final_holdout.json, NEVER passes Evidence.fact or a gold answer into
retrieval, the query passed to retrieval is always record.query, only
ever writes its own new output file, and is never imported by any
runtime module.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "replay_retrieval_for_grounding_validation.py"

RUNTIME_MODULE_PATHS = [
    Path(__file__).parent.parent / "src" / "mhrag" / "agent" / "loop.py",
    Path(__file__).parent.parent / "src" / "mhrag" / "agent" / "controller.py",
    Path(__file__).parent.parent / "src" / "mhrag" / "generation" / "answer.py",
    Path(__file__).parent.parent / "src" / "mhrag" / "adaptive" / "pipeline.py",
    Path(__file__).parent.parent / "src" / "mhrag" / "routing" / "router.py",
]


def _source() -> str:
    return SCRIPT_PATH.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_script_only_writes_to_its_own_out_path():
    write_calls = re.findall(r"(\w+)\.write_text\(", _source())
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"


def test_output_filename_is_new():
    assert 'OUTPUT_FILE = "results/fact_grounding_replay_raw.json"' in _source()


def test_script_never_references_holdout_in_code():
    code = _without_docstrings_and_comments(_source())
    for marker in ("final_holdout", "phase9_holdout_"):
        assert marker not in code


def test_script_makes_zero_new_llm_calls():
    """Local retrieval/rerank models only — no Mantle client, no judge, no
    generation, no controller, no agent/adaptive orchestration."""
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval",
                       "run_adaptive_pipeline", "call_controller("):
        assert forbidden not in source


def test_script_never_reads_evidence_fact_or_gold_answer():
    """The single most important guard for this script: nothing named
    `.fact`, `evidence_list`, or `gold_answer`/`.answer` (the gold answer
    field) may appear as an actual attribute/key ACCESS in its CODE — the
    only per-question field it may read is `.query`. Checked against code
    with docstrings, comments, AND string literals stripped (this
    script's own JSON "purpose" text legitimately narrates "never
    Evidence.fact" in prose, which must not itself trip the check)."""
    code = _without_docstrings_and_comments(_source())
    code_no_strings = re.sub(r'"[^"]*"', '""', re.sub(r"'[^']*'", "''", code))
    for forbidden in (".fact", "evidence_list", "gold_answer", "record.answer"):
        assert forbidden not in code_no_strings, f"replay script must never access {forbidden!r}"


def test_query_passed_to_retrieval_is_record_query():
    source = _source()
    assert "query = record.query" in source
    # every rerank_hybrid_search call must be passed the local `query` variable, not a
    # differently-named or freshly-constructed string
    calls = re.findall(r"rerank_hybrid_search\(\s*(\w+),", source)
    assert calls, "expected at least one rerank_hybrid_search(...) call"
    assert set(calls) == {"query"}, f"rerank_hybrid_search must always be called with `query`, found: {set(calls)}"


def test_script_has_no_split_flag_and_never_dumps_environ():
    source = _source()
    assert "--split" not in source
    assert "os.environ" not in source


def test_no_runtime_module_imports_this_script():
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists()
        source = path.read_text()
        assert "replay_retrieval_for_grounding_validation" not in source
