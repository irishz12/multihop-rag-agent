"""Structural guard for scripts/compute_fact_grounding.py — covers every
requirement in the Phase 5B brief's leakage/holdout-safety section:

 1. Evidence.fact is consumed only by evaluation code.
 2. No runtime module imports fact_grounding.
 3. The script never passes Evidence.fact/gold_fact/evidence_fact into retrieval.
 4. The retrieval query is exactly record.query.
 5. No fact text is concatenated into a query.
 6. No LLM/API call exists in the implementation.
 7. The script is development-only.
 8. Holdout paths cannot be selected (no CLI flag, hardcoded dev split).
 9. Existing result artifacts cannot be overwritten.
10. Phase 5A artifacts cannot be overwritten.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "compute_fact_grounding.py"
SRC_ROOT = Path(__file__).parent.parent / "src" / "mhrag"

RUNTIME_MODULE_PATHS = [
    SRC_ROOT / "routing" / "router.py",
    SRC_ROOT / "adaptive" / "pipeline.py",
    SRC_ROOT / "agent" / "loop.py",
    SRC_ROOT / "agent" / "controller.py",
    SRC_ROOT / "generation" / "answer.py",
    SRC_ROOT / "generation" / "context.py",
    SRC_ROOT / "generation" / "prompts.py",
    SRC_ROOT / "retrieval" / "rerank.py",
]

# Artifacts that must never be a write target of this script — every existing results/*.json
# this session's work has produced or relied on, PLUS the three Phase 5A validation artifacts.
PROTECTED_EXISTING_FILENAMES = (
    "phase9_hybrid_reranker_raw.json", "phase9_always_agentic_raw.json",
    "phase9_hybrid_reranker_matched_full_raw.json", "phase9_judge_hybrid_reranker.json",
    "phase9_judge_always_agentic.json", "phase9_judge_hybrid_reranker_matched_full.json",
    "task_success_report.json", "task_success_report_v2.json", "task_success_report_v3.json",
    "context_matched_ablation_report.json", "context_matched_ablation_full_report.json",
    # Phase 5A validation artifacts — must never be overwritten by Phase 5B
    "fact_grounding_chunk_survival.json", "fact_grounding_replay_raw.json",
    "fact_grounding_replay_fidelity.json",
)


def _source() -> str:
    return SCRIPT_PATH.read_text()


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def _code_without_strings(code: str) -> str:
    return re.sub(r'"[^"]*"', '""', re.sub(r"'[^']*'", "''", code))


def test_script_exists():
    assert SCRIPT_PATH.exists()


# --- 9 & 10: no overwrite of existing or Phase 5A artifacts --------------------------------


def test_script_only_writes_to_its_own_out_path():
    write_calls = re.findall(r"(\w+)\.write_text\(", _source())
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"


def test_output_filename_is_new_and_protected_filenames_are_never_write_targets():
    source = _source()
    assert 'OUTPUT_FILE = "results/fact_grounding_report.json"' in source
    assert "fact_grounding_report.json" not in PROTECTED_EXISTING_FILENAMES
    write_lines = [line.strip() for line in source.splitlines() if ".write_text(" in line]
    for line in write_lines:
        for protected in PROTECTED_EXISTING_FILENAMES:
            assert protected not in line, f"write_text() call must never target {protected!r}: {line}"


def test_phase5a_artifacts_are_read_only_constants_not_write_targets():
    source = _source()
    assert 'PHASE5A_REPLAY_FILE = "results/fact_grounding_replay_raw.json"' in source
    assert 'MATCHED_ORIGINAL_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"' in source
    # both constants must be used only inside a .read_text()/json.loads() call context, never write_text
    for const_name in ("PHASE5A_REPLAY_FILE", "MATCHED_ORIGINAL_RAW_FILE", "AGENTIC_RAW_FILE"):
        assert f"{const_name}))" not in source.replace(" ", "")  # loose sanity check, real check is write-target scan above


# --- 7 & 8: development-only, holdout unreachable ------------------------------------------


def test_script_dev_split_file_constant_and_no_split_flag():
    source = _source()
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in source
    assert "--split" not in source
    assert "argparse" not in source  # no CLI flags at all — nothing that could redirect input


def test_script_never_references_holdout_in_code():
    code = _without_docstrings_and_comments(_source())
    for marker in ("final_holdout", "phase9_holdout_", "final_evaluation_manifest", "final_holdout_consumed"):
        assert marker not in code, f"script CODE must never reference {marker!r}"


# --- 6: zero LLM/API calls ------------------------------------------------------------------


def test_script_makes_zero_llm_calls():
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval",
                       "run_adaptive_pipeline", "call_controller("):
        assert forbidden not in source, f"must be zero LLM/API calls — found {forbidden!r}"


def test_script_never_dumps_raw_environ():
    assert "os.environ" not in _source()


# --- 1, 3, 5: Evidence.fact / gold data never reaches retrieval ----------------------------


def test_evidence_fact_used_only_via_gold_facts_helper_never_passed_to_retrieval():
    """The only place `.fact` may appear in this script's CODE is building
    a GoldFact for the evaluator-only fact_grounding module — never as an
    argument to rerank_hybrid_search."""
    code = _code_without_strings(_without_docstrings_and_comments(_source()))
    fact_uses = re.findall(r"\S*\.fact\b", code)
    for use in fact_uses:
        assert "e.fact" in use or "evidence.fact" in use.lower() or use == ".fact", (
            f"unexpected .fact usage outside the GoldFact-building helper: {use!r}"
        )
    # and it must never appear inside a rerank_hybrid_search(...) call's argument list
    for call in re.findall(r"rerank_hybrid_search\(([^)]*)\)", code, flags=re.DOTALL):
        assert ".fact" not in call, f"rerank_hybrid_search() call must never reference .fact: {call}"


def test_query_passed_to_retrieval_is_always_record_query():
    source = _source()
    assert "query = record.query" in source
    calls = re.findall(r"rerank_hybrid_search\(\s*(\w+),", source)
    assert calls, "expected at least one rerank_hybrid_search(...) call"
    assert set(calls) == {"query"}, f"rerank_hybrid_search must always be called with `query`, found: {set(calls)}"


def test_no_fact_text_concatenated_into_a_query():
    """No string-concatenation/join/f-string construction of `query` that
    incorporates `.fact` — the query variable is assigned exactly once,
    directly from record.query, and never reassigned."""
    source = _source()
    assignments = re.findall(r"^\s*query\s*=.*$", source, flags=re.MULTILINE)
    assert len(assignments) == 1, f"expected exactly one assignment to `query`, found: {assignments}"
    assert assignments[0].strip() == "query = record.query  # THE ONLY THING passed to retrieval — never Evidence.fact, never gold answer"


def test_gold_answer_never_passed_to_retrieval():
    code = _code_without_strings(_without_docstrings_and_comments(_source()))
    for call in re.findall(r"rerank_hybrid_search\(([^)]*)\)", code, flags=re.DOTALL):
        assert "gold_answer" not in call and "record.answer" not in call


# --- 2: no runtime module imports this script's module or fact_grounding -------------------


def test_no_runtime_module_imports_this_script_or_fact_grounding():
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists()
        source = path.read_text()
        assert "compute_fact_grounding" not in source
        assert "fact_grounding" not in source, f"{path.name} must never import mhrag.eval.fact_grounding"


def test_script_uses_the_real_unmodified_assemble_context():
    """Must import and call the real production function — never
    reimplement its token-budget logic locally."""
    source = _source()
    assert "from mhrag.generation.context import approximate_token_count, assemble_context" in source
    assert "def assemble_context" not in source  # no local reimplementation
