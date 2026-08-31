"""Structural guard for scripts/analyze_multihop_success.py:

 1. Zero LLM/API calls anywhere in the script.
 2. Development-only — hardcoded dev split, no CLI flag, no holdout marker.
 3. Cannot overwrite any existing results/*.json (exhaustive list as of
    Phase 6A) or the fact_grounding_report.json artifact.
 4. Never imports mhrag.eval.fact_grounding — doc-level multi-hop success
    analysis must stay structurally separate from fact-level Tier A/B
    grounding (no accidental mixing).
 5. Never imports any runtime module (agent/generation/routing/adaptive).
 6. EXCLUDED_QA_IDS is applied only to example selection, never to any
    aggregate population count.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "analyze_multihop_success.py"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Every results/*.json filename that predates Phase 6A (i.e. everything
# except this phase's own two new artifacts) — none of these may ever be a
# write target of this script except its own OUTPUT_FILE.
_PHASE6A_NEW_FILENAMES = {"multihop_success_analysis.json", "multihop_examples_replay.json"}
PROTECTED_EXISTING_FILENAMES = tuple(
    sorted(p.name for p in RESULTS_DIR.glob("*.json") if p.name not in _PHASE6A_NEW_FILENAMES)
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


def test_script_makes_zero_llm_calls():
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval",
                       "run_adaptive_pipeline", "call_controller(", "rerank_hybrid_search(",
                       "dense_search(", "bm25_search("):
        assert forbidden not in source, f"must be zero LLM/API/retrieval calls — found {forbidden!r}"


def test_script_dev_split_hardcoded_no_cli_flag():
    source = _source()
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in source
    assert "argparse" not in source
    assert "--split" not in source


def test_script_never_references_holdout():
    code = _without_docstrings_and_comments(_source())
    for marker in ("final_holdout", "phase9_holdout_", "final_evaluation_manifest", "final_holdout_consumed"):
        assert marker not in code


def test_script_only_writes_to_its_own_out_path():
    write_calls = re.findall(r"(\w+)\.write_text\(", _source())
    assert write_calls == ["out_path"], f"unexpected write target(s): {write_calls}"


def test_output_filename_is_new_and_protected_filenames_never_write_targets():
    source = _source()
    assert 'OUTPUT_FILE = "results/multihop_success_analysis.json"' in source
    assert "multihop_success_analysis.json" not in PROTECTED_EXISTING_FILENAMES
    write_lines = [line.strip() for line in source.splitlines() if ".write_text(" in line]
    for line in write_lines:
        for protected in PROTECTED_EXISTING_FILENAMES:
            assert protected not in line, f"write_text() call must never target {protected!r}: {line}"


def test_script_never_imports_fact_grounding():
    """Checked against CODE only, stripped of docstrings/comments AND
    string literals — the report's own "purpose" string legitimately
    documents the non-mixing boundary in prose, which is not an import."""
    code = _code_without_strings(_without_docstrings_and_comments(_source()))
    assert "fact_grounding" not in code, "must stay structurally separate from Tier A/Tier B fact grounding"


def test_script_never_imports_runtime_modules():
    code = _without_docstrings_and_comments(_source())
    for forbidden_import in ("mhrag.agent", "mhrag.generation", "mhrag.routing", "mhrag.adaptive"):
        assert forbidden_import not in code


def test_excluded_qa_ids_only_referenced_in_example_selection_path():
    """EXCLUDED_QA_IDS must be passed to select_examples() and NEVER used
    to filter all_multihop_qa_ids / three_way_qa_ids / added_evidence_qa_ids
    (the aggregate population lists)."""
    source = _source()
    assert "EXCLUDED_QA_IDS = frozenset(" in source
    # the only use of EXCLUDED_QA_IDS besides its definition/report dict must be the select_examples
    # call — scanned against CODE only (docstrings/comments legitimately discuss it in prose)
    code = _without_docstrings_and_comments(source)
    lines_using_it = [
        line for line in code.splitlines()
        if "EXCLUDED_QA_IDS" in line and "EXCLUDED_QA_IDS = frozenset(" not in line
    ]
    assert lines_using_it, "expected EXCLUDED_QA_IDS to be used somewhere"
    for line in lines_using_it:
        assert "select_examples(" in line or '"excluded_qa_ids"' in line or "EXCLUSION_REASON" in line, (
            f"EXCLUDED_QA_IDS must only gate example selection or report labeling, found: {line}"
        )


def test_added_required_evidence_denominator_is_full_population_not_filtered():
    """The 33/92 statistic's denominator must be population_all_multihop
    (92), never a post-hoc filtered subset."""
    source = _source()
    assert '"denominator": len(all_multihop_qa_ids)' in source
