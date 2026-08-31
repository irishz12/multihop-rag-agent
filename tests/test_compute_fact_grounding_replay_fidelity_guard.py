"""Structural guard for scripts/compute_fact_grounding_replay_fidelity.py:
dev-only, offline (no LLM calls, no retrieval calls — reads only
already-written JSON), never touches final_holdout.json, never overwrites
any existing results/*.json or the Step 2 replay artifact, only ever
writes its own new output file, does not compute fact_grounded_rate.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "compute_fact_grounding_replay_fidelity.py"


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


def test_output_filename_is_new_and_distinct_from_replay_artifact():
    source = _source()
    assert 'OUTPUT_FILE = "results/fact_grounding_replay_fidelity.json"' in source
    assert '"results/fact_grounding_replay_raw.json"' not in [
        line.strip() for line in source.splitlines() if ".write_text(" in line
    ]


def test_script_is_offline_no_retrieval_or_llm_calls():
    source = _source()
    for forbidden in (
        "MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval",
        "run_adaptive_pipeline", "dense_search(", "bm25_search(", "rerank_hybrid_search(",
        "rerank_results(", "QdrantClient", "EmbeddingModel(", "Bm25Model(", "Reranker(",
    ):
        assert forbidden not in source, f"must be fully offline — found {forbidden!r}"


def test_script_does_not_compute_fact_grounded_rate():
    """Explicit scope guard per the Phase 5A brief: this validation phase
    must not compute fact_grounded_rate. Checked against code with
    string literals also stripped — this script's own "purpose" text
    legitimately narrates "no fact_grounded_rate computed" in prose,
    which must not itself trip the check."""
    code = _without_docstrings_and_comments(_source())
    code_no_strings = re.sub(r'"[^"]*"', '""', re.sub(r"'[^']*'", "''", code))
    assert "fact_grounded_rate" not in code_no_strings
    assert "fact_grounded(" not in code_no_strings


def test_script_never_references_holdout_in_code():
    code = _without_docstrings_and_comments(_source())
    for marker in ("final_holdout", "phase9_holdout_"):
        assert marker not in code


def test_script_uses_fixed_documented_seed():
    source = _source()
    assert "SAMPLE_SEED = 2029" in source
    assert "random.Random(SAMPLE_SEED)" in source


def test_script_has_no_split_flag_and_never_dumps_environ():
    source = _source()
    assert "--split" not in source
    assert "os.environ" not in source
