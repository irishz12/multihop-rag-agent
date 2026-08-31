"""Structural guard for scripts/validate_fact_chunk_survival.py: dev-only,
zero LLM/API calls (chunking + a local tokenizer only — no Mantle client,
no judge, no generation), never touches final_holdout.json, never
modifies mhrag.ingestion.chunking or any config, only ever writes its own
new output file.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "validate_fact_chunk_survival.py"


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
    assert 'OUTPUT_FILE = "results/fact_grounding_chunk_survival.json"' in _source()


def test_script_never_references_holdout_in_code():
    code = _without_docstrings_and_comments(_source())
    for marker in ("final_holdout", "phase9_holdout_"):
        assert marker not in code


def test_script_makes_zero_llm_calls():
    source = _source()
    for forbidden in ("MantleClient", "call_judge(", "generate_answer(", "run_agentic_retrieval",
                       "run_adaptive_pipeline", "dense_search(", "bm25_search(", "rerank_results(", "QdrantClient"):
        assert forbidden not in source, f"validate_fact_chunk_survival.py must not call {forbidden!r} — chunking only"


def test_script_dev_split_file_constant_and_no_split_flag():
    source = _source()
    assert 'DEV_SPLIT_FILE = "dev_subset.json"' in source
    assert "--split" not in source


def test_script_never_dumps_raw_environ():
    assert "os.environ" not in _source()


def test_script_uses_unmodified_chunking_functions_not_a_reimplementation():
    """Must import the real ChunkingConfig/chunk_corpus, not redefine its
    own copy — otherwise the whole point (validating the PRODUCTION
    chunker) is defeated."""
    source = _source()
    assert "from mhrag.ingestion.chunking import ChunkingConfig, chunk_corpus" in source
