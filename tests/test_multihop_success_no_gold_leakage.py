"""Structural guards for mhrag.eval.multihop_success:

 1. Never imported by any runtime module (mirrors
    tests/test_fact_grounding_no_gold_leakage.py exactly).
 2. Never imports mhrag.eval.fact_grounding, and never mixes Tier A/Tier B
    fact-grounding language into its own (doc-level, all-hops) analysis —
    the two are structurally different metrics over different scopes and
    must never be blended (see module docstring).
 3. Makes no LLM/API/network call.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "src" / "mhrag"
MODULE_PATH = SRC_ROOT / "eval" / "multihop_success.py"

RUNTIME_MODULE_PATHS = [
    SRC_ROOT / "adaptive" / "pipeline.py",
    SRC_ROOT / "agent" / "loop.py",
    SRC_ROOT / "agent" / "controller.py",
    SRC_ROOT / "generation" / "answer.py",
    SRC_ROOT / "generation" / "context.py",
    SRC_ROOT / "retrieval" / "rerank.py",
    SRC_ROOT / "routing" / "router.py",
]


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_module_exists():
    assert MODULE_PATH.exists()


def test_runtime_modules_never_import_multihop_success():
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists(), f"expected runtime module {path} not found"
        source = path.read_text()
        assert "multihop_success" not in source, f"{path.name} must never import mhrag.eval.multihop_success"


def test_module_never_imports_fact_grounding():
    """Doc-level (all-hops) evidence analysis must stay structurally
    separate from fact-level (Tier A/B) grounding — importing it would
    make an accidental Tier A/B blend trivially possible."""
    code = _without_docstrings_and_comments(MODULE_PATH.read_text())
    assert "fact_grounding" not in code


def test_module_never_imports_any_runtime_package():
    code = _without_docstrings_and_comments(MODULE_PATH.read_text())
    for forbidden_package in ("mhrag.agent", "mhrag.generation", "mhrag.routing", "mhrag.adaptive", "mhrag.retrieval"):
        assert forbidden_package not in code


def test_module_makes_no_llm_or_network_call():
    source = MODULE_PATH.read_text()
    for forbidden in ("MantleClient", "requests.", "urllib", "QdrantClient", "SentenceTransformer", "open("):
        assert forbidden not in source
