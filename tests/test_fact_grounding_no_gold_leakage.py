"""Structural guard: mhrag.eval.fact_grounding uses gold Evidence.fact
text directly (by design, evaluator-only — same pattern as
mhrag.eval.judge/answer_metrics/task_success) and must therefore NEVER be
imported by any RUNTIME module. Mirrors
tests/test_task_success_no_gold_leakage.py exactly, scoped to this new
module.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "src" / "mhrag"

RUNTIME_MODULE_PATHS = [
    SRC_ROOT / "routing" / "features.py",
    SRC_ROOT / "routing" / "heuristic.py",
    SRC_ROOT / "routing" / "glm_router.py",
    SRC_ROOT / "routing" / "prompts.py",
    SRC_ROOT / "routing" / "router.py",
    SRC_ROOT / "routing" / "evidence_gate.py",
    SRC_ROOT / "routing" / "evidence_gate_prompts.py",
    SRC_ROOT / "routing" / "sequential_router.py",
    SRC_ROOT / "routing" / "rerank_features.py",
    SRC_ROOT / "routing" / "learned_features.py",
    SRC_ROOT / "routing" / "learned_router.py",
    SRC_ROOT / "routing" / "learned_sequential_router.py",
    SRC_ROOT / "adaptive" / "pipeline.py",
    SRC_ROOT / "agent" / "loop.py",
    SRC_ROOT / "agent" / "controller.py",
    SRC_ROOT / "agent" / "evidence.py",
    SRC_ROOT / "agent" / "prompts.py",
    SRC_ROOT / "generation" / "answer.py",
    SRC_ROOT / "generation" / "context.py",
    SRC_ROOT / "generation" / "cost.py",
    SRC_ROOT / "generation" / "prompts.py",
    SRC_ROOT / "retrieval" / "dense.py",
    SRC_ROOT / "retrieval" / "bm25.py",
    SRC_ROOT / "retrieval" / "rrf.py",
    SRC_ROOT / "retrieval" / "rerank.py",
    SRC_ROOT / "retrieval" / "hybrid.py",
]

FORBIDDEN_EVAL_MODULES = {"judge", "judge_prompts", "answer_metrics", "task_success", "task_success_metrics", "fact_grounding"}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S*eval[.\s]+import\s+(\S+)|import\s+\S*eval\.(\S+))", re.MULTILINE)


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_every_scanned_runtime_module_exists():
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists(), f"expected runtime module {path} not found"


def test_fact_grounding_module_exists():
    assert (SRC_ROOT / "eval" / "fact_grounding.py").exists()


def test_runtime_modules_never_import_fact_grounding():
    for path in RUNTIME_MODULE_PATHS:
        source = path.read_text()
        for match in _IMPORT_RE.finditer(source):
            imported = (match.group(1) or match.group(2) or "").strip().rstrip(",")
            imported_root = imported.split(".")[0].split(",")[0].strip()
            assert imported_root not in FORBIDDEN_EVAL_MODULES, (
                f"{path.name} imports evaluator-only gold-bearing module {imported_root!r} — "
                "no runtime module may ever see gold fact text"
            )


def test_fact_grounding_module_never_imports_any_runtime_package():
    """Reverse direction, belt-and-suspenders: fact_grounding.py should
    have no reason to import mhrag.agent/.generation/.routing/.adaptive/
    .retrieval at all. Checked against CODE only — docstrings legitimately
    discuss these packages in prose."""
    code = _without_docstrings_and_comments((SRC_ROOT / "eval" / "fact_grounding.py").read_text())
    for forbidden_package in ("mhrag.agent", "mhrag.generation", "mhrag.routing", "mhrag.adaptive", "mhrag.retrieval"):
        assert forbidden_package not in code, (
            f"fact_grounding.py's CODE references {forbidden_package!r} — evaluator-only modules "
            "should not depend on runtime packages"
        )


def test_fact_grounding_module_makes_no_llm_or_network_call():
    """Pure text/string module — no client, no model, no I/O at all."""
    source = (SRC_ROOT / "eval" / "fact_grounding.py").read_text()
    for forbidden in ("MantleClient", "requests.", "urllib", "QdrantClient", "SentenceTransformer", "open("):
        assert forbidden not in source
