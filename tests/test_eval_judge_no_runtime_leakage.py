"""Structural guard: mhrag.eval.judge / mhrag.eval.answer_metrics use gold
answer text directly (by design — see mhrag.eval.judge's module docstring)
and must therefore NEVER be imported by any RUNTIME module. Scans every
runtime module across the whole `mhrag` package (routing runtime modules,
mhrag.adaptive, mhrag.agent, mhrag.generation) for a forbidden import.
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
]

FORBIDDEN_EVAL_MODULES = {"judge", "judge_prompts", "answer_metrics"}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S*eval[.\s]+import\s+(\S+)|import\s+\S*eval\.(\S+))", re.MULTILINE)


def test_every_scanned_runtime_module_exists():
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists(), f"expected runtime module {path} not found"


def test_runtime_modules_never_import_judge_or_answer_metrics():
    for path in RUNTIME_MODULE_PATHS:
        source = path.read_text()
        for match in _IMPORT_RE.finditer(source):
            imported = (match.group(1) or match.group(2) or "").strip().rstrip(",")
            imported_root = imported.split(".")[0].split(",")[0].strip()
            assert imported_root not in FORBIDDEN_EVAL_MODULES, (
                f"{path.name} imports evaluator-only gold-bearing module {imported_root!r} — "
                "no runtime module may ever see gold answer text"
            )


def test_judge_module_never_hardcodes_the_qwen_or_glm_model_ids():
    """Belt-and-suspenders: the judge module's actual CODE (docstrings may
    explain the design rationale in prose) must never hardcode the
    answer-generation model id (Qwen) OR the agent-controller model id
    (GLM/zai) — the judge model (`openai.gpt-oss-120b`, configured by its
    caller via `configs/judge.yaml`) must stay a THIRD, distinct model from
    both, never one of the two models it might be grading/adjacent to."""
    import re

    source = (SRC_ROOT / "eval" / "judge.py").read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    lowered = without_comments.lower()
    assert "qwen" not in lowered
    assert "glm" not in lowered
    assert "zai" not in lowered
