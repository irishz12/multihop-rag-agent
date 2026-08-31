"""Structural guard: mhrag.eval.task_success / mhrag.eval.task_success_metrics
use gold answer text directly (by design, same as mhrag.eval.judge/
answer_metrics) and must therefore NEVER be imported by any RUNTIME
module. Mirrors tests/test_eval_judge_no_runtime_leakage.py exactly,
scoped to the two new Task Success modules, plus a check that
Evidence.fact (parsed by mhrag.data.schema but not used by
task_success.py in this phase — fact-level groundedness is an explicitly
deferred, separate, not-yet-approved phase) still never reaches a
runtime prompt.
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

FORBIDDEN_EVAL_MODULES = {"judge", "judge_prompts", "answer_metrics", "task_success", "task_success_metrics"}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S*eval[.\s]+import\s+(\S+)|import\s+\S*eval\.(\S+))", re.MULTILINE)


def test_every_scanned_runtime_module_exists():
    for path in RUNTIME_MODULE_PATHS:
        assert path.exists(), f"expected runtime module {path} not found"


def test_task_success_modules_exist():
    assert (SRC_ROOT / "eval" / "task_success.py").exists()
    assert (SRC_ROOT / "eval" / "task_success_metrics.py").exists()


def test_runtime_modules_never_import_task_success():
    for path in RUNTIME_MODULE_PATHS:
        source = path.read_text()
        for match in _IMPORT_RE.finditer(source):
            imported = (match.group(1) or match.group(2) or "").strip().rstrip(",")
            imported_root = imported.split(".")[0].split(",")[0].strip()
            assert imported_root not in FORBIDDEN_EVAL_MODULES, (
                f"{path.name} imports evaluator-only gold-bearing module {imported_root!r} — "
                "no runtime module may ever see gold answer text"
            )


def _without_docstrings_and_comments(source: str) -> str:
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", without_docstrings)


def test_task_success_module_never_imports_any_runtime_module():
    """Reverse direction, belt-and-suspenders: task_success.py should have
    no reason to import mhrag.agent/.generation/.routing/.adaptive at
    all — if it ever does, that's worth a human looking at even though it
    wouldn't itself leak gold data outward. Checked against the module's
    CODE only (docstrings/comments may discuss these packages in prose —
    this module's own docstring does, precisely to explain the guard)."""
    code = _without_docstrings_and_comments((SRC_ROOT / "eval" / "task_success.py").read_text())
    for forbidden_package in ("mhrag.agent", "mhrag.generation", "mhrag.routing", "mhrag.adaptive"):
        assert forbidden_package not in code, (
            f"task_success.py's CODE references {forbidden_package!r} — evaluator-only modules should not "
            "depend on runtime packages"
        )


def test_task_success_module_does_not_use_evidence_fact():
    """Fact-level groundedness is an explicitly deferred, separate,
    not-yet-approved phase (see the Task Success design doc) — this
    implementation phase must not reach into Evidence.fact IN CODE at
    all (docstrings may and do discuss the deferral in prose), so there
    is nothing new here for the existing 'fact is never leaked' guards
    (tests/test_loader.py, tests/test_agent_loop.py, etc.) to miss."""
    code = _without_docstrings_and_comments((SRC_ROOT / "eval" / "task_success.py").read_text())
    assert ".fact" not in code


def test_task_success_metrics_module_has_no_gold_or_runtime_dependency():
    """task_success_metrics.py is pure statistics (proportions, CIs,
    deltas) — it must not import anything from mhrag at all, gold-bearing
    or otherwise, since it operates on plain numbers its caller already
    extracted."""
    source = (SRC_ROOT / "eval" / "task_success_metrics.py").read_text()
    assert "import mhrag" not in source
    assert "from mhrag" not in source


def test_judge_module_never_hardcodes_the_qwen_or_glm_model_ids():
    """Unchanged from tests/test_eval_judge_no_runtime_leakage.py — kept
    here too so this file alone fully covers the eval-package leakage
    contract for anyone reviewing Task Success in isolation."""
    source = (SRC_ROOT / "eval" / "judge.py").read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    lowered = without_comments.lower()
    assert "qwen" not in lowered
    assert "glm" not in lowered
    assert "zai" not in lowered
