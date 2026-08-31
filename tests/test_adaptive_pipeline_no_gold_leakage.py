"""Structural guard: mhrag.adaptive.pipeline (the RUNTIME Phase 8B
orchestrator) must never import an EVALUATOR-ONLY module, and none of its
function signatures may carry a gold-labeled parameter. Same pattern as
tests/test_routing_no_gold_leakage.py, scoped to the new `mhrag.adaptive`
package instead of `mhrag.routing`.
"""

from __future__ import annotations

import re
from pathlib import Path

ADAPTIVE_DIR = Path(__file__).parent.parent / "src" / "mhrag" / "adaptive"

RUNTIME_MODULES = ["pipeline.py"]
EVALUATOR_ONLY_MODULE_NAMES = {"oracle", "tune_thresholds", "metrics", "split", "gate_analysis", "learned_router_training"}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S*routing[.\s]+import\s+(\S+)|import\s+\S*routing\.(\S+))", re.MULTILINE)


def test_every_runtime_module_exists():
    for name in RUNTIME_MODULES:
        assert (ADAPTIVE_DIR / name).exists(), f"expected runtime module {name} not found"


def test_runtime_modules_never_import_evaluator_only_modules():
    for name in RUNTIME_MODULES:
        source = (ADAPTIVE_DIR / name).read_text()
        for match in _IMPORT_RE.finditer(source):
            imported = (match.group(1) or match.group(2) or "").strip().rstrip(",")
            imported_root = imported.split(".")[0].split(",")[0].strip()
            assert imported_root not in EVALUATOR_ONLY_MODULE_NAMES, (
                f"{name} imports evaluator-only module {imported_root!r} — "
                "the Adaptive runtime pipeline must never reach oracle/tuning/metrics/split code"
            )


def test_runtime_module_functions_have_no_gold_labeled_parameter():
    forbidden_param_names = {
        "answer", "evidence_list", "gold", "gold_doc_ids", "oracle_route", "oracle_label", "question_type",
    }
    def_re = re.compile(r"^def \w+\(([^)]*)\)", re.MULTILINE)
    for name in RUNTIME_MODULES:
        source = (ADAPTIVE_DIR / name).read_text()
        for match in def_re.finditer(source):
            params = match.group(1)
            param_names = {p.split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()}
            overlap = param_names & forbidden_param_names
            assert not overlap, f"{name} has forbidden gold parameter(s): {overlap}"


def test_pipeline_module_never_references_final_holdout():
    source = (ADAPTIVE_DIR / "pipeline.py").read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments
