"""Package-wide structural guard: the RUNTIME routing modules
(mhrag.routing.features, mhrag.routing.heuristic, mhrag.routing.
glm_router, mhrag.routing.prompts, mhrag.routing.router) must never import
the EVALUATOR-ONLY modules (mhrag.routing.oracle, mhrag.routing.
tune_thresholds, mhrag.routing.metrics, mhrag.routing.split) — this is the
structural mechanism that makes "the runtime router never sees a gold
answer / evidence_list / expected documents / question_type / oracle route
label" true by construction, not just by convention.
"""

from __future__ import annotations

import re
from pathlib import Path

ROUTING_DIR = Path(__file__).parent.parent / "src" / "mhrag" / "routing"

RUNTIME_MODULES = [
    "features.py", "heuristic.py", "glm_router.py", "prompts.py", "router.py",
    "evidence_gate.py", "evidence_gate_prompts.py", "sequential_router.py",
    "rerank_features.py", "learned_features.py", "learned_router.py", "learned_sequential_router.py",
]
EVALUATOR_ONLY_MODULE_NAMES = {
    "oracle", "tune_thresholds", "metrics", "split", "gate_analysis", "learned_router_training",
}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+\S*routing[.\s]+import\s+(\S+)|import\s+\S*routing\.(\S+))", re.MULTILINE)


def test_every_runtime_module_exists():
    for name in RUNTIME_MODULES:
        assert (ROUTING_DIR / name).exists(), f"expected runtime module {name} not found"


def test_runtime_modules_never_import_evaluator_only_modules():
    for name in RUNTIME_MODULES:
        source = (ROUTING_DIR / name).read_text()
        for match in _IMPORT_RE.finditer(source):
            imported = (match.group(1) or match.group(2) or "").strip().rstrip(",")
            imported_root = imported.split(".")[0].split(",")[0].strip()
            assert imported_root not in EVALUATOR_ONLY_MODULE_NAMES, (
                f"{name} imports evaluator-only module {imported_root!r} — "
                "the runtime router must never reach oracle/tuning/metrics/split code"
            )


def test_runtime_module_functions_have_no_gold_labeled_parameter():
    """Belt-and-suspenders text scan: no parameter name in any runtime
    module's function signatures may be named after a gold/evaluator
    concept. Complements the per-module `inspect.signature` tests."""
    forbidden_param_names = {"answer", "evidence_list", "gold", "gold_doc_ids", "oracle_route", "oracle_label"}
    def_re = re.compile(r"^def \w+\(([^)]*)\)", re.MULTILINE)
    for name in RUNTIME_MODULES:
        source = (ROUTING_DIR / name).read_text()
        for match in def_re.finditer(source):
            params = match.group(1)
            param_names = {p.split(":")[0].split("=")[0].strip() for p in params.split(",") if p.strip()}
            overlap = param_names & forbidden_param_names
            assert not overlap, f"{name} has forbidden gold parameter(s): {overlap}"


def test_evaluator_only_modules_exist_and_are_separate_from_runtime():
    for name in EVALUATOR_ONLY_MODULE_NAMES:
        path = ROUTING_DIR / f"{name}.py"
        assert path.exists(), f"expected evaluator-only module {name}.py not found"
        assert f"{name}.py" not in RUNTIME_MODULES
