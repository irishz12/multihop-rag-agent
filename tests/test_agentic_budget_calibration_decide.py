"""Tests for scripts/agentic_budget_calibration_decide.py — offline, using
synthetic per-budget artifact dicts (no live calls, no real result files
required). Proves: per-budget metrics are computed correctly from raw
results, the selected configuration is persisted with provenance, and
configs/agent.yaml is updated only when the selection actually changes.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "agentic_budget_calibration_decide.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agentic_budget_calibration_decide", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_result(stop_reason, new_docs_per_hop, recall, complete_evidence, cost, latency_ms):
    return {
        "stop_reason": stop_reason,
        "evaluation": {
            "new_unique_docs_per_hop": new_docs_per_hop,
            "recall": recall,
            "complete_evidence": complete_evidence,
        },
        "cost_usd": {"total": cost},
        "latency_ms": {"total": latency_ms},
    }


def test_script_source_never_references_final_holdout_outside_documentation():
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_budget_metrics_computed_correctly_from_raw_results():
    module = _load_module()
    artifact = {
        "results": [
            _fake_result("token_budget", [1, 0], 0.5, False, 0.01, 1000.0),
            _fake_result("evidence_sufficient", [2, 1], 1.0, True, 0.02, 2000.0),
        ]
    }
    metrics = module._budget_metrics(3000, artifact)
    assert metrics.token_budget == 3000
    assert metrics.token_budget_stop_rate == 0.5  # 1 of 2
    assert metrics.mean_new_unique_docs_after_hop1 == 0.5  # (0 + 1) / 2
    assert metrics.mean_recall == 0.75
    assert metrics.mean_complete_evidence_rate == 0.5
    assert metrics.mean_cost_usd == 0.015
    assert metrics.mean_latency_ms == 1500.0


def test_budget_metrics_raises_on_empty_results():
    module = _load_module()
    with __import__("pytest").raises(SystemExit):
        module._budget_metrics(3000, {"results": []})


def test_decision_persisted_with_provenance_and_config_updated(tmp_path, monkeypatch):
    module = _load_module()

    for budget, stop_reason, recovery in [
        (3000, "token_budget", 0.0),
        (4500, "evidence_sufficient", 1.0),
        (6000, "evidence_sufficient", 1.0),
    ]:
        results = [_fake_result(stop_reason, [0, recovery], 0.9, True, 0.01, 1000.0) for _ in range(5)]
        (tmp_path / f"calib_{budget}.json").write_text(json.dumps({"results": results}))

    monkeypatch.setattr(
        module, "_load_artifact", lambda budget: json.loads((tmp_path / f"calib_{budget}.json").read_text())
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "DECISION_OUTPUT", "decision.json")
    monkeypatch.setattr(module, "AGENT_CONFIG_PATH", "agent.yaml")

    agent_config = tmp_path / "agent.yaml"
    agent_config.write_text("loop:\n  max_hops: 3\n  max_context_tokens: 3000     # comment\n  timeout_seconds: 30\n")

    module.main()

    written = json.loads((tmp_path / "decision.json").read_text())
    assert written["selected_token_budget"] == 4500
    assert written["baseline_token_budget"] == 3000
    assert "rationale" in written and written["rationale"]
    assert written["provenance"]["n_questions_by_budget"] == {"3000": 5, "4500": 5, "6000": 5}
    assert "raw_result_files" in written["provenance"]

    updated_config = agent_config.read_text()
    assert "max_context_tokens: 4500" in updated_config
    assert "max_hops: 3" in updated_config  # untouched
    assert "timeout_seconds: 30" in updated_config  # untouched


def test_config_left_untouched_when_baseline_already_selected(tmp_path, monkeypatch):
    module = _load_module()
    for budget in (3000, 4500, 6000):
        results = [_fake_result("token_budget", [0, 0], 0.9, True, 0.01, 1000.0) for _ in range(5)]
        (tmp_path / f"calib_{budget}.json").write_text(json.dumps({"results": results}))

    monkeypatch.setattr(
        module, "_load_artifact", lambda budget: json.loads((tmp_path / f"calib_{budget}.json").read_text())
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "DECISION_OUTPUT", "decision.json")
    monkeypatch.setattr(module, "AGENT_CONFIG_PATH", "agent.yaml")

    agent_config = tmp_path / "agent.yaml"
    original_text = "loop:\n  max_hops: 3\n  max_context_tokens: 3000     # comment\n"
    agent_config.write_text(original_text)

    module.main()

    assert agent_config.read_text() == original_text  # baseline kept -> no config write
