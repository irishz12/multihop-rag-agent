#!/usr/bin/env python
"""Phase 7.1: merge the 3 per-budget calibration result files produced by
scripts/agentic_budget_calibration.py, compute aggregate `BudgetMetrics`
for each, and apply the objective selection rule
(`mhrag.calibration.decision.select_budget`) to freeze one token budget.

Offline — reads only the JSON artifacts already written by the live sweep,
makes no API calls, touches no dataset split file directly.

Usage:
    python scripts/agentic_budget_calibration_decide.py

Writes results/agentic_budget_calibration_decision.json (the full decision,
per-budget metrics, and provenance — which raw result files and how many
questions each covered). If the selected budget differs from
configs/agent.yaml's current `loop.max_context_tokens`, updates that one
field in place (nothing else in the file is touched) and reports the
change; if 3000 (the current value) remains selected, the config file is
left untouched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.calibration.decision import BudgetMetrics, select_budget
from mhrag.config import PROJECT_ROOT

RESULT_FILES = {
    3000: "results/agentic_budget_calibration_3000.json",
    4500: "results/agentic_budget_calibration_4500.json",
    6000: "results/agentic_budget_calibration_6000.json",
}
DECISION_OUTPUT = "results/agentic_budget_calibration_decision.json"
AGENT_CONFIG_PATH = "configs/agent.yaml"


def _load_artifact(budget: int) -> dict:
    path = PROJECT_ROOT / RESULT_FILES[budget]
    if not path.exists():
        raise SystemExit(
            f"Missing calibration result file for budget={budget}: {path}\n"
            f"Run: python scripts/agentic_budget_calibration.py --budget {budget}"
        )
    return json.loads(path.read_text())


def _budget_metrics(budget: int, artifact: dict) -> BudgetMetrics:
    results = artifact["results"]
    n = len(results)
    if n == 0:
        raise SystemExit(f"Calibration result file for budget={budget} has zero results")

    token_budget_stops = sum(1 for r in results if r["stop_reason"] == "token_budget")
    new_docs_after_hop1 = [sum(r["evaluation"]["new_unique_docs_per_hop"][1:]) for r in results]
    recalls = [r["evaluation"]["recall"] for r in results]
    complete_evidence = [1.0 if r["evaluation"]["complete_evidence"] else 0.0 for r in results]
    costs = [r["cost_usd"]["total"] for r in results if r["cost_usd"]["total"] is not None]
    latencies = [r["latency_ms"]["total"] for r in results]

    return BudgetMetrics(
        token_budget=budget,
        token_budget_stop_rate=token_budget_stops / n,
        mean_new_unique_docs_after_hop1=sum(new_docs_after_hop1) / n,
        mean_recall=sum(recalls) / n,
        mean_complete_evidence_rate=sum(complete_evidence) / n,
        mean_cost_usd=sum(costs) / len(costs) if costs else 0.0,
        mean_latency_ms=sum(latencies) / n,
    )


def main() -> None:
    artifacts = {budget: _load_artifact(budget) for budget in RESULT_FILES}
    metrics_by_budget = {budget: _budget_metrics(budget, artifact) for budget, artifact in artifacts.items()}

    for budget, metrics in sorted(metrics_by_budget.items()):
        print(
            f"budget={budget}: token_budget_stop_rate={metrics.token_budget_stop_rate:.2%} "
            f"new_docs_after_hop1={metrics.mean_new_unique_docs_after_hop1:.2f} "
            f"recall={metrics.mean_recall:.3f} complete_evidence={metrics.mean_complete_evidence_rate:.2%} "
            f"cost=${metrics.mean_cost_usd:.6f} latency={metrics.mean_latency_ms:.0f}ms"
        )

    decision = select_budget(list(metrics_by_budget.values()))
    print(f"\nDECISION: selected_token_budget={decision.selected_token_budget} "
          f"(baseline={decision.baseline_token_budget})")
    print(decision.rationale)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 7.1 objective token-budget selection",
        "selected_token_budget": decision.selected_token_budget,
        "baseline_token_budget": decision.baseline_token_budget,
        "rationale": decision.rationale,
        "criteria_by_candidate": decision.criteria_by_candidate,
        "metrics_by_budget": {
            str(budget): {
                "token_budget_stop_rate": m.token_budget_stop_rate,
                "mean_new_unique_docs_after_hop1": m.mean_new_unique_docs_after_hop1,
                "mean_recall": m.mean_recall,
                "mean_complete_evidence_rate": m.mean_complete_evidence_rate,
                "mean_cost_usd": m.mean_cost_usd,
                "mean_latency_ms": m.mean_latency_ms,
            }
            for budget, m in metrics_by_budget.items()
        },
        "provenance": {
            "raw_result_files": RESULT_FILES,
            "n_questions_by_budget": {
                str(budget): len(artifacts[budget]["results"]) for budget in RESULT_FILES
            },
        },
    }
    out_path = PROJECT_ROOT / DECISION_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")

    config_path = PROJECT_ROOT / AGENT_CONFIG_PATH
    config_text = config_path.read_text()
    current_line_match = None
    for line in config_text.splitlines():
        if line.strip().startswith("max_context_tokens:"):
            current_line_match = line
            break
    if current_line_match is None:
        raise SystemExit(f"Could not find loop.max_context_tokens line in {config_path}")

    current_value = int(current_line_match.split(":", 1)[1].split("#", 1)[0].strip())
    if decision.selected_token_budget == current_value:
        print(f"\n{AGENT_CONFIG_PATH} already has max_context_tokens={current_value} — no change needed.")
        return

    new_line = current_line_match.replace(str(current_value), str(decision.selected_token_budget), 1)
    updated_text = config_text.replace(current_line_match, new_line, 1)
    config_path.write_text(updated_text)
    print(
        f"\nUpdated {AGENT_CONFIG_PATH}: loop.max_context_tokens {current_value} -> "
        f"{decision.selected_token_budget} (Phase 7.1 calibration result)"
    )


if __name__ == "__main__":
    main()
