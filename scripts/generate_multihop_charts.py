#!/usr/bin/env python
"""Phase 6A charts — generated ONLY from results/multihop_success_analysis.json
(itself already generated offline from committed artifacts, zero LLM calls).
No live calls, no new measurements. Deliberately a SEPARATE script from
scripts/generate_phase9_charts.py so this phase never touches the script
that backs the README's already-published §8/§9 numbers.

Same color/style conventions as generate_phase9_charts.py: Hybrid+Reranker
gray, Agentic Multi-Hop blue, Context-Matched a mid-gray-blue to signal
"control, not a baseline" without inventing a new hue family.

Usage:
    python scripts/generate_multihop_charts.py

Writes results/charts/multihop_evidence_coverage.png,
results/charts/multihop_question_type_breakdown.png.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhrag.config import PROJECT_ROOT

CHARTS_DIR = PROJECT_ROOT / "results" / "charts"
INPUT_FILE = "results/multihop_success_analysis.json"

COLORS = {
    "baseline_hybrid_reranker": "#4D4D4D",
    "context_matched": "#7A9BB5",
    "agentic_final_all_hops": "#0072B2",
}
LABELS = {
    "baseline_hybrid_reranker": "Hybrid + Reranker\n(5-chunk)",
    "context_matched": "Context-Matched\n(control)",
    "agentic_final_all_hops": "Agentic Multi-Hop\n(all hops)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def chart_evidence_coverage(report: dict) -> None:
    three_way = report["evidence_coverage_three_way"]
    n = report["populations"]["population_three_way_n"]
    pipelines = ["baseline_hybrid_reranker", "context_matched", "agentic_final_all_hops"]
    vals = [three_way[p] for p in pipelines]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(range(len(pipelines)), vals, color=[COLORS[p] for p in pipelines], width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(pipelines)))
    ax.set_xticklabels([LABELS[p] for p in pipelines])
    ax.set_ylabel("Mean gold-document evidence coverage")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Evidence coverage on genuinely multi-hop-resolved questions (n={n})")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "multihop_evidence_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_question_type_breakdown(report: dict) -> None:
    by_type = report["added_required_evidence"]["by_question_type"]
    n_total = report["added_required_evidence"]["n"]
    types = sorted(by_type, key=lambda t: -by_type[t])
    vals = [by_type[t] for t in types]
    type_labels = {"comparison_query": "Comparison", "inference_query": "Inference", "temporal_query": "Temporal"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(types)), vals, color="#0072B2", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([type_labels.get(t, t) for t in types])
    ax.set_ylabel("Questions with added required evidence")
    ax.set_title(f"Question-type breakdown, {n_total} added-required-evidence cases", fontsize=12)
    fig.tight_layout(pad=1.5)
    fig.savefig(CHARTS_DIR / "multihop_question_type_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    report = _load(INPUT_FILE)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_evidence_coverage(report)
    chart_question_type_breakdown(report)
    print(f"Wrote {CHARTS_DIR / 'multihop_evidence_coverage.png'}")
    print(f"Wrote {CHARTS_DIR / 'multihop_question_type_breakdown.png'}")


if __name__ == "__main__":
    main()
