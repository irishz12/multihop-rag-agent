#!/usr/bin/env python
"""Portfolio finalization — result charts for the README, generated ONLY
from already-measured artifacts (results/phase9_sample_report.json,
results/phase9_holdout_report.json, results/phase9_*_raw.json). No live
calls, no new measurements, no retrieval/model/prompt/router code touched.

The frozen report/raw files still use this project's legacy pipeline
names (`always_agentic`, `adaptive`) internally — see
`mhrag.eval.legacy_pipeline_names` for why they're never renamed. This
script rekeys them to canonical names (`agentic_multi_hop`, `adaptive_rag`)
immediately after loading, through that one module; everything below that
point — including every color, chart label, and axis — uses canonical
names only.

Fixed, consistent categorical color assignment across every chart (never
re-cycled per chart): Dense/Hybrid/Hybrid+Reranker in a light->dark gray
ramp (baselines, de-emphasized), Agentic Multi-Hop RAG in blue, Adaptive
RAG in orange — the Okabe-Ito colorblind-safe blue/orange pair, so the
headline comparison (Agentic vs Adaptive) is never ambiguous under color
vision deficiency.

Usage:
    python scripts/generate_phase9_charts.py

Writes results/charts/*.png.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhrag.config import PROJECT_ROOT
from mhrag.eval.legacy_pipeline_names import (
    get_quality_retention_pct,
    rekey_legacy_prefixed_keys,
    rekey_legacy_report,
)

CHARTS_DIR = PROJECT_ROOT / "results" / "charts"

COLORS = {
    "dense": "#B0B0B0",
    "hybrid": "#808080",
    "hybrid_reranker": "#4D4D4D",
    "agentic_multi_hop": "#0072B2",  # Agentic Multi-Hop RAG
    "adaptive_rag": "#E69F00",  # Adaptive RAG
}
LABELS = {
    "dense": "Dense RAG",
    "hybrid": "Hybrid RAG",
    "hybrid_reranker": "Hybrid + Reranker",
    "agentic_multi_hop": "Agentic Multi-Hop RAG",
    "adaptive_rag": "Adaptive RAG",
}
GROUP_LABELS = {
    "inference_query": "Inference", "comparison_query": "Comparison",
    "temporal_query": "Temporal", "null_query": "Null",
    "hop2": "2-hop", "hop3": "3-hop", "hop4": "4-hop", "null": "Null",
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


def _bar_with_labels(ax, x, heights, colors, labels, fmt="{:.2f}"):
    bars = ax.bar(x, heights, color=colors, width=0.6)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h, fmt.format(h),
                 ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    return bars


def chart_answer_quality(dev: dict, holdout: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    dev_pipelines = ["hybrid_reranker", "agentic_multi_hop", "adaptive_rag"]
    dev_vals = [dev["combined_quality_mean"][p] for p in dev_pipelines]
    _bar_with_labels(axes[0], range(len(dev_pipelines)), dev_vals,
                      [COLORS[p] for p in dev_pipelines], [LABELS[p] for p in dev_pipelines])
    axes[0].set_title("Development sample (n=50)")
    axes[0].set_ylabel("Combined quality score")

    hold_pipelines = ["agentic_multi_hop", "adaptive_rag"]
    hold_vals = [holdout["combined_quality_mean"][p] for p in hold_pipelines]
    _bar_with_labels(axes[1], range(len(hold_pipelines)), hold_vals,
                      [COLORS[p] for p in hold_pipelines], [LABELS[p] for p in hold_pipelines])
    axes[1].set_title("Final holdout (n=50, one-time)")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(labelleft=False)

    fig.suptitle("Answer quality: Agentic Multi-Hop RAG vs baselines")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "answer_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_evidence_coverage(dev: dict, holdout: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    dev_pipelines = ["dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag"]
    dev_vals = [dev["evidence_coverage_mean"][p] for p in dev_pipelines]
    _bar_with_labels(axes[0], range(len(dev_pipelines)), dev_vals,
                      [COLORS[p] for p in dev_pipelines], [LABELS[p] for p in dev_pipelines],
                      fmt="{:.0%}")
    axes[0].set_title("Development sample (n=50)")
    axes[0].set_ylabel("Gold-evidence coverage (non-null questions)")
    axes[0].set_ylim(0, 1.0)

    hold_pipelines = ["agentic_multi_hop", "adaptive_rag"]
    hold_vals = [holdout["evidence_coverage_mean"][p] for p in hold_pipelines]
    _bar_with_labels(axes[1], range(len(hold_pipelines)), hold_vals,
                      [COLORS[p] for p in hold_pipelines], [LABELS[p] for p in hold_pipelines],
                      fmt="{:.0%}")
    axes[1].set_title("Final holdout (n=50, one-time)")
    axes[1].tick_params(labelleft=False)

    fig.suptitle("Evidence coverage: how much gold evidence reached the final answer")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "evidence_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_cost_or_latency(metric_key: str, title: str, ylabel: str, fmt: str, filename: str,
                           dev_costs: dict, holdout: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    dev_pipelines = ["dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag"]
    dev_vals = [dev_costs[p] for p in dev_pipelines]
    _bar_with_labels(axes[0], range(len(dev_pipelines)), dev_vals,
                      [COLORS[p] for p in dev_pipelines], [LABELS[p] for p in dev_pipelines], fmt=fmt)
    axes[0].set_title("Development sample (n=50)")
    axes[0].set_ylabel(ylabel)

    hold_pipelines = ["agentic_multi_hop", "adaptive_rag"]
    hold_vals = [holdout["cost_latency"][f"{p}_mean_{metric_key}"] for p in hold_pipelines]
    _bar_with_labels(axes[1], range(len(hold_pipelines)), hold_vals,
                      [COLORS[p] for p in hold_pipelines], [LABELS[p] for p in hold_pipelines], fmt=fmt)
    axes[1].set_title("Final holdout (n=50, one-time)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_breakdown(dev_breakdown: dict, holdout_breakdown: dict, key_order: list[str], title: str,
                     filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))  # independent y-axes — no sharey clipping artifacts
    for ax, breakdown, split_title in (
        (axes[0], dev_breakdown, "Development sample"), (axes[1], holdout_breakdown, "Final holdout"),
    ):
        keys = [k for k in key_order if k in breakdown]
        x = range(len(keys))
        width = 0.35
        agentic_vals = [breakdown[k]["agentic_multi_hop_mean_quality"] for k in keys]
        adaptive_vals = [breakdown[k]["adaptive_rag_mean_quality"] for k in keys]
        ax.bar([i - width / 2 for i in x], agentic_vals, width, color=COLORS["agentic_multi_hop"],
               label=LABELS["agentic_multi_hop"])
        ax.bar([i + width / 2 for i in x], adaptive_vals, width, color=COLORS["adaptive_rag"],
               label=LABELS["adaptive_rag"])
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{GROUP_LABELS.get(k, k)}\n(n={breakdown[k]['n']})" for k in keys], fontsize=9)
        ax.set_title(split_title)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.margins(x=0.08)
    axes[0].set_ylabel("Combined quality")
    axes[0].legend(loc="upper center", fontsize=9, frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.18))
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_dev_vs_holdout(dev: dict, holdout: dict) -> None:
    metrics = [
        ("Quality retention", get_quality_retention_pct(dev), get_quality_retention_pct(holdout)),
        ("Cost reduction", dev["cost_latency"]["cost_reduction_pct"], holdout["cost_latency"]["cost_reduction_pct"]),
        ("Latency reduction", dev["cost_latency"]["latency_reduction_pct"],
         holdout["cost_latency"]["latency_reduction_pct"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(metrics))
    width = 0.35
    dev_vals = [m[1] for m in metrics]
    hold_vals = [m[2] for m in metrics]
    ax.bar([i - width / 2 for i in x], dev_vals, width, color="#56B4E9", label="Development")
    ax.bar([i + width / 2 for i in x], hold_vals, width, color="#0072B2", label="Final holdout")
    for i, (v_dev, v_hold) in enumerate(zip(dev_vals, hold_vals)):
        ax.text(i - width / 2, v_dev, f"{v_dev:.1%}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, v_hold, f"{v_hold:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Adaptive RAG vs Agentic Multi-Hop RAG")
    ax.legend(frameon=False)
    ax.set_title("Development vs final holdout: does the trade-off hold up?")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "dev_vs_holdout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the frozen reports, then immediately rekey every legacy-named
    # pipeline field to canonical — everything from here on sees only
    # agentic_multi_hop/adaptive_rag, never always_agentic/adaptive.
    dev = json.loads((PROJECT_ROOT / "results" / "phase9_sample_report.json").read_text())
    holdout = json.loads((PROJECT_ROOT / "results" / "phase9_holdout_report.json").read_text())
    for report in (dev, holdout):
        report["combined_quality_mean"] = rekey_legacy_report(report["combined_quality_mean"])
        report["evidence_coverage_mean"] = rekey_legacy_report(report["evidence_coverage_mean"])
        report["cost_latency"] = rekey_legacy_prefixed_keys(report["cost_latency"])
        for breakdown_key in ("breakdown_by_question_type", "breakdown_by_hop_count"):
            report[breakdown_key] = {
                group: rekey_legacy_prefixed_keys(values) for group, values in report[breakdown_key].items()
            }

    sample = json.loads((PROJECT_ROOT / "results" / "phase9_sample.json").read_text())
    sample_ids = set(sample["qa_ids"])
    dev_cost = {}
    dev_latency = {}
    for p in ("dense", "hybrid", "hybrid_reranker"):
        raw = json.loads((PROJECT_ROOT / "results" / f"phase9_{p}_raw.json").read_text())
        recs = [r for r in raw["records"] if r["qa_id"] in sample_ids]
        dev_cost[p] = sum(r["total_cost_usd"] for r in recs) / len(recs)
        dev_latency[p] = sum(r["total_latency_ms"] for r in recs) / len(recs)
    for p in ("agentic_multi_hop", "adaptive_rag"):
        dev_cost[p] = dev["cost_latency"][f"{p}_mean_cost_usd"]
        dev_latency[p] = dev["cost_latency"][f"{p}_mean_latency_ms"]

    chart_answer_quality(dev, holdout)
    chart_evidence_coverage(dev, holdout)
    chart_cost_or_latency("cost_usd", "Cost per query", "Mean cost/query (USD)", "${:.5f}",
                           "cost_per_query.png", dev_cost, holdout)
    chart_cost_or_latency("latency_ms", "Latency per query", "Mean latency/query (ms)", "{:.0f}",
                           "latency_per_query.png", dev_latency, holdout)
    chart_breakdown(
        dev["breakdown_by_question_type"], holdout["breakdown_by_question_type"],
        ["inference_query", "comparison_query", "temporal_query", "null_query"],
        "Quality by query type", "query_type_performance.png",
    )
    chart_breakdown(
        dev["breakdown_by_hop_count"], holdout["breakdown_by_hop_count"],
        ["hop2", "hop3", "hop4", "null"],
        "Quality by hop-count difficulty", "hop_performance.png",
    )
    chart_dev_vs_holdout(dev, holdout)

    print(f"Wrote 7 charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
