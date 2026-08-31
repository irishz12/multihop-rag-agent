#!/usr/bin/env python
"""Phase 9 (reduced 50-question scope): OFFLINE aggregation — no live
calls, no Mantle client. Combines:

  - results/phase9_sample.json (the frozen 50-qa_id selection)
  - results/phase9_{pipeline}_raw.json x5 (already-completed pipeline traces)
  - results/phase9_judge_{pipeline}.json x3 (hybrid_reranker/agentic_multi_hop/
    adaptive_rag — the only 3 pipelines judged, per spec)
  - data/processed/dev_subset.json (gold evidence, read-only, for evidence-
    coverage scoring — same "score strictly AFTER the fact" pattern as
    mhrag.routing.gate_analysis / Phase 8B's smoke comparison script)

into one report: deterministic metrics (normalized EM / token F1 /
abstention) for all 5 pipelines, judge scores for the 3 judged pipelines,
Adaptive RAG vs Agentic Multi-Hop RAG quality retention, breakdowns by
query type and hop count, and an under-routed-failure list.

The already-frozen results/phase9_sample_report.json this script produced
still uses this project's legacy pipeline names (`always_agentic`,
`adaptive`) — see `mhrag.eval.legacy_pipeline_names` for why it's never
rewritten. Reading the pre-existing per-pipeline raw/judge files (which are
ALSO named with those legacy pipeline names) is the one place this script
still needs them; every canonical PIPELINES/JUDGED_PIPELINES name below is
translated at that one boundary and nowhere else.

Writes results/phase9_sample_report.json.

Usage:
    python scripts/analyze_phase9_sample.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.answer_metrics import exact_match, is_abstention, token_f1
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.legacy_pipeline_names import to_legacy_name

PIPELINES = ("dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag")
JUDGED_PIPELINES = ("hybrid_reranker", "agentic_multi_hop", "adaptive_rag")
SAMPLE_PATH = "results/phase9_sample.json"
DEV_SPLIT_FILE = "dev_subset.json"


def _load_sample_records(pipeline: str, sample_ids: set[str]) -> dict[str, dict]:
    raw = json.loads((PROJECT_ROOT / "results" / f"phase9_{to_legacy_name(pipeline)}_raw.json").read_text())
    return {r["qa_id"]: r for r in raw["records"] if r["qa_id"] in sample_ids}


def _load_judge_scores(pipeline: str) -> dict[str, dict]:
    d = json.loads((PROJECT_ROOT / "results" / f"phase9_judge_{to_legacy_name(pipeline)}.json").read_text())
    return {r["qa_id"]: r for r in d["records"]}


def main() -> None:
    sample = json.loads((PROJECT_ROOT / SAMPLE_PATH).read_text())
    sample_ids = set(sample["qa_ids"])
    print(f"Loaded sample: {len(sample_ids)} qa_ids (seed={sample['seed']})")

    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    dev_records = load_qa_records(dev_path)
    dev_by_qa_id = {qa_id(r): r for r in dev_records}
    gold_docs_by_qa_id = {qid: gold_doc_ids(r) for qid, r in dev_by_qa_id.items() if qid in sample_ids}

    pipeline_records = {p: _load_sample_records(p, sample_ids) for p in PIPELINES}
    for p in PIPELINES:
        assert len(pipeline_records[p]) == 50, f"{p} has {len(pipeline_records[p])}/50 sample records"

    judge_scores = {p: _load_judge_scores(p) for p in JUDGED_PIPELINES}

    null_ids = {qid for qid in sample_ids if dev_by_qa_id[qid].question_type == "null_query"}
    non_null_ids = sample_ids - null_ids
    print(f"non-null: {len(non_null_ids)}, null: {len(null_ids)}")

    # --- item 1: deterministic metrics for all 5 pipelines --------------------------------
    deterministic_metrics: dict[str, dict] = {}
    for p in PIPELINES:
        recs = pipeline_records[p]
        em_vals = [exact_match(recs[qid]["answer"], recs[qid]["gold_answer"]) for qid in non_null_ids]
        f1_vals = [token_f1(recs[qid]["answer"], recs[qid]["gold_answer"]) for qid in non_null_ids]
        abstain_correct = [int(is_abstention(recs[qid]["answer"])) for qid in null_ids]
        deterministic_metrics[p] = {
            "normalized_exact_match": sum(em_vals) / len(em_vals),
            "token_f1": sum(f1_vals) / len(f1_vals),
            "null_query_abstention_accuracy": sum(abstain_correct) / len(abstain_correct),
            "n_non_null": len(em_vals), "n_null": len(abstain_correct),
        }

    # --- item 2: judge scores for the 3 judged pipelines -----------------------------------
    judge_summary: dict[str, dict] = {}
    for p in JUDGED_PIPELINES:
        scores = [judge_scores[p][qid]["score"] for qid in non_null_ids]
        n_fallback = sum(1 for qid in non_null_ids if judge_scores[p][qid]["fallback_used"])
        judge_summary[p] = {
            "mean_judge_score": sum(scores) / len(scores),
            "n_correct": sum(1 for s in scores if s == 1.0),
            "n_partially_correct": sum(1 for s in scores if s == 0.5),
            "n_incorrect": sum(1 for s in scores if s == 0.0),
            "n_judge_fallbacks": n_fallback,
        }

    # --- combined per-query quality (judge score for non-null, abstention-correct for null) -
    def combined_quality(pipeline: str, qid: str) -> float:
        if qid in null_ids:
            return float(is_abstention(pipeline_records[pipeline][qid]["answer"]))
        return judge_scores[pipeline][qid]["score"]

    combined_quality_mean = {
        p: sum(combined_quality(p, qid) for qid in sample_ids) / 50 for p in JUDGED_PIPELINES
    }

    # --- item 3: Adaptive RAG quality retention vs Agentic Multi-Hop RAG -------------------
    adaptive_quality = combined_quality_mean["adaptive_rag"]
    agentic_quality = combined_quality_mean["agentic_multi_hop"]
    quality_retention_pct = (adaptive_quality / agentic_quality) if agentic_quality > 0 else None

    # --- evidence/retrieval coverage (gold docs found in the chunks actually used) ----------
    def evidence_coverage(pipeline: str, qid: str) -> float:
        gold = gold_docs_by_qa_id[qid]
        if not gold:  # null_query
            return None
        used = set(pipeline_records[pipeline][qid]["evidence_doc_ids_used"])
        return len(gold & used) / len(gold)

    evidence_coverage_mean = {}
    for p in PIPELINES:
        vals = [evidence_coverage(p, qid) for qid in non_null_ids]
        evidence_coverage_mean[p] = sum(vals) / len(vals)

    # --- item 5/6: cost/latency reduction (already established this session) ---------------
    ag_costs = [pipeline_records["agentic_multi_hop"][qid]["total_cost_usd"] for qid in sample_ids]
    ad_costs = [pipeline_records["adaptive_rag"][qid]["total_cost_usd"] for qid in sample_ids]
    ag_lat = [pipeline_records["agentic_multi_hop"][qid]["total_latency_ms"] for qid in sample_ids]
    ad_lat = [pipeline_records["adaptive_rag"][qid]["total_latency_ms"] for qid in sample_ids]
    mean_ag_cost, mean_ad_cost = sum(ag_costs) / 50, sum(ad_costs) / 50
    mean_ag_lat, mean_ad_lat = sum(ag_lat) / 50, sum(ad_lat) / 50
    cost_reduction_pct = (mean_ag_cost - mean_ad_cost) / mean_ag_cost
    latency_reduction_pct = (mean_ag_lat - mean_ad_lat) / mean_ag_lat

    # --- item 7: quality breakdown by query type and hop count -----------------------------
    from mhrag.eval.ground_truth import hop_count

    def group_breakdown(group_fn) -> dict:
        groups: dict[str, list[str]] = {}
        for qid in sample_ids:
            groups.setdefault(group_fn(dev_by_qa_id[qid]), []).append(qid)
        breakdown = {}
        for key, qids in groups.items():
            row = {"n": len(qids)}
            for p in JUDGED_PIPELINES:
                vals = [combined_quality(p, qid) for qid in qids]
                row[f"{p}_mean_quality"] = sum(vals) / len(vals)
            breakdown[key] = row
        return breakdown

    breakdown_by_question_type = group_breakdown(lambda r: r.question_type)
    breakdown_by_hop_count = group_breakdown(
        lambda r: f"hop{hop_count(r)}" if r.question_type != "null_query" else "null"
    )

    # --- item 8: under-routed Adaptive RAG failures -----------------------------------------
    under_routed_failures = []
    for qid in non_null_ids:
        route = pipeline_records["adaptive_rag"][qid]["predicted_route"]
        if route not in ("SIMPLE", "MEDIUM"):
            continue
        ad_score = judge_scores["adaptive_rag"][qid]["score"]
        ag_score = judge_scores["agentic_multi_hop"][qid]["score"]
        ad_cov = evidence_coverage("adaptive_rag", qid)
        ag_cov = evidence_coverage("agentic_multi_hop", qid)
        if ad_score < ag_score or ad_cov < ag_cov:
            under_routed_failures.append(
                {
                    "qa_id": qid, "question_type": dev_by_qa_id[qid].question_type,
                    "hop_count": hop_count(dev_by_qa_id[qid]), "route": route,
                    "adaptive_rag_judge_score": ad_score, "agentic_multi_hop_judge_score": ag_score,
                    "adaptive_rag_evidence_coverage": ad_cov, "agentic_multi_hop_evidence_coverage": ag_cov,
                }
            )

    # --- item 9: judge calls/tokens/latency --------------------------------------------------
    judge_call_stats = {}
    all_judge_records = [r for p in JUDGED_PIPELINES for r in judge_scores[p].values()]
    judge_call_stats["n_judge_calls"] = len(all_judge_records)
    judge_call_stats["total_input_tokens"] = sum(r["input_tokens"] or 0 for r in all_judge_records)
    judge_call_stats["total_output_tokens"] = sum(r["output_tokens"] or 0 for r in all_judge_records)
    judge_call_stats["mean_latency_ms"] = sum(r["latency_ms"] for r in all_judge_records) / len(all_judge_records)
    judge_call_stats["n_fallbacks"] = sum(1 for r in all_judge_records if r["fallback_used"])
    judge_call_stats["total_cost_usd"] = None  # pricing unverified — never guessed

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 9 (reduced 50-question scope) — answer-quality evaluation report",
        "sample_seed": sample["seed"],
        "sample_size": 50,
        "n_non_null": len(non_null_ids),
        "n_null": len(null_ids),
        "deterministic_metrics": deterministic_metrics,
        "judge_scores": judge_summary,
        "combined_quality_mean": combined_quality_mean,
        "adaptive_quality_retention_pct_vs_agentic_multi_hop": quality_retention_pct,
        "evidence_coverage_mean": evidence_coverage_mean,
        "cost_latency": {
            "agentic_multi_hop_mean_cost_usd": mean_ag_cost, "adaptive_rag_mean_cost_usd": mean_ad_cost,
            "agentic_multi_hop_mean_latency_ms": mean_ag_lat, "adaptive_rag_mean_latency_ms": mean_ad_lat,
            "cost_reduction_pct": cost_reduction_pct, "latency_reduction_pct": latency_reduction_pct,
        },
        "breakdown_by_question_type": breakdown_by_question_type,
        "breakdown_by_hop_count": breakdown_by_hop_count,
        "under_routed_failures": under_routed_failures,
        "judge_call_stats": judge_call_stats,
    }
    out_path = PROJECT_ROOT / "results" / "phase9_sample_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
