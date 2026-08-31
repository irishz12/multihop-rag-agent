#!/usr/bin/env python
"""AUDIT ABLATION, FULL-POPULATION SCALE-UP — OFFLINE three-way analysis,
NO LLM/API calls, NO Mantle client, NO access to final_holdout.json anywhere.

Reads only already-persisted DEVELOPMENT-split artifacts. For the two
EXISTING pipelines (Hybrid+Reranker 5-chunk, Agentic Multi-Hop RAG), quality
(judge-based) is computed over the UNION of their original judge file
(the 44-question sample) and their OPTIONAL extension file (up to 73 more,
from scripts/run_extended_baseline_agentic_judge.py) if it exists — read
only, never merged back into either original file. If the extension files
don't exist yet, quality comparisons fall back to whatever n the three
pipelines' judge coverage actually intersects at (typically 44), while
coverage/chunk-count/latency/cost statistics (which need no judge score)
still run at the full eligible n (117) regardless.

Writes ONLY results/context_matched_ablation_full_report.json — never
modifies any input file.

Does NOT declare a causal conclusion automatically.

Usage:
    python scripts/analyze_context_matched_ablation_full.py
"""

from __future__ import annotations

import json
import random
import statistics as st

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.answer_metrics import is_abstention
from mhrag.eval.ground_truth import gold_doc_ids

BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
BASELINE_JUDGE_FILE = "results/phase9_judge_hybrid_reranker.json"
BASELINE_JUDGE_EXT_FILE = "results/phase9_judge_hybrid_reranker_extended73.json"  # optional

AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
AGENTIC_JUDGE_FILE = "results/phase9_judge_always_agentic.json"
AGENTIC_JUDGE_EXT_FILE = "results/phase9_judge_always_agentic_extended73.json"  # optional

MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"
MATCHED_JUDGE_FILE = "results/phase9_judge_hybrid_reranker_matched_full.json"

OUTPUT_FILE = "results/context_matched_ablation_full_report.json"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2029
N_COMPARISONS_FOR_CORRECTION = 2


def _load(path: str) -> dict | None:
    p = PROJECT_ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def _records_by_qa_id(raw: dict) -> dict[str, dict]:
    return {r["qa_id"]: r for r in raw["records"]}


def _judge_union(primary_file: str, extension_file: str) -> dict[str, dict]:
    """Union of an existing judge file and its optional extension file, by
    qa_id — read-only against both, never writes either."""
    merged: dict[str, dict] = {}
    primary = _load(primary_file)
    if primary:
        merged.update(_records_by_qa_id(primary))
    extension = _load(extension_file)
    if extension:
        merged.update(_records_by_qa_id(extension))  # disjoint qa_id sets by construction; update is safe either way
    return merged


def combined_quality(judge_by_qa_id: dict, raw_by_qa_id: dict, null_ids: set[str], qid: str) -> float:
    if qid in null_ids:
        return float(is_abstention(raw_by_qa_id[qid]["answer"]))
    return judge_by_qa_id[qid]["score"]


def evidence_coverage(raw_by_qa_id: dict, gold_by_qa_id: dict, qid: str) -> float | None:
    gold = gold_by_qa_id[qid]
    if not gold:
        return None
    used = set(raw_by_qa_id[qid]["evidence_doc_ids_used"])
    return len(gold & used) / len(gold)


def paired_bootstrap_ci(deltas: list[float], n_resamples: int, seed: int, alpha: float) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return means[lo_idx], means[hi_idx]


def cohens_d_paired(deltas: list[float]) -> float | None:
    if len(deltas) < 2:
        return None
    sd = st.stdev(deltas)
    if sd == 0:
        return None
    return st.mean(deltas) / sd


def main() -> None:
    baseline_raw_doc = _load(BASELINE_RAW_FILE)
    agentic_raw_doc = _load(AGENTIC_RAW_FILE)
    matched_raw_doc = _load(MATCHED_RAW_FILE)
    if matched_raw_doc is None:
        raise SystemExit(f"{MATCHED_RAW_FILE} does not exist — run scripts/run_phase9_context_matched_ablation_full.py first")
    matched_judge_doc = _load(MATCHED_JUDGE_FILE)
    if matched_judge_doc is None:
        raise SystemExit(f"{MATCHED_JUDGE_FILE} does not exist — run scripts/run_context_matched_judge_full.py first")

    baseline_raw = _records_by_qa_id(baseline_raw_doc)
    agentic_raw = _records_by_qa_id(agentic_raw_doc)
    matched_raw = _records_by_qa_id(matched_raw_doc)
    matched_judge = _records_by_qa_id(matched_judge_doc)

    baseline_judge = _judge_union(BASELINE_JUDGE_FILE, BASELINE_JUDGE_EXT_FILE)
    agentic_judge = _judge_union(AGENTIC_JUDGE_FILE, AGENTIC_JUDGE_EXT_FILE)
    extension_files_present = (PROJECT_ROOT / BASELINE_JUDGE_EXT_FILE).exists() and (PROJECT_ROOT / AGENTIC_JUDGE_EXT_FILE).exists()

    eligible_ids = set(matched_raw.keys())  # the ablation's own eligible population (non-null by construction)
    print(f"Eligible (non-null) population for this ablation: {len(eligible_ids)}")

    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / "dev_subset.json"
    dev_records = {compute_qa_id(r): r for r in load_qa_records(dev_path)}
    gold_by_qa_id = {q: gold_doc_ids(dev_records[q]) for q in eligible_ids}

    pipelines_raw = {
        "hybrid_reranker_5chunk": baseline_raw,
        "hybrid_reranker_context_matched": matched_raw,
        "agentic_multi_hop": agentic_raw,
    }
    pipelines_judge = {
        "hybrid_reranker_5chunk": baseline_judge,
        "hybrid_reranker_context_matched": matched_judge,
        "agentic_multi_hop": agentic_judge,
    }

    # --- quality-eligible n: intersection of judge coverage across all three pipelines, within eligible_ids ---
    quality_ids = {
        q for q in eligible_ids
        if q in baseline_judge and q in agentic_judge and q in matched_judge
    }
    print(f"Quality (judge-based) comparison n: {len(quality_ids)} / {len(eligible_ids)} eligible "
          f"(extension files present: {extension_files_present})")

    null_ids: set[str] = set()  # this ablation's population is non-null by construction (see module docstring)

    # --- context-matching quality (chunk counts) — runs at FULL eligible n regardless of judge coverage --------
    target_n = [matched_raw[q]["target_n_chunks"] for q in eligible_ids]
    matched_actual_n = [matched_raw[q]["num_chunks_used_for_generation"] for q in eligible_ids]
    agentic_actual_n = [agentic_raw[q]["num_chunks_used_for_generation"] for q in eligible_ids]
    mismatches = [q for q in eligible_ids if matched_raw[q]["num_chunks_used_for_generation"] != agentic_raw[q]["num_chunks_used_for_generation"]]

    def _dist(vals: list[int]) -> dict[str, int]:
        d: dict[str, int] = {}
        for v in vals:
            d[str(v)] = d.get(str(v), 0) + 1
        return dict(sorted(d.items(), key=lambda kv: int(kv[0])))

    matching_quality = {
        "n_eligible": len(eligible_ids),
        "target_n_distribution": _dist(target_n),
        "matched_actual_n_distribution": _dist(matched_actual_n),
        "agentic_actual_n_distribution_same_qa_ids": _dist(agentic_actual_n),
        "n_mismatched_vs_agentic_realized_count": len(mismatches),
        "mismatch_rate": len(mismatches) / len(eligible_ids),
        "mean_target_n": st.mean(target_n), "mean_matched_actual_n": st.mean(matched_actual_n),
        "mean_agentic_actual_n": st.mean(agentic_actual_n),
        "mean_matched_total_tokens": st.mean(matched_raw[q]["total_token_count"] for q in eligible_ids),
    }

    # --- per-pipeline per-question metrics, computed for BOTH full-eligible (coverage/cost/latency) and
    #     quality-eligible (judge-based combined_quality) populations -----------------------------------
    per_q: dict[str, dict[str, dict]] = {}
    for name, raw in pipelines_raw.items():
        judge = pipelines_judge[name]
        per_q[name] = {}
        for q in eligible_ids:
            per_q[name][q] = {
                "combined_quality": (combined_quality(judge, raw, null_ids, q) if q in quality_ids else None),
                "evidence_coverage": evidence_coverage(raw, gold_by_qa_id, q),
                "num_chunks_used_for_generation": raw[q]["num_chunks_used_for_generation"],
                "total_cost_usd": raw[q].get("total_cost_usd"),
                "total_latency_ms": raw[q].get("total_latency_ms"),
            }

    summary = {}
    for name in pipelines_raw:
        ec_vals = [per_q[name][q]["evidence_coverage"] for q in eligible_ids if per_q[name][q]["evidence_coverage"] is not None]
        costs = [per_q[name][q]["total_cost_usd"] for q in eligible_ids if per_q[name][q]["total_cost_usd"] is not None]
        lats = [per_q[name][q]["total_latency_ms"] for q in eligible_ids if per_q[name][q]["total_latency_ms"] is not None]
        chunks = [per_q[name][q]["num_chunks_used_for_generation"] for q in eligible_ids]
        cq_vals = [per_q[name][q]["combined_quality"] for q in quality_ids if per_q[name][q]["combined_quality"] is not None]
        summary[name] = {
            "n_full_eligible": len(eligible_ids),
            "n_quality_eligible": len(cq_vals),
            "combined_quality_mean": (st.mean(cq_vals) if cq_vals else None),
            "combined_quality_median": (st.median(cq_vals) if cq_vals else None),
            "evidence_coverage_mean": st.mean(ec_vals) if ec_vals else None,
            "mean_chunks_used": st.mean(chunks), "median_chunks_used": st.median(chunks),
            "mean_cost_usd": st.mean(costs) if costs else None,
            "mean_latency_ms": st.mean(lats) if lats else None,
        }

    def paired_deltas(a: str, b: str, ids: set[str]) -> list[float]:
        return [per_q[a][q]["combined_quality"] - per_q[b][q]["combined_quality"] for q in ids
                if per_q[a][q]["combined_quality"] is not None and per_q[b][q]["combined_quality"] is not None]

    comparisons = {
        "agentic_vs_matched": paired_deltas("agentic_multi_hop", "hybrid_reranker_context_matched", quality_ids),
        "matched_vs_baseline5": paired_deltas("hybrid_reranker_context_matched", "hybrid_reranker_5chunk", quality_ids),
        "agentic_vs_baseline5": paired_deltas("agentic_multi_hop", "hybrid_reranker_5chunk", quality_ids),
    }

    nominal_alpha = 0.05
    bonferroni_alpha = nominal_alpha / N_COMPARISONS_FOR_CORRECTION

    paired_stats = {}
    for label, deltas in comparisons.items():
        if len(deltas) < 2:
            paired_stats[label] = {"n": len(deltas), "note": "insufficient paired n for statistics"}
            continue
        alpha_for_this = bonferroni_alpha if label in ("agentic_vs_matched", "matched_vs_baseline5") else nominal_alpha
        ci_nominal = paired_bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, nominal_alpha)
        ci_corrected = paired_bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, alpha_for_this)
        paired_stats[label] = {
            "n": len(deltas),
            "mean_delta": st.mean(deltas), "median_delta": st.median(deltas),
            "stdev_delta": st.stdev(deltas),
            "ci_95_nominal": ci_nominal,
            "ci_bonferroni_corrected": ci_corrected,
            "bonferroni_alpha_used": alpha_for_this if label in ("agentic_vs_matched", "matched_vs_baseline5") else None,
            "excludes_zero_nominal_95": not (ci_nominal[0] <= 0.0 <= ci_nominal[1]),
            "excludes_zero_bonferroni": not (ci_corrected[0] <= 0.0 <= ci_corrected[1]),
            "cohens_d_paired": cohens_d_paired(deltas),
        }

    report = {
        "purpose": "AUDIT ABLATION, FULL-POPULATION SCALE-UP (dev-only, offline)",
        "n_full_eligible": len(eligible_ids), "n_quality_eligible": len(quality_ids),
        "extension_judge_files_present": extension_files_present,
        "context_matching_quality": matching_quality,
        "pipeline_summary": summary,
        "paired_comparisons": paired_stats,
        "bootstrap_config": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "multiple_comparison_correction": {"method": "Bonferroni", "n_comparisons": N_COMPARISONS_FOR_CORRECTION,
                                            "corrected_alpha_per_comparison": bonferroni_alpha},
        "note": "No causal conclusion is declared here.",
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 78)
    print(f"CONTEXT-MATCHING QUALITY (n={len(eligible_ids)} full eligible population)")
    print("=" * 78)
    print(f"target N distribution:          {matching_quality['target_n_distribution']}")
    print(f"matched actual-N distribution:   {matching_quality['matched_actual_n_distribution']}")
    print(f"agentic actual-N (same qa_ids):  {matching_quality['agentic_actual_n_distribution_same_qa_ids']}")
    print(f"mismatch vs agentic's own realized count: {matching_quality['n_mismatched_vs_agentic_realized_count']}/{len(eligible_ids)} "
          f"({matching_quality['mismatch_rate']:.1%})")
    print(f"mean target N={matching_quality['mean_target_n']:.2f}  "
          f"mean matched actual N={matching_quality['mean_matched_actual_n']:.2f}  "
          f"mean agentic actual N={matching_quality['mean_agentic_actual_n']:.2f}")

    print("\n" + "=" * 78)
    print(f"THREE-WAY PIPELINE SUMMARY (coverage/cost/latency n={len(eligible_ids)}; quality n={len(quality_ids)})")
    print("=" * 78)
    header = f"{'pipeline':38}{'quality(n)':>14}{'evid.cov':>10}{'chunks':>9}{'cost($)':>11}{'lat(ms)':>10}"
    print(header)
    for name, s in summary.items():
        q_str = f"{s['combined_quality_mean']:.3f}(n={s['n_quality_eligible']})" if s['combined_quality_mean'] is not None else "n/a"
        print(f"{name:38}{q_str:>14}"
              f"{(s['evidence_coverage_mean'] or 0):>10.3f}{s['mean_chunks_used']:>9.2f}"
              f"{(s['mean_cost_usd'] or 0):>11.6f}{(s['mean_latency_ms'] or 0):>10.0f}")

    print("\n" + "=" * 78)
    print(f"PAIRED COMPARISONS (combined quality, per-question delta, n={len(quality_ids)})")
    print("=" * 78)
    for label, stats in paired_stats.items():
        print(f"\n-- {label} --")
        if "note" in stats:
            print(f"   {stats['note']}")
            continue
        print(f"   mean delta={stats['mean_delta']:+.4f}  median={stats['median_delta']:+.4f}  stdev={stats['stdev_delta']:.4f}")
        print(f"   95% bootstrap CI (nominal):        [{stats['ci_95_nominal'][0]:+.4f}, {stats['ci_95_nominal'][1]:+.4f}]"
              f"  excludes 0: {stats['excludes_zero_nominal_95']}")
        if stats["bonferroni_alpha_used"] is not None:
            print(f"   Bonferroni-corrected CI (alpha={stats['bonferroni_alpha_used']:.3f}): "
                  f"[{stats['ci_bonferroni_corrected'][0]:+.4f}, {stats['ci_bonferroni_corrected'][1]:+.4f}]"
                  f"  excludes 0: {stats['excludes_zero_bonferroni']}")
        print(f"   Cohen's d (paired): {stats['cohens_d_paired']}")


if __name__ == "__main__":
    main()
