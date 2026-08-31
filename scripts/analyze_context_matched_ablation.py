#!/usr/bin/env python
"""AUDIT ABLATION (dev-only) — OFFLINE three-way analysis, NO LLM/API calls,
NO Mantle client, NO access to final_holdout.json anywhere.

Reads only already-persisted DEVELOPMENT-split artifacts:
  - results/phase9_sample.json                          (the frozen 50-qa_id sample)
  - results/phase9_hybrid_reranker_raw.json              (existing, fixed 5 chunks)
  - results/phase9_judge_hybrid_reranker.json            (existing)
  - results/phase9_always_agentic_raw.json               (existing, Agentic Multi-Hop RAG)
  - results/phase9_judge_always_agentic.json             (existing)
  - results/phase9_hybrid_reranker_matched_raw.json      (NEW, this ablation)
  - results/phase9_judge_hybrid_reranker_matched.json    (NEW, this ablation)
  - data/processed/dev_subset.json                       (DEVELOPMENT split only, for gold_doc_ids)

Writes ONLY results/context_matched_ablation_report.json — never modifies
any input file.

Does NOT declare a causal conclusion — reports the three-way numbers,
paired deltas, bootstrap CIs, and effect sizes; interpretation against
Cases 1/2/3 is left to the reader (and stated explicitly in the printed
summary as evidence-plus-uncertainty, not a verdict).

Usage:
    python scripts/analyze_context_matched_ablation.py
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

SAMPLE_FILE = "results/phase9_sample.json"
BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
BASELINE_JUDGE_FILE = "results/phase9_judge_hybrid_reranker.json"
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
AGENTIC_JUDGE_FILE = "results/phase9_judge_always_agentic.json"
MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_raw.json"
MATCHED_JUDGE_FILE = "results/phase9_judge_hybrid_reranker_matched.json"
OUTPUT_FILE = "results/context_matched_ablation_report.json"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2029  # same seed the project already uses for its stratified evaluation samples
N_COMPARISONS_FOR_CORRECTION = 2  # (Agentic vs Matched) and (Matched vs Baseline) — Bonferroni family size


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def _records_by_qa_id(raw: dict) -> dict[str, dict]:
    return {r["qa_id"]: r for r in raw["records"]}


def combined_quality(judge_by_qa_id: dict, raw_by_qa_id: dict, null_ids: set[str], qid: str) -> float:
    """Judge score for non-null; abstention correctness for null — same
    construction analyze_phase9_holdout.py already uses."""
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
    """Percentile bootstrap CI on the mean of `deltas` (paired differences,
    one per question). Standard nonparametric technique — not previously
    used anywhere in this codebase (mhrag.eval has no existing CI/bootstrap
    utility; this audit's own report flagged that gap), implemented fresh
    here rather than borrowed from a nonexistent prior implementation."""
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
    sample = _load(SAMPLE_FILE)
    sample_ids = set(sample["qa_ids"])

    baseline_raw = _records_by_qa_id(_load(BASELINE_RAW_FILE))
    baseline_judge = _records_by_qa_id(_load(BASELINE_JUDGE_FILE))
    agentic_raw = _records_by_qa_id(_load(AGENTIC_RAW_FILE))
    agentic_judge = _records_by_qa_id(_load(AGENTIC_JUDGE_FILE))

    matched_raw_path = PROJECT_ROOT / MATCHED_RAW_FILE
    matched_judge_path = PROJECT_ROOT / MATCHED_JUDGE_FILE
    if not matched_raw_path.exists() or not matched_judge_path.exists():
        raise SystemExit(
            f"Ablation not yet complete — expected both {MATCHED_RAW_FILE} and {MATCHED_JUDGE_FILE} to exist. "
            "Run scripts/run_phase9_context_matched_ablation.py then scripts/run_context_matched_judge.py first."
        )
    matched_raw = _records_by_qa_id(_load(MATCHED_RAW_FILE))
    matched_judge = _records_by_qa_id(_load(MATCHED_JUDGE_FILE))

    missing = [q for q in sample_ids if q not in matched_raw]
    if missing:
        raise SystemExit(f"Ablation raw file is missing {len(missing)} sample qa_id(s): {missing}")

    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / "dev_subset.json"
    dev_records = {compute_qa_id(r): r for r in load_qa_records(dev_path)}
    gold_by_qa_id = {q: gold_doc_ids(dev_records[q]) for q in sample_ids}

    null_ids = {q for q in sample_ids if agentic_raw[q]["question_type"] == "null_query"}
    non_null_ids = sample_ids - null_ids
    print(f"Sample: {len(sample_ids)} total ({len(null_ids)} null, {len(non_null_ids)} non-null)")

    pipelines = {
        "hybrid_reranker_5chunk": (baseline_raw, baseline_judge),
        "hybrid_reranker_context_matched": (matched_raw, matched_judge),
        "agentic_multi_hop": (agentic_raw, agentic_judge),
    }

    # --- context-matching quality -----------------------------------------------------------
    target_n = [matched_raw[q]["target_n_chunks"] for q in sample_ids]
    matched_actual_n = [matched_raw[q]["num_chunks_used_for_generation"] for q in sample_ids]
    agentic_actual_n = [agentic_raw[q]["num_chunks_used_for_generation"] for q in sample_ids]
    mismatches = [q for q in sample_ids if matched_raw[q]["num_chunks_used_for_generation"] != agentic_raw[q]["num_chunks_used_for_generation"]]

    def _dist(vals: list[int]) -> dict[str, int]:
        d: dict[str, int] = {}
        for v in vals:
            d[str(v)] = d.get(str(v), 0) + 1
        return dict(sorted(d.items(), key=lambda kv: int(kv[0])))

    matching_quality = {
        "target_n_distribution": _dist(target_n),
        "matched_actual_n_distribution": _dist(matched_actual_n),
        "agentic_actual_n_distribution_same_qa_ids": _dist(agentic_actual_n),
        "n_mismatched_vs_agentic_realized_count": len(mismatches),
        "mismatch_rate": len(mismatches) / len(sample_ids),
        "mismatched_qa_ids": mismatches,
        "mean_target_n": st.mean(target_n), "mean_matched_actual_n": st.mean(matched_actual_n),
        "mean_agentic_actual_n": st.mean(agentic_actual_n),
        "mean_matched_total_tokens": st.mean(matched_raw[q]["total_token_count"] for q in sample_ids),
    }

    # --- per-pipeline per-question metrics ----------------------------------------------------
    per_q: dict[str, dict[str, dict]] = {}
    for name, (raw, judge) in pipelines.items():
        per_q[name] = {
            q: {
                "combined_quality": combined_quality(judge, raw, null_ids, q),
                "evidence_coverage": evidence_coverage(raw, gold_by_qa_id, q),
                "num_chunks_used_for_generation": raw[q]["num_chunks_used_for_generation"],
                "total_cost_usd": raw[q].get("total_cost_usd"),
                "total_latency_ms": raw[q].get("total_latency_ms"),
                "generation_latency_ms": raw[q].get("generation_latency_ms"),
                "retrieval_latency_ms": raw[q].get("retrieval_latency_ms"),
            }
            for q in sample_ids
        }

    # --- aggregate summary per pipeline -------------------------------------------------------
    summary = {}
    for name in pipelines:
        cq_all = [per_q[name][q]["combined_quality"] for q in sample_ids]
        ec_non_null = [per_q[name][q]["evidence_coverage"] for q in non_null_ids if per_q[name][q]["evidence_coverage"] is not None]
        costs = [per_q[name][q]["total_cost_usd"] for q in sample_ids if per_q[name][q]["total_cost_usd"] is not None]
        lats = [per_q[name][q]["total_latency_ms"] for q in sample_ids if per_q[name][q]["total_latency_ms"] is not None]
        chunks = [per_q[name][q]["num_chunks_used_for_generation"] for q in sample_ids]
        summary[name] = {
            "n": len(sample_ids),
            "combined_quality_mean": st.mean(cq_all), "combined_quality_median": st.median(cq_all),
            "evidence_coverage_mean": st.mean(ec_non_null) if ec_non_null else None,
            "mean_chunks_used": st.mean(chunks), "median_chunks_used": st.median(chunks),
            "mean_cost_usd": st.mean(costs) if costs else None,
            "mean_latency_ms": st.mean(lats) if lats else None,
        }

    # --- paired comparisons (same 50 qa_ids both sides, matched pairing by qa_id) -------------
    def paired_deltas(a: str, b: str, ids: set[str]) -> list[float]:
        return [per_q[a][q]["combined_quality"] - per_q[b][q]["combined_quality"] for q in ids]

    comparisons = {
        "agentic_vs_matched": paired_deltas("agentic_multi_hop", "hybrid_reranker_context_matched", sample_ids),
        "matched_vs_baseline5": paired_deltas("hybrid_reranker_context_matched", "hybrid_reranker_5chunk", sample_ids),
        "agentic_vs_baseline5": paired_deltas("agentic_multi_hop", "hybrid_reranker_5chunk", sample_ids),
    }

    nominal_alpha = 0.05
    bonferroni_alpha = nominal_alpha / N_COMPARISONS_FOR_CORRECTION  # applied to the two primary contrasts

    paired_stats = {}
    for label, deltas in comparisons.items():
        alpha_for_this = bonferroni_alpha if label in ("agentic_vs_matched", "matched_vs_baseline5") else nominal_alpha
        ci_nominal = paired_bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, nominal_alpha)
        ci_corrected = paired_bootstrap_ci(deltas, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, alpha_for_this)
        paired_stats[label] = {
            "n": len(deltas),
            "mean_delta": st.mean(deltas),
            "median_delta": st.median(deltas),
            "stdev_delta": st.stdev(deltas) if len(deltas) > 1 else None,
            "ci_95_nominal": ci_nominal,
            "ci_bonferroni_corrected": ci_corrected,
            "bonferroni_alpha_used": alpha_for_this if label in ("agentic_vs_matched", "matched_vs_baseline5") else None,
            "excludes_zero_nominal_95": not (ci_nominal[0] <= 0.0 <= ci_nominal[1]),
            "excludes_zero_bonferroni": not (ci_corrected[0] <= 0.0 <= ci_corrected[1]),
            "cohens_d_paired": cohens_d_paired(deltas),
        }

    report = {
        "purpose": "AUDIT ABLATION (dev-only, offline) — three-way comparison: Hybrid+Reranker (5 chunks) vs. "
                   "Hybrid+Reranker context-matched (single pass) vs. Agentic Multi-Hop RAG",
        "sample_seed": sample.get("seed"), "sample_size": len(sample_ids),
        "n_null": len(null_ids), "n_non_null": len(non_null_ids),
        "context_matching_quality": matching_quality,
        "pipeline_summary": summary,
        "paired_comparisons": paired_stats,
        "bootstrap_config": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                              "method": "percentile bootstrap on the mean of per-question paired deltas"},
        "multiple_comparison_correction": {
            "family": ["agentic_vs_matched", "matched_vs_baseline5"],
            "method": "Bonferroni", "n_comparisons": N_COMPARISONS_FOR_CORRECTION,
            "corrected_alpha_per_comparison": bonferroni_alpha,
        },
        "note": "No causal conclusion is declared here — see the printed CASE 1/2/3 interpretation, "
                "which is descriptive of the numbers below, not asserted as proven.",
    }

    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    # --- human-readable summary --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CONTEXT-MATCHING QUALITY")
    print("=" * 78)
    print(f"target N distribution:          {matching_quality['target_n_distribution']}")
    print(f"matched actual-N distribution:   {matching_quality['matched_actual_n_distribution']}")
    print(f"agentic actual-N (same qa_ids):  {matching_quality['agentic_actual_n_distribution_same_qa_ids']}")
    print(f"mismatch vs agentic's own realized count: {matching_quality['n_mismatched_vs_agentic_realized_count']}/{len(sample_ids)} "
          f"({matching_quality['mismatch_rate']:.1%})")
    print(f"mean target N={matching_quality['mean_target_n']:.2f}  "
          f"mean matched actual N={matching_quality['mean_matched_actual_n']:.2f}  "
          f"mean agentic actual N={matching_quality['mean_agentic_actual_n']:.2f}")

    print("\n" + "=" * 78)
    print("THREE-WAY PIPELINE SUMMARY (n=50 each, same qa_ids)")
    print("=" * 78)
    header = f"{'pipeline':38}{'quality':>10}{'evid.cov':>10}{'chunks':>9}{'cost($)':>11}{'lat(ms)':>10}"
    print(header)
    for name, s in summary.items():
        print(f"{name:38}{s['combined_quality_mean']:>10.3f}"
              f"{(s['evidence_coverage_mean'] or 0):>10.3f}{s['mean_chunks_used']:>9.2f}"
              f"{(s['mean_cost_usd'] or 0):>11.6f}{(s['mean_latency_ms'] or 0):>10.0f}")

    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS (combined quality, per-question delta, n=50)")
    print("=" * 78)
    for label, stats in paired_stats.items():
        print(f"\n-- {label} --")
        print(f"   mean delta={stats['mean_delta']:+.4f}  median={stats['median_delta']:+.4f}  "
              f"stdev={stats['stdev_delta']:.4f}")
        print(f"   95% bootstrap CI (nominal):        [{stats['ci_95_nominal'][0]:+.4f}, {stats['ci_95_nominal'][1]:+.4f}]"
              f"  excludes 0: {stats['excludes_zero_nominal_95']}")
        if stats["bonferroni_alpha_used"] is not None:
            print(f"   Bonferroni-corrected CI (alpha={stats['bonferroni_alpha_used']:.3f}): "
                  f"[{stats['ci_bonferroni_corrected'][0]:+.4f}, {stats['ci_bonferroni_corrected'][1]:+.4f}]"
                  f"  excludes 0: {stats['excludes_zero_bonferroni']}")
        print(f"   Cohen's d (paired): {stats['cohens_d_paired']}")

    print("\n" + "=" * 78)
    print("DESCRIPTIVE READ AGAINST CASES 1/2/3 (not an automatic causal conclusion)")
    print("=" * 78)
    am = paired_stats["agentic_vs_matched"]
    mb = paired_stats["matched_vs_baseline5"]
    print(f"Agentic vs Context-Matched:  mean delta={am['mean_delta']:+.4f}, CI excludes 0 (nominal)={am['excludes_zero_nominal_95']}")
    print(f"Context-Matched vs Baseline-5: mean delta={mb['mean_delta']:+.4f}, CI excludes 0 (nominal)={mb['excludes_zero_nominal_95']}")
    print("Read the printed numbers above against Case 1 (Agentic >> Matched), Case 2 (Agentic ≈ Matched >> Baseline-5),")
    print("or Case 3 (Matched between the two) — this script deliberately does not pick one for you.")


if __name__ == "__main__":
    main()
