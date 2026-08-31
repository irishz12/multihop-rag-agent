#!/usr/bin/env python
"""Task Success (dev-only, offline, not a Phase 9 script): recomputes the
deterministic-first Task Success classification (`mhrag.eval.task_success`)
against ALREADY-PERSISTED development-split artifacts — the existing
Hybrid+Reranker 5-chunk baseline, this session's Context-Matched
single-pass ablation, and the existing Agentic Multi-Hop RAG pipeline.

ZERO NEW LLM/API CALLS. Every judge grade, every evidence-coverage input,
every generated answer used here was already computed and persisted by an
earlier script this session (or earlier Phase 9 work) — this script only
reads existing JSON and applies the pure, offline `mhrag.eval.task_success`
functions to it.

TWO POPULATIONS, kept explicitly separate rather than silently merged:

  1. NON-NULL QUALITY POPULATION (n=117): every eligible non-null dev
     question with full judge coverage for all three pipelines (baseline,
     context-matched, agentic) — see this session's Option B work. Used
     for deterministic correctness, judge-vs-deterministic agreement, and
     the unsupported-answer rate.

  2. ABSTENTION POPULATIONS (per pipeline, DIFFERENT sizes — reported
     honestly, not equalized): the Context-Matched ablation was built to
     exclude null_query entirely (it targets a non-null-only confound), so
     it has ZERO null_query records and its abstention analysis covers
     only the incorrect_abstention / normal_non_abstention cells. Baseline
     (Hybrid+Reranker) covers the full 300-question development
     population, including all 35 null_query records — the richest
     population available. Agentic Multi-Hop RAG covers whatever was
     already generated (123 records: 117 non-null + 6 null, from the
     original Phase 9 sample). See §E of the printed report for the exact
     n behind every abstention number.

DEV-ONLY BY CONSTRUCTION: every input path below is a hardcoded module
constant already known to be development-split — none reads
data/processed/final_holdout.json or any results/phase9_holdout_*.json
file. See tests/test_compute_task_success_guard.py.

Writes ONLY results/task_success_report.json — never modifies any
existing results/*.json.

Usage:
    python scripts/compute_task_success.py --validate    # small hand-checked sample first
    python scripts/compute_task_success.py               # full report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.task_success import classify_task_success
from mhrag.eval.task_success_metrics import (
    bonferroni_alpha,
    paired_delta_summary,
    proportion,
    wilson_ci,
)

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json

BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
BASELINE_JUDGE_FILE = "results/phase9_judge_hybrid_reranker.json"
BASELINE_JUDGE_EXT_FILE = "results/phase9_judge_hybrid_reranker_extended73.json"

AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
AGENTIC_JUDGE_FILE = "results/phase9_judge_always_agentic.json"
AGENTIC_JUDGE_EXT_FILE = "results/phase9_judge_always_agentic_extended73.json"

MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"
MATCHED_JUDGE_FILE = "results/phase9_judge_hybrid_reranker_matched_full.json"

OUTPUT_FILE = "results/task_success_report_v3.json"  # this script's ONLY write target — Phase 4
# NEW filename, not an overwrite: Phase 2's results/task_success_report.json AND Phase 3's
# results/task_success_report_v2.json are both left completely untouched, per the "report both
# old and revised numbers rather than overwriting the previous artifact" requirement. See
# scripts/task_success_hardening_error_analysis.py (Phase 2->3) and
# scripts/task_success_phase4_error_analysis.py (Phase 3->4) for the direct diffs.

QUESTION_TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")
NON_NULL_QUESTION_TYPES = ("inference_query", "comparison_query", "temporal_query")

# Pre-specified comparison family for Bonferroni correction — declared explicitly here,
# not inferred, per the "define the family before calculating corrected intervals" rule.
PRIMARY_COMPARISON_FAMILY = ("agentic_vs_matched", "matched_vs_baseline5")


def _load(path: str) -> dict | None:
    p = PROJECT_ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def _records_by_qa_id(raw: dict) -> dict[str, dict]:
    return {r["qa_id"]: r for r in raw["records"]}


def _judge_union(primary_file: str, extension_file: str | None) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    primary = _load(primary_file)
    if primary:
        merged.update(_records_by_qa_id(primary))
    if extension_file:
        extension = _load(extension_file)
        if extension:
            merged.update(_records_by_qa_id(extension))
    return merged


def _evidence_coverage(raw_record: dict, gold: frozenset[str]) -> float | None:
    if not gold:
        return None
    used = set(raw_record["evidence_doc_ids_used"])
    return len(gold & used) / len(gold)


def _gold_answers_by_qa_id() -> dict[str, str]:
    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    return {compute_qa_id(r): r.answer for r in load_qa_records(dev_path)}


def _gold_doc_ids_by_qa_id() -> dict[str, frozenset[str]]:
    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    return {compute_qa_id(r): gold_doc_ids(r) for r in load_qa_records(dev_path)}


def _classify_population(
    raw_by_qa_id: dict[str, dict],
    judge_by_qa_id: dict[str, dict],
    gold_answers: dict[str, str],
    gold_docs: dict[str, frozenset[str]],
) -> dict[str, dict]:
    """Run mhrag.eval.task_success.classify_task_success over every record
    in `raw_by_qa_id`, pairing in each qa_id's already-persisted judge
    grade/score (None if never judged — e.g. null_query) and
    evidence_coverage (None if no gold docs — e.g. null_query)."""
    results: dict[str, dict] = {}
    for qid, raw in raw_by_qa_id.items():
        judge = judge_by_qa_id.get(qid)
        coverage = _evidence_coverage(raw, gold_docs.get(qid, frozenset()))
        result = classify_task_success(
            question_type=raw["question_type"],
            gold_answer=gold_answers[qid],
            generated_answer=raw["answer"],
            judge_grade=(judge["grade"] if judge else None),
            judge_score=(judge["score"] if judge else None),
            evidence_coverage=coverage,
        )
        results[qid] = {
            "question_type": result.question_type,
            "abstention_status": result.abstention_status,
            "is_abstention": result.is_abstention,
            "deterministic_match_type": result.deterministic_match_type,
            "deterministic_correctness": result.deterministic_correctness,
            "extracted_verdict": result.extracted_verdict,
            "gold_verdict": result.gold_verdict,
            "response_structure": result.response_structure,
            "entity_containment_match": result.entity_containment_match,
            "judge_grade": result.judge_grade,
            "judge_score": result.judge_score,
            "evidence_coverage": result.evidence_coverage,
            "unsupported": result.unsupported,
            "judge_deterministic_agree": result.judge_deterministic_agree,
            "task_success_confident": result.task_success_confident,
        }
    return results


def _abstention_summary(classified: dict[str, dict]) -> dict:
    null_ids = [q for q, r in classified.items() if r["question_type"] == "null_query"]
    non_null_ids = [q for q, r in classified.items() if r["question_type"] != "null_query"]

    summary: dict = {"n_null": len(null_ids), "n_non_null": len(non_null_ids)}
    if null_ids:
        correct_abstention = sum(1 for q in null_ids if classified[q]["abstention_status"] == "correct_abstention")
        summary["correct_abstention_rate"] = proportion(correct_abstention, len(null_ids))
        ca_ci = wilson_ci(correct_abstention, len(null_ids))
        summary["correct_abstention_ci_95"] = (ca_ci.lower, ca_ci.upper)
        summary["hallucinated_non_abstention_rate"] = proportion(len(null_ids) - correct_abstention, len(null_ids))
    if non_null_ids:
        incorrect_abstention = sum(1 for q in non_null_ids if classified[q]["abstention_status"] == "incorrect_abstention")
        summary["incorrect_abstention_rate"] = proportion(incorrect_abstention, len(non_null_ids))
        ia_ci = wilson_ci(incorrect_abstention, len(non_null_ids))
        summary["incorrect_abstention_ci_95"] = (ia_ci.lower, ia_ci.upper)
        summary["normal_non_abstention_rate"] = proportion(len(non_null_ids) - incorrect_abstention, len(non_null_ids))
    return summary


def _response_structure_summary(classified: dict[str, dict]) -> dict:
    """The NEW Phase 3 signal, reported alongside (never in place of)
    abstention_status. Only meaningful where abstention_status flagged
    something (correct_abstention / incorrect_abstention / hallucinated_
    non_abstention) — for normal_non_abstention records response_structure
    is almost always 'substantive_answer' by construction and adds little,
    so this breakdown is restricted to the flagged subset."""
    flagged_ids = [q for q, r in classified.items() if r["abstention_status"] != "normal_non_abstention"]
    if not flagged_ids:
        return {"n_flagged_by_abstention_status": 0}
    counts: dict[str, int] = {}
    for q in flagged_ids:
        state = classified[q]["response_structure"]
        counts[state] = counts.get(state, 0) + 1
    return {
        "n_flagged_by_abstention_status": len(flagged_ids),
        "response_structure_counts_among_flagged": counts,
        "reclassified_as_answer_with_uncertainty_rate": proportion(
            counts.get("answer_with_uncertainty", 0), len(flagged_ids)
        ),
        "reclassified_qa_ids": [
            q for q in flagged_ids if classified[q]["response_structure"] == "answer_with_uncertainty"
        ],
    }


def _deterministic_summary(classified: dict[str, dict], question_type: str) -> dict:
    ids = [q for q, r in classified.items() if r["question_type"] == question_type]
    resolved = [q for q in ids if classified[q]["deterministic_correctness"] in ("correct", "incorrect")]
    ambiguous = [q for q in ids if classified[q]["deterministic_correctness"] == "ambiguous"]
    not_applicable = [q for q in ids if classified[q]["deterministic_correctness"] == "not_applicable"]
    summary: dict = {
        "n_total": len(ids), "n_resolved": len(resolved), "n_ambiguous": len(ambiguous),
        "n_not_applicable": len(not_applicable),
    }
    if resolved:
        n_correct = sum(1 for q in resolved if classified[q]["deterministic_correctness"] == "correct")
        summary["accuracy_among_resolved"] = proportion(n_correct, len(resolved))
        acc_ci = wilson_ci(n_correct, len(resolved))
        summary["accuracy_ci_95"] = (acc_ci.lower, acc_ci.upper)
    agreement_pairs = [q for q in resolved if classified[q]["judge_deterministic_agree"] is not None]
    if agreement_pairs:
        n_agree = sum(1 for q in agreement_pairs if classified[q]["judge_deterministic_agree"])
        summary["judge_deterministic_agreement_rate"] = proportion(n_agree, len(agreement_pairs))
        summary["judge_deterministic_disagreement_n"] = len(agreement_pairs) - n_agree
        summary["disagreement_qa_ids"] = [q for q in agreement_pairs if not classified[q]["judge_deterministic_agree"]]
    return summary


def _unsupported_summary(classified: dict[str, dict]) -> dict:
    scored = [q for q, r in classified.items() if r["unsupported"] is not None]
    if not scored:
        return {"n_scored": 0}
    n_unsupported = sum(1 for q in scored if classified[q]["unsupported"])
    return {
        "n_scored": len(scored),
        "unsupported_rate": proportion(n_unsupported, len(scored)),
        "unsupported_ci_95": (lambda c: (c.lower, c.upper))(wilson_ci(n_unsupported, len(scored))),
        "unsupported_qa_ids": [q for q in scored if classified[q]["unsupported"]],
    }


def _print_validation_sample(
    baseline_classified: dict, baseline_raw: dict, gold_answers: dict, n_per_type: int = 3
) -> None:
    print("=" * 100)
    print("VALIDATION SAMPLE — inspect before trusting the full computation")
    print("=" * 100)
    by_type: dict[str, list[str]] = {}
    for qid, r in baseline_classified.items():
        by_type.setdefault(r["question_type"], []).append(qid)

    for qtype in QUESTION_TYPES:
        for qid in sorted(by_type.get(qtype, []))[:n_per_type]:
            rec = baseline_raw[qid]
            r = baseline_classified[qid]
            print(f"\nqa_id: {qid}")
            print(f"  question_type:      {qtype}")
            print(f"  gold_answer:         {gold_answers[qid]!r}")
            print(f"  generated_answer:    {rec['answer'][:200]!r}")
            print(f"  deterministic:       match_type={r['deterministic_match_type']} "
                  f"correctness={r['deterministic_correctness']} "
                  f"extracted={r['extracted_verdict']!r} gold_verdict={r['gold_verdict']!r} "
                  f"entity_containment={r['entity_containment_match']}")
            print(f"  judge_result:        grade={r['judge_grade']} score={r['judge_score']}")
            print(f"  evidence_coverage:   {r['evidence_coverage']}")
            print(f"  abstention_status:   {r['abstention_status']}  (unchanged since Phase 2)")
            print(f"  response_structure:  {r['response_structure']}  (NEW in Phase 3)")
            print(f"  unsupported:         {r['unsupported']}")
            print(f"  judge_det_agree:     {r['judge_deterministic_agree']}   confident={r['task_success_confident']}")
    print("\n" + "=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="print a small hand-checkable sample and exit")
    args = parser.parse_args()

    gold_answers = _gold_answers_by_qa_id()
    gold_docs = _gold_doc_ids_by_qa_id()

    baseline_raw_doc = _load(BASELINE_RAW_FILE)
    agentic_raw_doc = _load(AGENTIC_RAW_FILE)
    matched_raw_doc = _load(MATCHED_RAW_FILE)
    if baseline_raw_doc is None or agentic_raw_doc is None or matched_raw_doc is None:
        raise SystemExit("one or more required raw artifacts are missing — run this session's earlier scripts first")

    baseline_raw = _records_by_qa_id(baseline_raw_doc)
    agentic_raw = _records_by_qa_id(agentic_raw_doc)
    matched_raw = _records_by_qa_id(matched_raw_doc)

    baseline_judge = _judge_union(BASELINE_JUDGE_FILE, BASELINE_JUDGE_EXT_FILE)
    agentic_judge = _judge_union(AGENTIC_JUDGE_FILE, AGENTIC_JUDGE_EXT_FILE)
    matched_judge = _judge_union(MATCHED_JUDGE_FILE, None)

    baseline_classified = _classify_population(baseline_raw, baseline_judge, gold_answers, gold_docs)
    agentic_classified = _classify_population(agentic_raw, agentic_judge, gold_answers, gold_docs)
    matched_classified = _classify_population(matched_raw, matched_judge, gold_answers, gold_docs)

    if args.validate:
        _print_validation_sample(baseline_classified, baseline_raw, gold_answers)
        return

    # --- the n=117 non-null quality population: intersection across all three pipelines ---
    quality_ids = sorted(set(matched_raw.keys()) & set(agentic_raw.keys()) & set(baseline_raw.keys()))
    quality_ids = [q for q in quality_ids if baseline_raw[q]["question_type"] != "null_query"]
    print(f"Non-null quality population (all three pipelines): n={len(quality_ids)}")

    pipelines = {
        "hybrid_reranker_5chunk": (baseline_raw, baseline_classified),
        "hybrid_reranker_context_matched": (matched_raw, matched_classified),
        "agentic_multi_hop": (agentic_raw, agentic_classified),
    }

    deterministic_by_pipeline = {
        name: {qtype: _deterministic_summary(classified, qtype) for qtype in NON_NULL_QUESTION_TYPES}
        for name, (_, classified) in pipelines.items()
    }
    unsupported_by_pipeline = {name: _unsupported_summary(classified) for name, (_, classified) in pipelines.items()}
    abstention_by_pipeline = {
        "hybrid_reranker_5chunk": _abstention_summary(baseline_classified),  # full 300-question dev population
        "agentic_multi_hop": _abstention_summary(agentic_classified),  # 123 available (117 non-null + 6 null)
        "hybrid_reranker_context_matched": _abstention_summary(matched_classified),  # 117, ZERO null by ablation design
    }
    response_structure_by_pipeline = {
        "hybrid_reranker_5chunk": _response_structure_summary(baseline_classified),
        "agentic_multi_hop": _response_structure_summary(agentic_classified),
        "hybrid_reranker_context_matched": _response_structure_summary(matched_classified),
    }

    # --- paired comparisons on the n=117 quality population: task-success-confident correctness ---
    def _confident_correct_indicator(classified: dict, qid: str) -> float | None:
        r = classified[qid]
        if not r["task_success_confident"]:
            return None
        return 1.0 if r["deterministic_correctness"] == "correct" else 0.0

    def _paired_on_confident_subset(name_a: str, name_b: str) -> dict:
        _, classified_a = pipelines[name_a]
        _, classified_b = pipelines[name_b]
        common_confident = [
            q for q in quality_ids
            if _confident_correct_indicator(classified_a, q) is not None
            and _confident_correct_indicator(classified_b, q) is not None
        ]
        a_vals = [_confident_correct_indicator(classified_a, q) for q in common_confident]
        b_vals = [_confident_correct_indicator(classified_b, q) for q in common_confident]
        if len(common_confident) < 2:
            return {"n": len(common_confident), "note": "insufficient paired n (both sides task_success_confident)"}
        summary = paired_delta_summary(a_vals, b_vals)
        return {
            "n": summary.n, "mean_delta": summary.mean_delta, "median_delta": summary.median_delta,
            "stdev_delta": summary.stdev_delta,
            "ci_95_nominal": (summary.ci.lower, summary.ci.upper) if summary.ci else None,
            "cohens_d": summary.cohens_d,
        }

    def _paired_deltas(name_a: str, name_b: str) -> tuple[list[float], int] | None:
        _, classified_a = pipelines[name_a]
        _, classified_b = pipelines[name_b]
        common_confident = [
            q for q in quality_ids
            if _confident_correct_indicator(classified_a, q) is not None
            and _confident_correct_indicator(classified_b, q) is not None
        ]
        if len(common_confident) < 2:
            return None
        deltas = [
            _confident_correct_indicator(classified_a, q) - _confident_correct_indicator(classified_b, q)
            for q in common_confident
        ]
        return deltas, len(common_confident)

    paired_comparisons = {
        "agentic_vs_matched": _paired_on_confident_subset("agentic_multi_hop", "hybrid_reranker_context_matched"),
        "matched_vs_baseline5": _paired_on_confident_subset("hybrid_reranker_context_matched", "hybrid_reranker_5chunk"),
    }
    corrected_alpha = bonferroni_alpha(0.05, len(PRIMARY_COMPARISON_FAMILY))  # 0.025 per comparison
    corrected_confidence = round(1 - corrected_alpha, 3)  # 0.975 — a supported, explicitly-added confidence level

    from mhrag.eval.task_success_metrics import paired_bootstrap_ci

    family_pairs = {
        "agentic_vs_matched": ("agentic_multi_hop", "hybrid_reranker_context_matched"),
        "matched_vs_baseline5": ("hybrid_reranker_context_matched", "hybrid_reranker_5chunk"),
    }
    for label in PRIMARY_COMPARISON_FAMILY:
        deltas_and_n = _paired_deltas(*family_pairs[label])
        if deltas_and_n is None:
            paired_comparisons[label]["ci_bonferroni_corrected"] = None
            continue
        deltas, _n = deltas_and_n
        ci = paired_bootstrap_ci(deltas, confidence=corrected_confidence)
        paired_comparisons[label]["ci_bonferroni_corrected"] = (ci.lower, ci.upper)
        paired_comparisons[label]["bonferroni_alpha_used"] = corrected_alpha

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Task Success evaluation (dev-only, offline, zero new LLM calls) — deterministic-first, "
                   "judge reported separately, never used to override a deterministic failure",
        "quality_population_n": len(quality_ids),
        "quality_population_qa_ids": quality_ids,
        "primary_comparison_family": list(PRIMARY_COMPARISON_FAMILY),
        "bonferroni_alpha_per_comparison": corrected_alpha,
        "bonferroni_note": f"Comparison family = {list(PRIMARY_COMPARISON_FAMILY)} (n=2), declared explicitly "
                            f"before correction. nominal alpha 0.05 / 2 = {corrected_alpha} per comparison -> "
                            f"a {corrected_confidence:.1%} two-sided CI per comparison, computed directly (not "
                            "derived from the nominal 95% interval). No other significance threshold is applied "
                            "anywhere in this report.",
        "deterministic_correctness_by_pipeline_and_question_type": deterministic_by_pipeline,
        "unsupported_by_pipeline": unsupported_by_pipeline,
        "abstention_by_pipeline": abstention_by_pipeline,
        "response_structure_by_pipeline_new_in_phase3": response_structure_by_pipeline,
        "schema_note": "Phase 3: adds response_structure (new field, additive) and hardens verdict_match's "
                        "tier-3 fallback (clause-final restriction). abstention_status/is_abstention/"
                        "entity_containment_match/unsupported/judge fields are BYTE-IDENTICAL to Phase 2's "
                        "results/task_success_report.json — see "
                        "scripts/task_success_hardening_error_analysis.py for the exact verdict-match diff.",
        "paired_comparisons_confident_subset_only": paired_comparisons,
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2, default=list))
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2, default=list)[:3000])


if __name__ == "__main__":
    main()
