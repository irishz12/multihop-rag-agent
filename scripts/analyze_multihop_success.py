#!/usr/bin/env python
"""Phase 6A (dev-only, ZERO LLM/API calls — pure aggregation over already-
persisted artifacts): multi-hop success analysis — did later agentic hops
recover required (gold) evidence hop 1 missed, and did that translate into
a better graded answer? Document-level, all-hops, offline. See
mhrag.eval.multihop_success's module docstring for why this is deliberately
NOT blended with mhrag.eval.fact_grounding's Tier A/Tier B fact-level
numbers (different metric, different scope).

POPULATIONS (both reported, never silently switched):
  - population_all_multihop (n=92): every dev question with
    num_agent_hops > 1 in results/phase9_always_agentic_raw.json — the
    primary Agentic population for this analysis.
  - population_three_way (n=86): the subset of the above also present in
    results/phase9_hybrid_reranker_raw.json AND
    results/phase9_hybrid_reranker_matched_full_raw.json — used ONLY for
    the 3-pipeline evidence-coverage comparison, since that comparison
    needs a qa_id to exist in all three raw files.

INPUTS (all read-only, all already-committed):
  - data/processed/dev_subset.json (hardcoded dev split, no CLI flag)
  - results/phase9_always_agentic_raw.json, phase9_hybrid_reranker_raw.json,
    phase9_hybrid_reranker_matched_full_raw.json
  - results/fact_grounding_replay_raw.json (Phase 5A) — hop-1 doc set,
    reused, never recomputed: `hybrid_reranker.replayed_doc_ids_unique` for
    a given qa_id is mathematically identical to agentic hop 1 (same
    query, same frozen retrieval code, same top_k — see Phase 5A/5B).
  - results/phase9_judge_always_agentic(.json|_extended73.json),
    results/phase9_judge_hybrid_reranker(.json|_extended73.json),
    results/phase9_judge_hybrid_reranker_matched_full.json

The population and exclusion list are fixed BEFORE this script inspects
any outcome — see EXCLUDED_QA_IDS below, applied only to example
selection, never to the aggregate counts (92/86/33 must never move
depending on what the aggregates turn out to show).

Writes ONLY results/multihop_success_analysis.json — never modifies any
existing results/*.json.

Usage:
    python scripts/analyze_multihop_success.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.multihop_success import (
    ExampleCandidate,
    QuestionOutcome,
    added_required_evidence,
    classify_tier,
    coverage,
    select_examples,
)

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json

AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"
REPLAY_FILE = "results/fact_grounding_replay_raw.json"

JUDGE_AGENTIC_FILES = ("results/phase9_judge_always_agentic.json", "results/phase9_judge_always_agentic_extended73.json")
JUDGE_BASELINE_FILES = ("results/phase9_judge_hybrid_reranker.json", "results/phase9_judge_hybrid_reranker_extended73.json")
JUDGE_MATCHED_FILES = ("results/phase9_judge_hybrid_reranker_matched_full.json",)

# Known evaluator-quirk case, excluded from EXAMPLE SELECTION ONLY — see
# the Task Success hardening phases' error analysis for why this qa_id's
# judge grade is unreliable. This exclusion is fixed here, in code, BEFORE
# any outcome is computed, and never applied to the aggregate population
# counts (92 / 86 / 33 below are unaffected by this list).
EXCLUDED_QA_IDS = frozenset({"03ea05f6e99ffb38"})
EXCLUSION_REASON = (
    "03ea05f6e99ffb38: known Task Success evaluator quirk identified during Phase 3/4 hardening "
    "(judge grade unreliable on this qa_id's answer text) — excluded from example selection only, "
    "not from any aggregate count in this report."
)

OUTPUT_FILE = "results/multihop_success_analysis.json"  # this script's ONLY write target
MAX_EXAMPLES = 5


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def _union_judge_grades(paths: tuple[str, ...]) -> dict[str, str]:
    grades: dict[str, str] = {}
    for path in paths:
        for rec in _load(path)["records"]:
            qid = rec["qa_id"]
            if qid in grades and grades[qid] != rec["grade"]:
                raise ValueError(f"conflicting judge grade for {qid} across {paths}")
            grades[qid] = rec["grade"]
    return grades


def main() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)
    records_by_qa_id = {compute_qa_id(r): r for r in all_records}

    agentic_raw = {r["qa_id"]: r for r in _load(AGENTIC_RAW_FILE)["records"]}
    baseline_raw = {r["qa_id"]: r for r in _load(BASELINE_RAW_FILE)["records"]}
    matched_raw = {r["qa_id"]: r for r in _load(MATCHED_RAW_FILE)["records"]}
    replay = _load(REPLAY_FILE)["records"]  # dict keyed by qa_id

    agentic_grades = _union_judge_grades(JUDGE_AGENTIC_FILES)
    baseline_grades = _union_judge_grades(JUDGE_BASELINE_FILES)
    matched_grades = _union_judge_grades(JUDGE_MATCHED_FILES)

    # --- population_all_multihop (n=92): primary Agentic population -----------------------
    all_multihop_qa_ids = sorted(qid for qid, r in agentic_raw.items() if r.get("num_agent_hops", 0) > 1)

    outcomes: dict[str, QuestionOutcome] = {}
    for qid in all_multihop_qa_ids:
        record = records_by_qa_id.get(qid)
        if record is None:
            raise ValueError(f"qa_id {qid} from {AGENTIC_RAW_FILE} not found in {DEV_SPLIT_FILE}")
        agentic_rec = agentic_raw[qid]
        replay_entry = replay.get(qid)
        hop1_doc_ids = (
            frozenset(replay_entry["hybrid_reranker"]["replayed_doc_ids_unique"]) if replay_entry else frozenset()
        )
        baseline_rec = baseline_raw.get(qid)
        matched_rec = matched_raw.get(qid)
        outcomes[qid] = QuestionOutcome(
            qa_id=qid,
            question_type=record.question_type,
            stop_reason=agentic_rec["stop_reason"],
            num_agent_hops=agentic_rec["num_agent_hops"],
            gold_doc_ids=gold_doc_ids(record),
            hop1_doc_ids=hop1_doc_ids,
            final_doc_ids=frozenset(agentic_rec["evidence_doc_ids_used"]),
            baseline_doc_ids=frozenset(baseline_rec["evidence_doc_ids_used"]) if baseline_rec else None,
            matched_doc_ids=frozenset(matched_rec["evidence_doc_ids_used"]) if matched_rec else None,
            agentic_grade=agentic_grades.get(qid),
            baseline_grade=baseline_grades.get(qid),
            matched_grade=matched_grades.get(qid),
        )

    # --- population_three_way (n=86): subset present in all three raw files ---------------
    three_way_qa_ids = sorted(
        qid for qid in all_multihop_qa_ids if qid in baseline_raw and qid in matched_raw
    )

    def _mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    evidence_coverage_three_way = {
        "agentic_final_all_hops": _mean(
            [c for qid in three_way_qa_ids if (c := coverage(outcomes[qid].final_doc_ids, outcomes[qid].gold_doc_ids)) is not None]
        ),
        "baseline_hybrid_reranker": _mean(
            [c for qid in three_way_qa_ids if (c := coverage(outcomes[qid].baseline_doc_ids, outcomes[qid].gold_doc_ids)) is not None]
        ),
        "context_matched": _mean(
            [c for qid in three_way_qa_ids if (c := coverage(outcomes[qid].matched_doc_ids, outcomes[qid].gold_doc_ids)) is not None]
        ),
    }

    # --- evidence coverage before vs after iteration, population_all_multihop (n=92) ------
    before_after = {
        "agentic_hop1_only_mean": _mean(
            [c for qid in all_multihop_qa_ids if (c := coverage(outcomes[qid].hop1_doc_ids, outcomes[qid].gold_doc_ids)) is not None]
        ),
        "agentic_final_all_hops_mean": _mean(
            [c for qid in all_multihop_qa_ids if (c := coverage(outcomes[qid].final_doc_ids, outcomes[qid].gold_doc_ids)) is not None]
        ),
        "per_question": [
            {
                "qa_id": qid,
                "coverage_before_hop1": coverage(outcomes[qid].hop1_doc_ids, outcomes[qid].gold_doc_ids),
                "coverage_after_all_hops": coverage(outcomes[qid].final_doc_ids, outcomes[qid].gold_doc_ids),
            }
            for qid in all_multihop_qa_ids
        ],
    }

    # --- 33/92 later-hop-added-required-evidence -------------------------------------------
    added_evidence_qa_ids = sorted(qid for qid in all_multihop_qa_ids if added_required_evidence(outcomes[qid]))

    by_type: dict[str, int] = {}
    for qid in added_evidence_qa_ids:
        by_type[outcomes[qid].question_type] = by_type.get(outcomes[qid].question_type, 0) + 1

    # --- final judge outcome breakdown, within the 33 --------------------------------------
    judge_covered = [qid for qid in added_evidence_qa_ids if outcomes[qid].agentic_grade is not None]
    beats_both = [qid for qid in judge_covered if classify_tier(outcomes[qid]) == 1]
    beats_one = [qid for qid in judge_covered if classify_tier(outcomes[qid]) == 2]
    ties_or_losses = [qid for qid in judge_covered if classify_tier(outcomes[qid]) is None]

    # --- example selection: pure, deterministic function ------------------------------------
    candidates = [
        ExampleCandidate(qa_id=qid, question_type=outcomes[qid].question_type,
                          stop_reason=outcomes[qid].stop_reason, tier=t)
        for qid in added_evidence_qa_ids
        if (t := classify_tier(outcomes[qid])) is not None
    ]
    selected_qa_ids = select_examples(candidates, excluded_qa_ids=EXCLUDED_QA_IDS, max_examples=MAX_EXAMPLES)

    selected_examples_detail = []
    for qid in selected_qa_ids:
        o = outcomes[qid]
        record = records_by_qa_id[qid]
        added = sorted(added_required_evidence(o))
        selected_examples_detail.append({
            "qa_id": qid,
            "question": record.query,
            "question_type": o.question_type,
            "gold_answer": record.answer,
            "num_agent_hops": o.num_agent_hops,
            "stop_reason": o.stop_reason,
            "gold_doc_ids": sorted(o.gold_doc_ids),
            "hop1_doc_ids": sorted(o.hop1_doc_ids),
            "final_doc_ids_all_hops": sorted(o.final_doc_ids),
            "newly_added_required_doc_ids": added,
            "agentic_answer": agentic_raw[qid]["answer"],
            "baseline_answer": baseline_raw.get(qid, {}).get("answer"),
            "matched_answer": matched_raw.get(qid, {}).get("answer"),
            "agentic_grade": o.agentic_grade,
            "baseline_grade": o.baseline_grade,
            "matched_grade": o.matched_grade,
            "tier": classify_tier(o),
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 6A — multi-hop success analysis: did later agentic hops recover required gold "
                   "evidence hop 1 missed, document-level, all-hops, dev-only, ZERO LLM/API calls. NOT "
                   "fact-level grounding — see mhrag.eval.multihop_success module docstring for the "
                   "explicit non-mixing boundary with mhrag.eval.fact_grounding's Tier A/Tier B.",
        "scope_label": "doc-level multi-hop success analysis (all hops)",
        "populations": {
            "population_all_multihop_n": len(all_multihop_qa_ids),
            "population_all_multihop_qa_ids": all_multihop_qa_ids,
            "population_three_way_n": len(three_way_qa_ids),
            "population_three_way_qa_ids": three_way_qa_ids,
            "note": "population_all_multihop is the PRIMARY Agentic population (num_agent_hops>1 in "
                    "phase9_always_agentic_raw.json). population_three_way is used ONLY for the "
                    "3-pipeline evidence-coverage comparison, since that comparison requires a qa_id "
                    "present in the baseline and context-matched raw files too.",
        },
        "evidence_coverage_three_way": evidence_coverage_three_way,
        "evidence_coverage_before_after_iteration": before_after,
        "added_required_evidence": {
            "n": len(added_evidence_qa_ids),
            "denominator": len(all_multihop_qa_ids),
            "pct": len(added_evidence_qa_ids) / len(all_multihop_qa_ids),
            "qa_ids": added_evidence_qa_ids,
            "by_question_type": by_type,
        },
        "final_judge_outcome_breakdown": {
            "judge_covered_n": len(judge_covered),
            "denominator": len(added_evidence_qa_ids),
            "beats_both_baselines": {"n": len(beats_both), "qa_ids": beats_both},
            "beats_one_baseline": {"n": len(beats_one), "qa_ids": beats_one},
            "ties_or_losses": {"n": len(ties_or_losses), "qa_ids": ties_or_losses},
        },
        "excluded_qa_ids": {"qa_ids": sorted(EXCLUDED_QA_IDS), "reason": EXCLUSION_REASON,
                             "scope": "example selection only — never applied to any aggregate count above"},
        "selected_example_qa_ids": selected_qa_ids,
        "selected_examples_detail": selected_examples_detail,
        "denominators": {
            "dev_non_null_records_loaded": len(all_records),
            "population_all_multihop_n": len(all_multihop_qa_ids),
            "population_three_way_n": len(three_way_qa_ids),
            "added_required_evidence_n": len(added_evidence_qa_ids),
            "judge_covered_within_added_evidence_n": len(judge_covered),
            "selected_examples_n": len(selected_qa_ids),
        },
    }

    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"population_all_multihop_n={len(all_multihop_qa_ids)}  population_three_way_n={len(three_way_qa_ids)}")
    print(f"evidence_coverage_three_way={evidence_coverage_three_way}")
    print(f"added_required_evidence: {len(added_evidence_qa_ids)}/{len(all_multihop_qa_ids)}")
    print(f"beats_both={len(beats_both)} beats_one={len(beats_one)} ties_or_losses={len(ties_or_losses)}")
    print(f"selected_example_qa_ids={selected_qa_ids}")


if __name__ == "__main__":
    main()
