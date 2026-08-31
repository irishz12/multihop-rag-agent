#!/usr/bin/env python
"""Phase 5A, STEPS 3 + 5 (dev-only, offline, ZERO LLM/API calls,
VALIDATION ONLY — no fact_grounded_rate is computed here): compares the
replayed document IDs (results/fact_grounding_replay_raw.json, written by
scripts/replay_retrieval_for_grounding_validation.py) against the
ALREADY-PERSISTED original document IDs in:

  - results/phase9_hybrid_reranker_raw.json           (pipeline A)
  - results/phase9_hybrid_reranker_matched_full_raw.json (pipeline B)
  - results/phase9_always_agentic_raw.json            (pipeline C — FINAL
    persisted evidence pool, which may include hops 2-3 for multi-hop
    questions; hop-1 replay fidelity is reported SEPARATELY and is never
    presented as "the Agentic pipeline's fidelity" — see module docstring
    for why an exact match is NOT expected for multi-hop-resolved
    questions here, by design, not by replay failure)

STEP 3: exact-set-match rate, partial-overlap rate, zero-overlap rate,
original/replay document counts, per pipeline.

STEP 5: a reproducible (fixed seed) stratified random sample of >=20
qa_ids across question types and all three pipelines, printing original
vs. replayed doc IDs, overlap, and replayed chunk count — for manual
inspection.

Writes ONLY results/fact_grounding_replay_fidelity.json — never modifies
any existing results/*.json or the Step 2 replay artifact.

Usage:
    python scripts/compute_fact_grounding_replay_fidelity.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT

REPLAY_FILE = "results/fact_grounding_replay_raw.json"  # READ-ONLY — Step 2's own output
BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
OUTPUT_FILE = "results/fact_grounding_replay_fidelity.json"  # this script's ONLY write target

SAMPLE_SEED = 2029  # same seed this project's evaluation sampling already uses
SAMPLE_SIZE_MIN = 20


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def _fidelity_stats(pairs: list[tuple[set[str], set[str]]]) -> dict:
    """`pairs` = list of (original_doc_set, replayed_doc_set)."""
    n = len(pairs)
    exact = sum(1 for o, r in pairs if o == r)
    zero_overlap = sum(1 for o, r in pairs if o and r and not (o & r))
    partial = sum(1 for o, r in pairs if o != r and (o & r))
    return {
        "n": n,
        "exact_match_count": exact, "exact_match_rate": exact / n if n else None,
        "partial_overlap_count": partial, "partial_overlap_rate": partial / n if n else None,
        "zero_overlap_count": zero_overlap, "zero_overlap_rate": zero_overlap / n if n else None,
        "mean_original_doc_count": sum(len(o) for o, _ in pairs) / n if n else None,
        "mean_replayed_doc_count": sum(len(r) for _, r in pairs) / n if n else None,
    }


def main() -> None:
    replay = _load(REPLAY_FILE)["records"]
    baseline_raw = {r["qa_id"]: r for r in _load(BASELINE_RAW_FILE)["records"]}
    matched_raw = {r["qa_id"]: r for r in _load(MATCHED_RAW_FILE)["records"]}
    agentic_raw = {r["qa_id"]: r for r in _load(AGENTIC_RAW_FILE)["records"]}

    # --- STEP 3: fidelity by pipeline ---------------------------------------------------------
    pairs_a, pairs_b, pairs_c_hop1_vs_final = [], [], []
    qa_ids_missing_original = {"A": [], "B": [], "C": []}

    for qa_id, entry in replay.items():
        replayed_a_docs = set(entry["hybrid_reranker"]["replayed_doc_ids_unique"])
        if qa_id in baseline_raw:
            pairs_a.append((set(baseline_raw[qa_id]["evidence_doc_ids_used"]), replayed_a_docs))
        else:
            qa_ids_missing_original["A"].append(qa_id)

        if entry.get("hybrid_reranker_matched") and qa_id in matched_raw:
            replayed_b_docs = set(entry["hybrid_reranker_matched"]["replayed_doc_ids_unique"])
            pairs_b.append((set(matched_raw[qa_id]["evidence_doc_ids_used"]), replayed_b_docs))
        elif qa_id not in matched_raw:
            qa_ids_missing_original["B"].append(qa_id)

        if qa_id in agentic_raw:
            replayed_c_docs = set(entry["agentic_hop1"]["replayed_doc_ids_unique"])
            pairs_c_hop1_vs_final.append((set(agentic_raw[qa_id]["evidence_doc_ids_used"]), replayed_c_docs))
        else:
            qa_ids_missing_original["C"].append(qa_id)

    fidelity = {
        "hybrid_reranker_full_pipeline_replay": _fidelity_stats(pairs_a),
        "hybrid_reranker_matched_full_pipeline_replay": _fidelity_stats(pairs_b),
        "agentic_hop1_vs_FINAL_persisted_evidence_pool": {
            **_fidelity_stats(pairs_c_hop1_vs_final),
            "warning": "This compares hop-1-ONLY replayed evidence against the FINAL persisted "
                       "Agentic evidence pool, which may include hops 2-3 for multi-hop-resolved "
                       "questions. A low exact-match rate here is EXPECTED and does NOT indicate "
                       "replay failure — see breakdown_by_num_agent_hops below.",
        },
        "qa_ids_missing_original_record": qa_ids_missing_original,
    }

    # breakdown of C's fidelity split by how many hops the original agentic run actually took —
    # this is the honest way to present hop-1 fidelity without overclaiming
    single_hop_pairs, multi_hop_pairs = [], []
    for qa_id, entry in replay.items():
        if qa_id not in agentic_raw:
            continue
        rec = agentic_raw[qa_id]
        pair = (set(rec["evidence_doc_ids_used"]), set(entry["agentic_hop1"]["replayed_doc_ids_unique"]))
        if rec["num_agent_hops"] <= 1:
            single_hop_pairs.append(pair)
        else:
            multi_hop_pairs.append(pair)
    fidelity["agentic_hop1_breakdown_by_original_hop_count"] = {
        "single_hop_originals_n_agent_hops_1_or_0": _fidelity_stats(single_hop_pairs),
        "multi_hop_originals_n_agent_hops_2_or_3": _fidelity_stats(multi_hop_pairs),
    }

    # --- STEP 5: reproducible stratified spot-check sample -----------------------------------
    by_type: dict[str, list[str]] = {}
    for qa_id, entry in replay.items():
        by_type.setdefault(entry["question_type"], []).append(qa_id)

    rng = random.Random(SAMPLE_SEED)
    per_type = max(1, SAMPLE_SIZE_MIN // max(1, len(by_type)))
    sample_ids: list[str] = []
    for qtype, ids in sorted(by_type.items()):
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        sample_ids.extend(ids_sorted[:per_type + 1])  # +1 so 3 types x 7 >= 20
    sample_ids = sample_ids[:max(SAMPLE_SIZE_MIN, len(sample_ids))]

    spot_check = []
    for qa_id in sample_ids:
        entry = replay[qa_id]
        row = {
            "qa_id": qa_id, "question_type": entry["question_type"],
            "query": entry["hybrid_reranker"]["query"], "pipelines": {},
        }
        for label, raw_lookup, replay_key in (
            ("hybrid_reranker", baseline_raw, "hybrid_reranker"),
            ("hybrid_reranker_matched", matched_raw, "hybrid_reranker_matched"),
            ("agentic_hop1_vs_final", agentic_raw, "agentic_hop1"),
        ):
            if qa_id not in raw_lookup or entry.get(replay_key) is None:
                continue
            original_docs = set(raw_lookup[qa_id]["evidence_doc_ids_used"])
            replayed_docs = set(entry[replay_key]["replayed_doc_ids_unique"])
            row["pipelines"][label] = {
                "original_doc_ids": sorted(original_docs),
                "replayed_doc_ids": sorted(replayed_docs),
                "overlap": sorted(original_docs & replayed_docs),
                "exact_match": original_docs == replayed_docs,
                "num_replayed_chunks": entry[replay_key]["num_replayed_chunks"],
            }
        spot_check.append(row)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 5A STEPS 3+5 (validation only, no fact_grounded_rate computed) — replay "
                   "fidelity by pipeline + reproducible spot-check sample",
        "replay_fidelity_by_pipeline": fidelity,
        "spot_check_sample": {"seed": SAMPLE_SEED, "n": len(spot_check), "records": spot_check},
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}\n")

    print("=" * 90)
    print("STEP 3: REPLAY FIDELITY BY PIPELINE")
    print("=" * 90)
    for name, stats in fidelity.items():
        if name in ("qa_ids_missing_original_record",):
            continue
        print(f"\n{name}:")
        print(json.dumps(stats, indent=2))

    print("\n" + "=" * 90)
    print(f"STEP 5: SPOT-CHECK SAMPLE (n={len(spot_check)}, seed={SAMPLE_SEED})")
    print("=" * 90)
    for row in spot_check:
        print(f"\nqa_id={row['qa_id']} type={row['question_type']} query={row['query'][:80]!r}")
        for label, p in row["pipelines"].items():
            print(f"  [{label}] original={p['original_doc_ids']} replayed={p['replayed_doc_ids']} "
                  f"exact={p['exact_match']} n_chunks={p['num_replayed_chunks']}")


if __name__ == "__main__":
    main()
