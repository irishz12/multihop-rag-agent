#!/usr/bin/env python
"""Phase 5B (dev-only, ZERO LLM/API calls — local retrieval/rerank models
only): retrieval-side fact-level grounding — "did the required gold facts
reach the generation context?" NOT "did the generated answer use them"
(see mhrag.eval.fact_grounding's module docstring for the explicit scope
boundary this script must never blur in its own output labeling).

FLOW, per Phase 5B's explicit requirement — reproduces the actual
generation context, not raw retrieval output:

    query (record.query, NEVER Evidence.fact/gold answer)
      -> existing, UNMODIFIED rerank_hybrid_search (dense+BM25->RRF->rerank)
      -> existing, UNMODIFIED assemble_context (SAME token-budget logic
         every production pipeline uses: max_context_tokens=4500,
         approximate_token_count)
      -> chunks that actually survive truncation into chunks_included
      -> mhrag.eval.fact_grounding.compute_question_fact_grounding

PIPELINES (query and retrieval config match the ALREADY-VALIDATED Phase
5A replay exactly — see results/fact_grounding_replay_raw.json, used here
as a cross-check, not re-derived from scratch):

  - hybrid_reranker: final_top_k=GENERATION_TOP_K=5, all 265 non-null
    development questions.
  - hybrid_reranker_matched: final_top_k=N (N = the already-persisted
    Agentic num_chunks_used_for_generation for that qa_id, same source
    Phase 5A/this session's earlier context-matched ablation used),
    n=117 (the subset with a known Agentic target-N).
  - agentic_hop1: IDENTICAL computation to hybrid_reranker (same query,
    same retrieval config — see Phase 5A's own documented finding this is
    mathematically guaranteed, not a shortcut) — reused, not recomputed,
    then split into two tiers by the ALREADY-PERSISTED
    results/phase9_always_agentic_raw.json's num_agent_hops:
      Tier A (num_agent_hops <= 1, n=31): hop-1 replay EQUALS the
        original final evidence pool — full-fidelity, headline-eligible
        WITHIN this tier only.
      Tier B (num_agent_hops > 1, n=86): only hop-1 is reconstructable —
        reported as an explicit LOWER BOUND, never blended with Tier A,
        never described as "Agentic fact grounding" unqualified.

CROSS-CHECK AGAINST PHASE 5A (per the Phase 5B brief's explicit
regression/fidelity requirement): before trusting the fact-grounding
numbers, this script verifies its own freshly-computed PRE-BUDGET
document sets against Phase 5A's already-persisted replay document sets
— expected 265/265 and ~116/117 exact matches, matching Phase 5A's
findings exactly. It also verifies that applying assemble_context's
token-budget truncation (which Phase 5A's replay deliberately did not do)
resolves the one discrepancy Phase 5A diagnosed
(qa_id=c2c7d987ec96ecb8, target_n=12, 1 chunk dropped for budget) — i.e.
this script's POST-budget doc set for that qa_id should now exactly match
results/phase9_hybrid_reranker_matched_full_raw.json's persisted
evidence_doc_ids_used, closing the loop on Phase 5A's finding.

Writes ONLY results/fact_grounding_report.json — never modifies any
existing results/*.json, the Phase 5A artifacts, mhrag.retrieval.*,
mhrag.generation.*, or any config.

Usage:
    python scripts/compute_fact_grounding.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.data.schema import doc_id_from_url
from mhrag.eval.fact_grounding import GoldFact, compute_question_fact_grounding
from mhrag.eval.task_success_metrics import (
    bonferroni_alpha,
    paired_bootstrap_ci,
    paired_delta_summary,
    proportion,
    wilson_ci,
)
from mhrag.generation.context import approximate_token_count, assemble_context
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker, rerank_hybrid_search

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"  # READ-ONLY — target-N + hop-count source
MATCHED_ORIGINAL_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"  # READ-ONLY — fidelity cross-check
PHASE5A_REPLAY_FILE = "results/fact_grounding_replay_raw.json"  # READ-ONLY — Phase 5A fidelity cross-check
GENERATION_TOP_K = 5  # matches scripts/run_phase9_benchmark.py's GENERATION_TOP_K exactly
MAX_CONTEXT_TOKENS = 4500  # matches configs/agent.yaml's loop.max_context_tokens exactly (read at runtime below)
OUTPUT_FILE = "results/fact_grounding_report.json"  # this script's ONLY write target

PRIMARY_COMPARISON_FAMILY = ("context_matched_vs_agentic_tier_a", "context_matched_vs_hybrid_reranker")


def _load(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text())


def _gold_facts(record) -> list[GoldFact]:
    return [GoldFact(fact=e.fact, doc_id=doc_id_from_url(e.url)) for e in record.evidence_list]


def main() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")
    agent_config = load_config("configs/agent.yaml")
    max_context_tokens = agent_config["loop"]["max_context_tokens"]
    assert max_context_tokens == MAX_CONTEXT_TOKENS, "configs/agent.yaml drifted from this script's expectation"

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    all_records = load_qa_records(dev_path)
    non_null = [r for r in all_records if r.question_type != "null_query"]
    records_by_qa_id = {compute_qa_id(r): r for r in non_null}
    print(f"Non-null development questions: {len(non_null)}")

    agentic_raw = {r["qa_id"]: r for r in _load(AGENTIC_RAW_FILE)["records"]}
    target_n_by_qa_id = {qid: r["num_chunks_used_for_generation"] for qid, r in agentic_raw.items()}
    hops_by_qa_id = {qid: r["num_agent_hops"] for qid, r in agentic_raw.items()}

    phase5a_replay = _load(PHASE5A_REPLAY_FILE)["records"]
    matched_original_raw = {r["qa_id"]: r for r in _load(MATCHED_ORIGINAL_RAW_FILE)["records"]}

    print(f"Loading embedding model {retrieval_config['embedding']['model_name']} ...")
    embedding_model = EmbeddingModel(
        model_name=retrieval_config["embedding"]["model_name"],
        device=retrieval_config["embedding"].get("device"),
        normalize=retrieval_config["embedding"]["normalize"],
        query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
        batch_size=retrieval_config["embedding"]["batch_size"],
    )
    print(f"Loading BM25 model {retrieval_config['bm25']['model_name']} ...")
    bm25_model = Bm25Model(model_name=retrieval_config["bm25"]["model_name"])
    print(f"Loading reranker model {retrieval_config['reranker']['model_name']} ...")
    reranker = Reranker(
        model_name=retrieval_config["reranker"]["model_name"],
        device=retrieval_config["reranker"].get("device"),
        batch_size=retrieval_config["reranker"]["batch_size"],
    )
    qdrant_client = get_client(retrieval_config["qdrant"]["url"])
    collection_name = retrieval_config["qdrant"]["collection_name"]

    per_question: dict[str, dict] = {}
    fidelity_checks = {
        "hybrid_reranker_vs_phase5a": {"n": 0, "exact": 0},
        "hybrid_reranker_matched_prebudget_vs_phase5a": {"n": 0, "exact": 0},
        "hybrid_reranker_matched_postbudget_vs_original_persisted": {"n": 0, "exact": 0, "mismatches": []},
    }

    for i, record in enumerate(non_null):
        qid = compute_qa_id(record)
        query = record.query  # THE ONLY THING passed to retrieval — never Evidence.fact, never gold answer
        gold_facts = _gold_facts(record)

        # --- hybrid_reranker (also = agentic_hop1, identical computation) ---
        results_a = rerank_hybrid_search(
            query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            final_top_k=GENERATION_TOP_K,
        )
        assembled_a = assemble_context(
            results_a, approximate_token_count, top_k=GENERATION_TOP_K, max_context_tokens=max_context_tokens
        )
        chunk_texts_a = [c.text for c in assembled_a.chunks_included]
        fg_a = compute_question_fact_grounding(qid, record.question_type, gold_facts, chunk_texts_a)

        # cross-check vs Phase 5A's persisted replay (pre-budget doc set)
        p5a_entry = phase5a_replay.get(qid)
        if p5a_entry:
            fidelity_checks["hybrid_reranker_vs_phase5a"]["n"] += 1
            prebudget_docs_a = {r.doc_id for r in results_a}
            if prebudget_docs_a == set(p5a_entry["hybrid_reranker"]["replayed_doc_ids_unique"]):
                fidelity_checks["hybrid_reranker_vs_phase5a"]["exact"] += 1

        entry = {
            "qa_id": qid, "question_type": record.question_type,
            "n_agent_hops_original": hops_by_qa_id.get(qid),
            "hybrid_reranker": {
                "n_gold_facts": fg_a.n_gold_facts, "n_grounded_facts": fg_a.n_grounded_facts,
                "fact_grounded_rate": fg_a.fact_grounded_rate, "per_fact_grounded": list(fg_a.per_fact_grounded),
                "num_context_chunks": len(chunk_texts_a),
            },
        }
        # agentic_hop1 is the identical computation — reused, not recomputed
        entry["agentic_hop1"] = dict(entry["hybrid_reranker"])

        # --- hybrid_reranker_matched (context-matched) ---
        target_n = target_n_by_qa_id.get(qid)
        if target_n is not None:
            results_b = rerank_hybrid_search(
                query, qdrant_client, collection_name, embedding_model, bm25_model, reranker,
                final_top_k=target_n,
            )
            assembled_b = assemble_context(
                results_b, approximate_token_count, top_k=target_n, max_context_tokens=max_context_tokens
            )
            chunk_texts_b = [c.text for c in assembled_b.chunks_included]
            fg_b = compute_question_fact_grounding(qid, record.question_type, gold_facts, chunk_texts_b)
            entry["hybrid_reranker_matched"] = {
                "n_gold_facts": fg_b.n_gold_facts, "n_grounded_facts": fg_b.n_grounded_facts,
                "fact_grounded_rate": fg_b.fact_grounded_rate, "per_fact_grounded": list(fg_b.per_fact_grounded),
                "num_context_chunks": len(chunk_texts_b), "target_n": target_n,
            }

            if p5a_entry and p5a_entry.get("hybrid_reranker_matched"):
                fidelity_checks["hybrid_reranker_matched_prebudget_vs_phase5a"]["n"] += 1
                prebudget_docs_b = {r.doc_id for r in results_b}
                if prebudget_docs_b == set(p5a_entry["hybrid_reranker_matched"]["replayed_doc_ids_unique"]):
                    fidelity_checks["hybrid_reranker_matched_prebudget_vs_phase5a"]["exact"] += 1

            if qid in matched_original_raw:
                fidelity_checks["hybrid_reranker_matched_postbudget_vs_original_persisted"]["n"] += 1
                postbudget_docs_b = {c.doc_id for c in assembled_b.chunks_included}
                original_docs_b = set(matched_original_raw[qid]["evidence_doc_ids_used"])
                if postbudget_docs_b == original_docs_b:
                    fidelity_checks["hybrid_reranker_matched_postbudget_vs_original_persisted"]["exact"] += 1
                else:
                    fidelity_checks["hybrid_reranker_matched_postbudget_vs_original_persisted"]["mismatches"].append(
                        {"qa_id": qid, "original": sorted(original_docs_b), "replayed_postbudget": sorted(postbudget_docs_b)}
                    )
        else:
            entry["hybrid_reranker_matched"] = None

        per_question[qid] = entry
        if (i + 1) % 50 == 0 or (i + 1) == len(non_null):
            print(f"  [{i + 1}/{len(non_null)}] qa_id={qid}")

    # --- aggregate ------------------------------------------------------------------------
    def _pipeline_summary(pipeline_key: str, qa_ids: list[str]) -> dict:
        rates = [per_question[q][pipeline_key]["fact_grounded_rate"] for q in qa_ids
                 if per_question[q][pipeline_key] is not None and per_question[q][pipeline_key]["fact_grounded_rate"] is not None]
        all_flags = [f for q in qa_ids if per_question[q][pipeline_key] is not None
                     for f in per_question[q][pipeline_key]["per_fact_grounded"]]
        n_pooled = len(all_flags)
        n_pooled_grounded = sum(all_flags)
        summary = {
            "n_questions": len(rates),
            "mean_per_question_fact_grounded_rate": sum(rates) / len(rates) if rates else None,
            "n_pooled_facts": n_pooled,
            "pooled_fact_grounded_rate": proportion(n_pooled_grounded, n_pooled) if n_pooled else None,
        }
        if n_pooled:
            ci = wilson_ci(n_pooled_grounded, n_pooled)
            summary["pooled_fact_grounded_rate_ci_95"] = (ci.lower, ci.upper)
        return summary

    def _by_question_type(pipeline_key: str, qa_ids: list[str]) -> dict:
        by_type: dict[str, list[str]] = {}
        for q in qa_ids:
            by_type.setdefault(records_by_qa_id[q].question_type, []).append(q)
        return {qt: _pipeline_summary(pipeline_key, ids) for qt, ids in sorted(by_type.items())}

    all_qa_ids = list(per_question.keys())
    matched_qa_ids = [q for q in all_qa_ids if per_question[q]["hybrid_reranker_matched"] is not None]
    tier_a_qa_ids = [q for q in matched_qa_ids if hops_by_qa_id.get(q, 0) <= 1]
    tier_b_qa_ids = [q for q in matched_qa_ids if hops_by_qa_id.get(q, 0) > 1]

    pipeline_summaries = {
        "hybrid_reranker": {"scope": f"n={len(all_qa_ids)}, all non-null development questions",
                             **_pipeline_summary("hybrid_reranker", all_qa_ids),
                             "by_question_type": _by_question_type("hybrid_reranker", all_qa_ids)},
        "hybrid_reranker_matched": {"scope": f"n={len(matched_qa_ids)}, subset with a known Agentic target-N",
                                     **_pipeline_summary("hybrid_reranker_matched", matched_qa_ids),
                                     "by_question_type": _by_question_type("hybrid_reranker_matched", matched_qa_ids)},
        "agentic_tier_a_full_fidelity": {
            "scope": f"n={len(tier_a_qa_ids)}, num_agent_hops<=1 — hop-1 replay EQUALS the original final "
                     "evidence pool for these questions; headline-eligible WITHIN this tier only",
            **_pipeline_summary("agentic_hop1", tier_a_qa_ids),
            "by_question_type": _by_question_type("agentic_hop1", tier_a_qa_ids),
        },
        "agentic_tier_b_hop1_lower_bound": {
            "scope": f"n={len(tier_b_qa_ids)}, num_agent_hops>1 — HOP-1 ONLY, an INCOMPLETE LOWER BOUND on the "
                     "true multi-hop Agentic fact-grounding rate — hops 2-3's queries cannot be reconstructed "
                     "from persisted artifacts (see Phase 5 audit, Phase 5A). NEVER combine with Tier A.",
            **_pipeline_summary("agentic_hop1", tier_b_qa_ids),
            "by_question_type": _by_question_type("agentic_hop1", tier_b_qa_ids),
        },
    }

    # --- paired comparisons (only the two informative ones — see script docstring) --------
    def _paired_rates(key_a: str, ids_a_key: str, key_b: str, qa_ids: list[str]) -> tuple[list[float], list[float]]:
        a_vals, b_vals = [], []
        for q in qa_ids:
            ea = per_question[q][key_a]
            eb = per_question[q][key_b]
            if ea is None or eb is None or ea["fact_grounded_rate"] is None or eb["fact_grounded_rate"] is None:
                continue
            a_vals.append(ea["fact_grounded_rate"])
            b_vals.append(eb["fact_grounded_rate"])
        return a_vals, b_vals

    corrected_alpha = bonferroni_alpha(0.05, len(PRIMARY_COMPARISON_FAMILY))  # 0.025 per comparison -> 97.5% CI
    corrected_confidence = round(1 - corrected_alpha, 3)  # 0.975 — a supported confidence level (task_success_metrics)

    paired_comparisons = {}

    a_vals, b_vals = _paired_rates("hybrid_reranker_matched", None, "agentic_hop1", tier_a_qa_ids)
    if len(a_vals) >= 2:
        deltas = [b - a for a, b in zip(a_vals, b_vals)]  # agentic_tier_a - context_matched
        s = paired_delta_summary(b_vals, a_vals)
        paired_comparisons["context_matched_vs_agentic_tier_a"] = {
            "n": s.n, "mean_delta_agentic_minus_matched": s.mean_delta, "median_delta": s.median_delta,
            "stdev_delta": s.stdev_delta, "ci_95_nominal": (s.ci.lower, s.ci.upper) if s.ci else None,
            "ci_bonferroni_corrected": None if s.stdev_delta == 0 else
                (lambda c: (c.lower, c.upper))(paired_bootstrap_ci(deltas, confidence=corrected_confidence)),
            "cohens_d": s.cohens_d,
            "note": "For Tier A (num_agent_hops<=1), Agentic hop-1's retrieval config and query are "
                    "IDENTICAL to hybrid_reranker's, and Context-Matched's target-N for these single-hop "
                    "originals is itself sourced from the Agentic run's own realized chunk count — so this "
                    "comparison is a TAUTOLOGICAL check on this tier (both sides compute the same thing for "
                    "single-hop questions by construction), not evidence that iteration adds value. See report.",
        }

    a_vals2, b_vals2 = _paired_rates("hybrid_reranker", None, "hybrid_reranker_matched", matched_qa_ids)
    if len(a_vals2) >= 2:
        deltas2 = [b - a for a, b in zip(a_vals2, b_vals2)]  # matched - baseline5
        s2 = paired_delta_summary(b_vals2, a_vals2)
        paired_comparisons["context_matched_vs_hybrid_reranker"] = {
            "n": s2.n, "mean_delta_matched_minus_baseline5": s2.mean_delta, "median_delta": s2.median_delta,
            "stdev_delta": s2.stdev_delta, "ci_95_nominal": (s2.ci.lower, s2.ci.upper) if s2.ci else None,
            "ci_bonferroni_corrected": (lambda c: (c.lower, c.upper))(
                paired_bootstrap_ci(deltas2, confidence=corrected_confidence)
            ),
            "cohens_d": s2.cohens_d,
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 5B — RETRIEVAL-SIDE fact-level grounding (did required gold facts reach the "
                   "generation context; NOT whether the generated answer used them), dev-only, zero LLM/API calls",
        "scope_label": "retrieval-side fact-level grounding",
        "fact_matching_method": "exact/normalized substring, single-chunk only (see mhrag.eval.fact_grounding)",
        "pipeline_summaries": pipeline_summaries,
        "paired_comparisons": paired_comparisons,
        "bonferroni_alpha_per_comparison": corrected_alpha,
        "primary_comparison_family": list(PRIMARY_COMPARISON_FAMILY),
        "phase5a_fidelity_cross_check": fidelity_checks,
        "denominators": {
            "hybrid_reranker_n": len(all_qa_ids), "hybrid_reranker_matched_n": len(matched_qa_ids),
            "agentic_tier_a_n": len(tier_a_qa_ids), "agentic_tier_b_n": len(tier_b_qa_ids),
        },
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps({k: v for k, v in report.items() if k not in ("pipeline_summaries",)}, indent=2)[:4000])
    print(json.dumps(pipeline_summaries, indent=2)[:4000])


if __name__ == "__main__":
    main()
