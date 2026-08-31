#!/usr/bin/env python
"""OPTIONAL extension (dev-only, not a Phase 9 script): judges the
previously-UNJUDGED already-generated answers for the two EXISTING, frozen
pipelines (Hybrid+Reranker and Agentic Multi-Hop RAG), so the full-population
context-matched ablation's three-way QUALITY comparison can run at the same
n as the ablation itself (117 non-null questions) instead of being capped
at the original 44-question judge coverage.

WHY THIS EXISTS: results/phase9_judge_hybrid_reranker.json and
results/phase9_judge_always_agentic.json only ever judged the 50-question
Phase 9 sample's 44 non-null questions. results/phase9_hybrid_reranker_raw.json
(300 records) and results/phase9_always_agentic_raw.json (123 records, 117
non-null) already contain MORE already-generated answers than were ever
judged — this script judges the REMAINING, never-before-judged ones only
(computed as a set difference against the existing judge files at runtime),
using the exact same frozen judge model/rubric/config as every other judge
run in this project.

THIS SCRIPT NEVER GENERATES A NEW ANSWER — it only reads `answer` text that
was already produced by scripts/run_phase9_benchmark.py months before this
ablation existed, from the two existing, UNMODIFIED raw files. It is
read-only with respect to both.

DOES NOT TOUCH THE EXISTING JUDGE FILES: writes to two brand-new files
(results/phase9_judge_hybrid_reranker_extended73.json,
results/phase9_judge_always_agentic_extended73.json) containing ONLY the
newly-judged qa_ids — results/phase9_judge_hybrid_reranker.json and
results/phase9_judge_always_agentic.json are opened read-only (to compute
the set difference) and never written. The analysis script reads BOTH the
original and the extended file and takes their union at read time; neither
original file is ever modified, appended to, or overwritten.

Requires $OPENAI_API_KEY in the environment (never printed/logged/persisted).

Usage:
    python scripts/run_extended_baseline_agentic_judge.py

Writes results/phase9_judge_hybrid_reranker_extended73.json and
results/phase9_judge_always_agentic_extended73.json, checkpointed per
qa_id (resumable, never repeats a completed paid call).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.eval.judge import call_judge
from mhrag.generation.mantle_client import MantleClient, MantleConfigError

PIPELINES = {
    "hybrid_reranker": {
        "raw_file": "results/phase9_hybrid_reranker_raw.json",          # READ-ONLY, existing, unmodified
        "existing_judge_file": "results/phase9_judge_hybrid_reranker.json",  # READ-ONLY, existing, unmodified
        "extension_output_file": "results/phase9_judge_hybrid_reranker_extended73.json",  # this script's write target
    },
    "always_agentic": {
        "raw_file": "results/phase9_always_agentic_raw.json",           # READ-ONLY, existing, unmodified
        "existing_judge_file": "results/phase9_judge_always_agentic.json",  # READ-ONLY, existing, unmodified
        "extension_output_file": "results/phase9_judge_always_agentic_extended73.json",  # this script's write target
    },
}

# Only these 117 non-null qa_ids (the full-population ablation's eligible set) are ever
# in scope — read from the Agentic raw file, same eligibility definition as
# scripts/run_phase9_context_matched_ablation_full.py, so the three files stay aligned.
ELIGIBILITY_SOURCE_FILE = "results/phase9_always_agentic_raw.json"


def _eligible_ids() -> set[str]:
    agentic_raw = json.loads((PROJECT_ROOT / ELIGIBILITY_SOURCE_FILE).read_text())
    return {r["qa_id"] for r in agentic_raw["records"] if r["question_type"] != "null_query"}


def _judge_pipeline(pipeline_key: str, judge_client: MantleClient, prompt_version: str, pricing: dict, eligible_ids: set[str]) -> None:
    cfg = PIPELINES[pipeline_key]
    raw = json.loads((PROJECT_ROOT / cfg["raw_file"]).read_text())
    raw_by_qa_id = {r["qa_id"]: r for r in raw["records"]}

    existing_judge_path = PROJECT_ROOT / cfg["existing_judge_file"]
    already_judged_ids = set()
    if existing_judge_path.exists():
        already_judged_ids = {r["qa_id"] for r in json.loads(existing_judge_path.read_text())["records"]}

    to_judge_ids = sorted(q for q in eligible_ids if q not in already_judged_ids)
    print(f"\n[{pipeline_key}] eligible={len(eligible_ids)} already_judged(existing)={len(already_judged_ids & eligible_ids)} "
          f"NEW to judge={len(to_judge_ids)}")

    out_path = PROJECT_ROOT / cfg["extension_output_file"]
    existing_extension: dict[str, dict] = {}
    if out_path.exists():
        existing_extension = {r["qa_id"]: r for r in json.loads(out_path.read_text()).get("records", [])}
        print(f"[{pipeline_key}] found existing extension file with {len(existing_extension)} completed record(s) — will skip those")

    for i, qid in enumerate(to_judge_ids):
        if qid in existing_extension:
            continue
        rec = raw_by_qa_id[qid]

        result = call_judge(
            judge_client, rec["query"], rec["gold_answer"], rec["answer"],
            prompt_version=prompt_version,
            input_price_per_million=pricing["input_per_million_tokens"],
            output_price_per_million=pricing["output_per_million_tokens"],
        )
        existing_extension[qid] = {
            "qa_id": qid,
            "grade": result.verdict.grade,
            "score": result.verdict.score,
            "reason": result.verdict.reason,
            "fallback_used": result.fallback_used,
            "input_tokens": result.mantle_response.usage.input_tokens,
            "output_tokens": result.mantle_response.usage.output_tokens,
            "cost_usd": result.cost.total_cost_usd if result.cost is not None else None,
            "latency_ms": result.mantle_response.llm_latency_ms,
        }

        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": f"OPTIONAL extension (dev-only) — judges previously-unjudged, already-generated "
                       f"{pipeline_key} answers so the full-population ablation's quality comparison can "
                       f"reach n=117. Does NOT modify {cfg['existing_judge_file']}.",
            "judge_model": "see configs/judge.yaml",
            "pipeline": pipeline_key,
            "source_raw_file": cfg["raw_file"],
            "does_not_modify": cfg["existing_judge_file"],
            "n_new_questions_total": len(to_judge_ids),
            "n_questions_completed": len(existing_extension),
            "records": list(existing_extension.values()),
        }
        out_path.write_text(json.dumps(artifact, indent=2))

        if (i + 1) % 10 == 0 or (i + 1) == len(to_judge_ids):
            print(f"  [{pipeline_key} {i + 1}/{len(to_judge_ids)}] qa_id={qid} grade={result.verdict.grade}")

    print(f"[{pipeline_key}] total accumulated in extension file: {len(existing_extension)}/{len(to_judge_ids)}")
    print(f"[{pipeline_key}] wrote {out_path}")


def main() -> None:
    judge_config = load_config("configs/judge.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    try:
        judge_client = MantleClient(
            model_id=judge_config["judge"]["model_id"],
            base_url_env=mantle_config["client"]["base_url_env"],
            default_base_url=mantle_config["client"]["default_base_url"],
            api_key_env=mantle_config["client"]["api_key_env"],
            timeout_seconds=mantle_config["client"]["timeout_seconds"],
            temperature=judge_config["judge"]["temperature"],
            max_output_tokens=judge_config["judge"]["max_output_tokens"],
            max_retries=mantle_config["client"]["max_retries"],
            retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
        )
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run the extended baseline/agentic judge: {exc}") from exc

    prompt_version = judge_config["judge"]["prompt_version"]
    pricing = judge_config["pricing"]
    eligible_ids = _eligible_ids()

    for pipeline_key in PIPELINES:
        _judge_pipeline(pipeline_key, judge_client, prompt_version, pricing, eligible_ids)


if __name__ == "__main__":
    main()
