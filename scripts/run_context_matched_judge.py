#!/usr/bin/env python
"""AUDIT ABLATION judge (dev-only, not a Phase 9 script): scores the
"Hybrid+Reranker, context-matched" ablation's answers with the SAME frozen
judge implementation (`mhrag.eval.judge.call_judge`) and the SAME frozen
configuration (`configs/judge.yaml`: `openai.gpt-oss-120b`, temperature 0,
prompt version "v1") as every existing Phase 9 pipeline judge run.

Judge input is EXACTLY (question, gold_answer, candidate_answer) — this
script has no code path that could pass a pipeline-identifying value into
`call_judge`, same structural guarantee `scripts/run_phase9_judge.py`
already relies on (see `call_judge`'s signature in `mhrag.eval.judge`).

null_query is excluded from judge scoring, same rule as every existing
Phase 9 judge run — null-query correctness is abstention accuracy,
computed offline from already-generated answer text
(`mhrag.eval.answer_metrics.is_abstention`), not graded by this judge.

READ-ONLY w.r.t. every existing artifact: only reads
results/phase9_hybrid_reranker_matched_raw.json (this ablation's own raw
output). Writes only its own new output file.

Requires $OPENAI_API_KEY in the environment (never printed/logged/persisted).

Usage:
    python scripts/run_context_matched_judge.py

Writes results/phase9_judge_hybrid_reranker_matched.json, checkpointed per
qa_id (resumable, never repeats a completed paid call).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.eval.judge import call_judge
from mhrag.generation.mantle_client import MantleClient, MantleConfigError

RAW_FILE = "results/phase9_hybrid_reranker_matched_raw.json"  # READ-ONLY — this ablation's own raw output
OUTPUT_FILE = "results/phase9_judge_hybrid_reranker_matched.json"  # this script's ONLY write target


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
        raise SystemExit(f"Cannot run the context-matched ablation judge: {exc}") from exc

    prompt_version = judge_config["judge"]["prompt_version"]
    pricing = judge_config["pricing"]  # deliberately null — cost is never guessed, same rule as every Phase 9 judge run

    raw_path = PROJECT_ROOT / RAW_FILE
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} does not exist yet — run scripts/run_phase9_context_matched_ablation.py first")
    raw = json.loads(raw_path.read_text())
    non_null_records = [r for r in raw["records"] if r["question_type"] != "null_query"]
    print(f"Loaded {len(non_null_records)} non-null record(s) to judge from {RAW_FILE}")

    out_path = PROJECT_ROOT / OUTPUT_FILE
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = {r["qa_id"]: r for r in json.loads(out_path.read_text()).get("records", [])}
        print(f"Found existing {out_path} with {len(existing)} completed record(s) — will skip those")

    for i, rec in enumerate(non_null_records):
        if rec["qa_id"] in existing:
            continue

        result = call_judge(
            judge_client, rec["query"], rec["gold_answer"], rec["answer"],
            prompt_version=prompt_version,
            input_price_per_million=pricing["input_per_million_tokens"],
            output_price_per_million=pricing["output_per_million_tokens"],
        )
        existing[rec["qa_id"]] = {
            "qa_id": rec["qa_id"],
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
            "purpose": "AUDIT ABLATION (dev-only) judge scores — Hybrid+Reranker, context-matched, single pass",
            "judge_model": judge_config["judge"]["model_id"],
            "pipeline": "hybrid_reranker_matched",
            "n_non_null_questions_total": len(non_null_records),
            "n_questions_completed": len(existing),
            "records": list(existing.values()),
        }
        out_path.write_text(json.dumps(artifact, indent=2))

        print(f"  [{i + 1}/{len(non_null_records)}] qa_id={rec['qa_id']} grade={result.verdict.grade}")

    print(f"\nTotal accumulated: {len(existing)}/{len(non_null_records)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
