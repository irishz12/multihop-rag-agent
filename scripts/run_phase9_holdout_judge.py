#!/usr/bin/env python
"""FINAL HOLDOUT evaluation — LIVE judge scoring for the 44 non-null
holdout-sample answers, `agentic_multi_hop` and `adaptive_rag` only,
checkpointed per (pipeline, qa_id).

Same FROZEN judge as the development-sample evaluation
(`configs/judge.yaml`, unchanged: `openai.gpt-oss-120b`, temperature 0.0,
rubric prompt version "v1") and the same blind input contract —
`mhrag.eval.judge.call_judge`'s signature accepts only (question,
gold_answer, candidate_answer), never a pipeline name/route/model
identifier (see tests/test_eval_judge.py::
test_call_judge_signature_has_no_pipeline_identifying_parameter, which
this script cannot violate even by mistake).

Reads ONLY results/phase9_holdout_{pipeline}_raw.json (must already exist
— this script never generates pipeline answers itself) restricted to
non-null_query records. No CLI flag reaches final_holdout.json directly;
question/gold_answer text comes from the already-completed raw checkpoint,
not a fresh file read.

Usage:
    python scripts/run_phase9_holdout_judge.py --pipeline agentic_multi_hop
    python scripts/run_phase9_holdout_judge.py --pipeline adaptive_rag

Writes results/phase9_holdout_judge_{pipeline}.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.eval.judge import call_judge
from mhrag.generation.mantle_client import MantleClient, MantleConfigError

PIPELINES = ("agentic_multi_hop", "adaptive_rag")


def _build_judge_client() -> tuple[MantleClient, dict]:
    judge_config = load_config("configs/judge.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    try:
        client = MantleClient(
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
        raise SystemExit(f"Cannot run live final holdout judge: {exc}") from exc
    return client, judge_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", required=True, choices=PIPELINES)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    judge_client, judge_config = _build_judge_client()
    prompt_version = judge_config["judge"]["prompt_version"]
    pricing = judge_config["pricing"]

    raw_path = PROJECT_ROOT / "results" / f"phase9_holdout_{args.pipeline}_raw.json"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} does not exist yet — run scripts/run_phase9_holdout_benchmark.py "
                          f"--pipeline {args.pipeline} first")
    raw = json.loads(raw_path.read_text())
    non_null_records = [r for r in raw["records"] if r["question_type"] != "null_query"]
    end = None if args.limit is None else args.offset + args.limit
    slice_records = non_null_records[args.offset : end]
    print(f"Loaded {len(non_null_records)} non-null holdout records for pipeline={args.pipeline}; "
          f"processing [{args.offset}:{end if end is not None else len(non_null_records)}] = {len(slice_records)}")

    out_path = PROJECT_ROOT / "results" / f"phase9_holdout_judge_{args.pipeline}.json"
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = {r["qa_id"]: r for r in json.loads(out_path.read_text()).get("records", [])}
        print(f"Found existing {out_path} with {len(existing)} completed record(s) — will skip those")

    for i, rec in enumerate(slice_records):
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
            "purpose": f"FINAL HOLDOUT evaluation — LLM-as-judge (openai.gpt-oss-120b) scores "
                       f"for pipeline={args.pipeline}",
            "judge_model": judge_config["judge"]["model_id"],
            "pipeline": args.pipeline,
            "split": "final_holdout",
            "n_non_null_questions_total": len(non_null_records),
            "n_questions_completed": len(existing),
            "records": list(existing.values()),
        }
        out_path.write_text(json.dumps(artifact, indent=2))

        if (i + 1) % 20 == 0 or (i + 1) == len(slice_records):
            print(f"  [{i + 1}/{len(slice_records)}] qa_id={rec['qa_id']} grade={result.verdict.grade}")

    print(f"\nTotal accumulated: {len(existing)}/{len(non_null_records)} for pipeline={args.pipeline}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
