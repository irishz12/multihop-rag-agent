#!/usr/bin/env python
"""Phase 9: LIVE fixed LLM-as-judge scoring — `openai.gpt-oss-120b`
(configs/judge.yaml, frozen), checkpointed per (pipeline, qa_id) so an
interrupted run never repeats a paid Mantle call.

Judge input is EXACTLY (question, reference answer, candidate answer) —
never the pipeline name, predicted route, retrieval method, or generation
model (see `mhrag.eval.judge`'s module docstring and
tests/test_eval_judge.py::test_call_judge_signature_has_no_pipeline_identifying_parameter
for the structural guarantee this script cannot violate even by mistake:
`call_judge`'s signature has no parameter to pass any of that through).

Two modes:

  --validate
      Runs the judge on a SMALL, fixed sample BEFORE trusting it on the
      full ~1,325 (pipeline x non-null question) pairs: 26 REAL answer
      pairs already generated in Phase 8B's live smoke comparison
      (results/adaptive_smoke_comparison.json — 13 questions x 2
      pipelines, read-only here) PLUS 4 SYNTHETIC sanity-check pairs with
      obvious expected verdicts (identical text -> "correct"; unrelated
      text -> "incorrect"; the null_query gold "Insufficient information."
      paired with a matching/non-matching decline). Writes
      results/phase9_judge_validation.json for manual inspection — this
      script does NOT decide pass/fail on its own; a human reviews the
      verdicts before the full run proceeds.

  --pipeline {dense,hybrid,hybrid_reranker,agentic_multi_hop,adaptive_rag}
      Scores every NON-null_query record in the corresponding
      results/phase9_{pipeline}_raw.json (must already exist — this script
      never generates pipeline answers itself). null_query is excluded
      from judge scoring by design (it is not an open-ended question to
      grade for correctness — Phase 9's null-query metric is abstention
      correctness, computed separately, offline, from
      mhrag.eval.answer_metrics.is_abstention).

Reads ONLY data/processed/dev_subset.json / results/phase9_{pipeline}_raw
.json / results/adaptive_smoke_comparison.json — all read-only here; this
script never writes to any prior-phase output file except its own
checkpoints (results/phase9_judge_{pipeline}.json,
results/phase9_judge_validation.json). No code path in this script can
reach final_holdout.json (no CLI flag, no config option).

Requires `$OPENAI_API_KEY` in the environment. Never printed/logged/persisted.

Usage:
    python scripts/run_phase9_judge.py --validate
    python scripts/run_phase9_judge.py --pipeline dense --offset 0 --limit 100
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.eval.judge import call_judge
from mhrag.eval.legacy_pipeline_names import to_legacy_name
from mhrag.generation.mantle_client import MantleClient, MantleConfigError

PIPELINES = ("dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag")

_SYNTHETIC_VALIDATION_CASES = [
    {
        "case_id": "synthetic_identical_text",
        "question": "Who is the CEO of Acme Corp?",
        "gold_answer": "Jane Smith is the CEO of Acme Corp.",
        "candidate_answer": "Jane Smith is the CEO of Acme Corp.",
        "expected_grade": "correct",
    },
    {
        "case_id": "synthetic_paraphrase",
        "question": "Who is the CEO of Acme Corp?",
        "gold_answer": "Jane Smith is the CEO of Acme Corp.",
        "candidate_answer": "The chief executive of Acme Corp is Jane Smith.",
        "expected_grade": "correct",
    },
    {
        "case_id": "synthetic_unrelated",
        "question": "Who is the CEO of Acme Corp?",
        "gold_answer": "Jane Smith is the CEO of Acme Corp.",
        "candidate_answer": "The weather in Paris is sunny today.",
        "expected_grade": "incorrect",
    },
    {
        "case_id": "synthetic_null_query_correct_abstention",
        "question": "Which division within Microsoft is both central to the new strategy and top-performing?",
        "gold_answer": "Insufficient information.",
        "candidate_answer": "The available information is insufficient to answer this question.",
        "expected_grade": "correct",
    },
    {
        "case_id": "synthetic_null_query_wrong_specific_answer",
        "question": "Which division within Microsoft is both central to the new strategy and top-performing?",
        "gold_answer": "Insufficient information.",
        "candidate_answer": "The Azure division is both central to the new strategy and top-performing.",
        "expected_grade": "incorrect",
    },
]


def _build_judge_client() -> MantleClient:
    judge_config = load_config("configs/judge.yaml")
    mantle_config = load_config("configs/mantle.yaml")
    try:
        return MantleClient(
            model_id=judge_config["judge"]["model_id"],
            base_url_env=mantle_config["client"]["base_url_env"],
            default_base_url=mantle_config["client"]["default_base_url"],
            api_key_env=mantle_config["client"]["api_key_env"],
            timeout_seconds=mantle_config["client"]["timeout_seconds"],
            temperature=judge_config["judge"]["temperature"],
            max_output_tokens=judge_config["judge"]["max_output_tokens"],
            max_retries=mantle_config["client"]["max_retries"],
            retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
        ), judge_config
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run live Phase 9 judge: {exc}") from exc


def _run_validate() -> None:
    judge_client, judge_config = _build_judge_client()
    prompt_version = judge_config["judge"]["prompt_version"]
    pricing = judge_config["pricing"]

    cases = list(_SYNTHETIC_VALIDATION_CASES)

    # adaptive_smoke_comparison.json is an already-frozen Phase 8B artifact —
    # its per-record keys are still the legacy "adaptive"/"always_agentic"
    # (see mhrag.eval.legacy_pipeline_names); everything this script itself
    # writes below (case_id) uses the canonical name instead.
    smoke_path = PROJECT_ROOT / "results" / "adaptive_smoke_comparison.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text())
        for r in smoke["results"]:
            for canonical_pipeline in ("adaptive_rag", "agentic_multi_hop"):
                cases.append(
                    {
                        "case_id": f"phase8b_{canonical_pipeline}_{r['qa_id']}",
                        "question": r["query"],
                        "gold_answer": None,  # filled below from dev_subset.json
                        "candidate_answer": r[to_legacy_name(canonical_pipeline)]["answer"],
                        "expected_grade": None,
                        "qa_id": r["qa_id"],
                    }
                )

    # Fill in real gold answers for the Phase 8B cases from dev_subset.json.
    dataset_config = load_config("configs/dataset.yaml")
    from mhrag.data.benchmark import qa_id as compute_qa_id
    from mhrag.data.loader import load_qa_records

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / "dev_subset.json"
    dev_by_qa_id = {compute_qa_id(r): r for r in load_qa_records(dev_path)}
    for case in cases:
        if case.get("gold_answer") is None and "qa_id" in case:
            case["gold_answer"] = dev_by_qa_id[case["qa_id"]].answer

    print(f"Running judge validation on {len(cases)} cases "
          f"({len(_SYNTHETIC_VALIDATION_CASES)} synthetic + {len(cases) - len(_SYNTHETIC_VALIDATION_CASES)} "
          f"real Phase 8B answers)")

    results = []
    for i, case in enumerate(cases):
        result = call_judge(
            judge_client, case["question"], case["gold_answer"], case["candidate_answer"],
            prompt_version=prompt_version,
            input_price_per_million=pricing["input_per_million_tokens"],
            output_price_per_million=pricing["output_per_million_tokens"],
        )
        expected = case.get("expected_grade")
        match_marker = ""
        if expected is not None:
            match_marker = " [MATCH]" if result.verdict.grade == expected else " [**MISMATCH**]"
        print(f"[{i + 1}/{len(cases)}] {case['case_id']}: grade={result.verdict.grade} "
              f"score={result.verdict.score} fallback={result.fallback_used}{match_marker}")
        print(f"    reason: {result.verdict.reason[:150]!r}")

        results.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "gold_answer": case["gold_answer"],
                "candidate_answer": case["candidate_answer"],
                "expected_grade": expected,
                "grade": result.verdict.grade,
                "score": result.verdict.score,
                "reason": result.verdict.reason,
                "fallback_used": result.fallback_used,
                "matches_expected": (result.verdict.grade == expected) if expected is not None else None,
                "input_tokens": result.mantle_response.usage.input_tokens,
                "output_tokens": result.mantle_response.usage.output_tokens,
                "latency_ms": result.mantle_response.llm_latency_ms,
            }
        )

    n_synthetic_checked = sum(1 for r in results if r["expected_grade"] is not None)
    n_synthetic_matched = sum(1 for r in results if r["matches_expected"])
    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 9 LLM-as-judge (openai.gpt-oss-120b) small-sample VALIDATION — "
                   "manual review before the full run, not an automated pass/fail gate",
        "judge_model": judge_config["judge"]["model_id"],
        "n_cases": len(cases),
        "n_synthetic_cases_with_expected_grade": n_synthetic_checked,
        "n_synthetic_cases_matched_expected": n_synthetic_matched,
        "results": results,
    }
    out_path = PROJECT_ROOT / "results" / "phase9_judge_validation.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n{'=' * 70}")
    print(f"Synthetic sanity checks: {n_synthetic_matched}/{n_synthetic_checked} matched expected grade")
    print(f"Wrote {out_path}")


def _run_pipeline_scoring(pipeline: str, offset: int, limit: int | None, sample_file: str | None) -> None:
    judge_client, judge_config = _build_judge_client()
    prompt_version = judge_config["judge"]["prompt_version"]
    pricing = judge_config["pricing"]

    raw_path = PROJECT_ROOT / "results" / f"phase9_{pipeline}_raw.json"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} does not exist yet — run scripts/run_phase9_benchmark.py "
                          f"--pipeline {pipeline} first")
    raw = json.loads(raw_path.read_text())
    non_null_records = [r for r in raw["records"] if r["question_type"] != "null_query"]

    if sample_file:
        sample_qa_ids = set(json.loads((PROJECT_ROOT / sample_file).read_text())["qa_ids"])
        non_null_records = [r for r in non_null_records if r["qa_id"] in sample_qa_ids]
        print(f"Restricting to --sample-file {sample_file} — {len(non_null_records)} non-null sample records")

    end = None if limit is None else offset + limit
    slice_records = non_null_records[offset:end]
    print(f"Loaded {len(non_null_records)} non-null records for pipeline={pipeline}; "
          f"processing [{offset}:{end if end is not None else len(non_null_records)}] = {len(slice_records)}")

    out_path = PROJECT_ROOT / "results" / f"phase9_judge_{pipeline}.json"
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
            "purpose": f"Phase 9 LLM-as-judge (openai.gpt-oss-120b) scores for pipeline={pipeline}",
            "judge_model": judge_config["judge"]["model_id"],
            "pipeline": pipeline,
            "n_non_null_questions_total": len(non_null_records),
            "n_questions_completed": len(existing),
            "records": list(existing.values()),
        }
        out_path.write_text(json.dumps(artifact, indent=2))

        if (i + 1) % 20 == 0 or (i + 1) == len(slice_records):
            print(f"  [{i + 1}/{len(slice_records)}] qa_id={rec['qa_id']} grade={result.verdict.grade}")

    print(f"\nTotal accumulated: {len(existing)}/{len(non_null_records)} for pipeline={pipeline}")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--pipeline", choices=PIPELINES)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-file", default=None,
        help="if given, restrict non-null scoring to the qa_ids listed in this "
             "results/phase9_sample.json-shaped file",
    )
    args = parser.parse_args()

    if args.validate:
        _run_validate()
    elif args.pipeline:
        _run_pipeline_scoring(args.pipeline, args.offset, args.limit, args.sample_file)
    else:
        raise SystemExit("pass either --validate or --pipeline <name>")


if __name__ == "__main__":
    main()
