#!/usr/bin/env python
"""Phase 4 — real-data validation (dev-only, offline, ZERO new LLM/API calls).

Two independent analyses:

  1. VERDICT EXTRACTION: reconstructs Phase 3's exact fallback logic
     (comma-inclusive clause splitting, no uncertainty-blocker check)
     inline, below, clearly labeled as a reference implementation only —
     src/mhrag/eval/task_success.py is not touched by this comparison —
     and diffs it against the current (Phase 4) hardened behavior across
     every comparison_query/temporal_query record in the three
     already-persisted development pipelines this project's Task Success
     work already uses.

  2. RESPONSE STRUCTURE (EXPERIMENTAL): draws a fixed, reproducible
     stratified random sample (seed 2029, same seed this project already
     uses for its evaluation sampling) of records flagged by
     abstention_status, prints full text + old (Phase 3 residue-word) vs
     new (Phase 4 assertion-marker) classification side by side, for
     MANUAL review — this script does not and cannot auto-grade semantic
     correctness; a human must read the printed sample and judge it.

Writes ONLY results/task_success_hardening_error_analysis_v2.json — never
modifies results/task_success_report.json, results/task_success_report_v2.json,
or results/task_success_hardening_error_analysis.json (Phase 3's).

Usage:
    python scripts/task_success_phase4_error_analysis.py
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.task_success import (
    CANONICAL_VERDICTS,
    DECLINE_PHRASES,
    HEDGE_PHRASES,
    _resolve_negation,
    _strip_meta_or_phrases,
    classify_response_structure,
    extract_verdict,
    normalize_gold_verdict,
)

BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"
DEV_SPLIT_FILE = "dev_subset.json"
OUTPUT_FILE = "results/task_success_hardening_error_analysis_v2.json"  # this script's ONLY write target

SAMPLE_SEED = 2029
RESPONSE_STRUCTURE_SAMPLE_SIZE = 27  # 9 per pipeline, stratified

_VERDICT_ALTERNATION = "|".join(CANONICAL_VERDICTS)

# --- PHASE 3 reference implementation (comma-inclusive clause split, no uncertainty check) --
# Reproduced here ONLY for this diff — NOT the current implementation (which is hardened
# further in src/mhrag/eval/task_success.py). See scripts/task_success_hardening_error_analysis.py
# for the earlier Phase2->Phase3 diff this one continues from.

_P3_LEADING_RE = re.compile(rf"^\s*(not\s+)?\b({_VERDICT_ALTERNATION})\b", re.IGNORECASE)
_P3_EXPLICIT_RE = re.compile(
    rf"\b(?:the answer is|it is|overall,?)\s+(not\s+)?\b({_VERDICT_ALTERNATION})\b", re.IGNORECASE
)
_P3_CLAUSE_SPLIT_RE = re.compile(r"[.;,!?]")  # Phase 3: comma-INCLUSIVE (the bug)
_P3_NEGATORS = ("not", "n't", "never", "cannot", "can't", "doesn't", "does not", "did not", "didn't")


def _p3_scan_priority(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    leading = _P3_LEADING_RE.match(text)
    if leading:
        resolved = _resolve_negation(leading.group(2).lower(), leading.group(1) is not None)
        if resolved is not None:
            matches.append((leading.start(), resolved))
    for m in _P3_EXPLICIT_RE.finditer(text):
        resolved = _resolve_negation(m.group(2).lower(), m.group(1) is not None)
        if resolved is not None:
            matches.append((m.start(), resolved))
    matches.sort(key=lambda p: p[0])
    return matches


def _p3_scan_clause_fallback(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    offset = 0
    for clause in _P3_CLAUSE_SPLIT_RE.split(text):
        clause_lower = clause.lower()
        for word_match in re.finditer(rf"\b({_VERDICT_ALTERNATION})\b", clause_lower):
            if clause_lower[word_match.end() :].strip():
                continue
            preceding = clause_lower[: word_match.start()]
            negated = any(re.search(rf"\b{re.escape(neg)}\b", preceding) for neg in _P3_NEGATORS)
            resolved = _resolve_negation(word_match.group(1), negated)
            if resolved is not None:
                matches.append((offset + word_match.start(), resolved))
        offset += len(clause) + 1
    matches.sort(key=lambda p: p[0])
    return matches


def p3_verdict_correctness(gold_answer: str, generated_answer: str) -> str:
    gold_verdict = normalize_gold_verdict(gold_answer)
    if gold_verdict is None:
        return "not_applicable"
    cleaned = _strip_meta_or_phrases(generated_answer)
    priority = _p3_scan_priority(cleaned)
    if priority:
        verdict = priority[-1][1]
    else:
        fallback = _p3_scan_clause_fallback(cleaned)
        if not fallback:
            return "ambiguous"
        verdict = fallback[-1][1]
    return "correct" if verdict == gold_verdict else "incorrect"


def p4_verdict_correctness(gold_answer: str, generated_answer: str) -> str:
    gold_verdict = normalize_gold_verdict(gold_answer)
    if gold_verdict is None:
        return "not_applicable"
    extraction = extract_verdict(generated_answer)
    if extraction.is_ambiguous:
        return "ambiguous"
    return "correct" if extraction.verdict == gold_verdict else "incorrect"


# --- Phase 3 response_structure reference (residue-word count) — for the printed comparison only ---

_P3_GENERIC_RESIDUE_WORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "on", "in", "for", "as",
        "answer", "question", "this", "that", "it", "information", "context", "source", "sources",
        "based", "given", "determine", "determined", "know", "certain", "sure",
        "i", "we", "they", "you",
    }
)
_P3_CONTRASTIVE_CONJUNCTIONS = ("however", "but", "although", "though", "yet")
_P3_WHITESPACE_RE = re.compile(r"\s+")


def _p3_strip_phrases(text_lower: str, phrases: tuple[str, ...]) -> str:
    for phrase in sorted(phrases, key=len, reverse=True):
        text_lower = text_lower.replace(phrase, " ")
    return text_lower


def p3_response_structure(generated_answer: str) -> str:
    text = generated_answer.strip()
    if not text:
        return "ambiguous"
    lowered = _P3_WHITESPACE_RE.sub(" ", text.lower())
    has_decline = any(p in lowered for p in DECLINE_PHRASES)
    has_hedge = any(p in lowered for p in HEDGE_PHRASES)
    if not has_decline and not has_hedge:
        return "substantive_answer"
    stripped = _p3_strip_phrases(lowered, DECLINE_PHRASES + HEDGE_PHRASES)
    words = re.findall(r"[a-z0-9']+", stripped)
    residue = [w for w in words if w not in _P3_GENERIC_RESIDUE_WORDS and w not in _P3_CONTRASTIVE_CONJUNCTIONS]
    if residue:
        return "answer_with_uncertainty"
    if has_decline:
        return "clean_abstention"
    return "ambiguous"


def _load(path: str) -> dict | None:
    p = PROJECT_ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def _gold_answers_by_qa_id() -> dict[str, str]:
    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    return {compute_qa_id(r): r.answer for r in load_qa_records(dev_path)}


def _verdict_diff(name: str, raw_doc: dict, gold_answers: dict[str, str]) -> dict:
    changed = []
    n_applicable = n_p3_ambiguous = n_p4_ambiguous = 0
    for rec in raw_doc["records"]:
        if rec["question_type"] not in ("comparison_query", "temporal_query"):
            continue
        gold = gold_answers[rec["qa_id"]]
        if normalize_gold_verdict(gold) is None:
            continue
        n_applicable += 1
        old = p3_verdict_correctness(gold, rec["answer"])
        new = p4_verdict_correctness(gold, rec["answer"])
        n_p3_ambiguous += old == "ambiguous"
        n_p4_ambiguous += new == "ambiguous"
        if old != new:
            changed.append(
                {"qa_id": rec["qa_id"], "gold_answer": gold, "answer_snippet": rec["answer"][:200],
                 "phase3_correctness": old, "phase4_correctness": new}
            )
    return {
        "pipeline": name, "n_applicable": n_applicable, "n_changed": len(changed),
        "n_phase3_ambiguous": n_p3_ambiguous, "n_phase4_ambiguous": n_p4_ambiguous,
        "changed_records": changed,
    }


def _sample_response_structure_records(pipelines: dict[str, dict]) -> list[dict]:
    """Stratified random sample (fixed seed) of records flagged by
    DECLINE_PHRASES/HEDGE_PHRASES presence, across all three pipelines —
    for manual review, printed below. Deterministic given the same input
    files (same seed, same selection order)."""
    per_pipeline = RESPONSE_STRUCTURE_SAMPLE_SIZE // len(pipelines)
    sample: list[dict] = []
    for name, raw_doc in pipelines.items():
        flagged = []
        for rec in raw_doc["records"]:
            lowered = rec["answer"].lower()
            if any(p in lowered for p in DECLINE_PHRASES) or any(p in lowered for p in HEDGE_PHRASES):
                flagged.append(rec)
        rng = random.Random(SAMPLE_SEED)
        rng.shuffle(flagged)
        for rec in flagged[:per_pipeline]:
            sample.append(
                {
                    "pipeline": name, "qa_id": rec["qa_id"], "question_type": rec["question_type"],
                    "answer": rec["answer"],
                    "phase3_response_structure": p3_response_structure(rec["answer"]),
                    "phase4_response_structure": classify_response_structure(rec["answer"]),
                }
            )
    return sample


def main() -> None:
    gold_answers = _gold_answers_by_qa_id()
    pipelines = {
        "hybrid_reranker_5chunk": _load(BASELINE_RAW_FILE),
        "agentic_multi_hop": _load(AGENTIC_RAW_FILE),
        "hybrid_reranker_context_matched": _load(MATCHED_RAW_FILE),
    }

    verdict_diffs = [_verdict_diff(name, doc, gold_answers) for name, doc in pipelines.items()]
    total_applicable = sum(d["n_applicable"] for d in verdict_diffs)
    total_changed = sum(d["n_changed"] for d in verdict_diffs)

    response_structure_sample = _sample_response_structure_records(pipelines)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 4 real-data validation: Phase3->Phase4 verdict-extraction diff + a stratified "
                   "random sample of response_structure classifications for MANUAL review "
                   "(this script does not and cannot auto-grade semantic correctness)",
        "verdict_extraction": {
            "population_diffs": verdict_diffs,
            "summary": {
                "total_comparison_or_temporal_records_scanned": total_applicable,
                "total_records_whose_correctness_changed_phase3_to_phase4": total_changed,
                "fraction_changed": (total_changed / total_applicable) if total_applicable else None,
            },
        },
        "response_structure_manual_review_sample": {
            "sample_size": len(response_structure_sample),
            "sample_seed": SAMPLE_SEED,
            "note": "EXPERIMENTAL signal — this sample is for human inspection, printed to stdout; "
                    "the JSON below records what was shown, not a verdict on correctness.",
            "records": response_structure_sample,
        },
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}\n")

    print("=" * 100)
    print("VERDICT EXTRACTION: Phase 3 -> Phase 4 diff")
    print("=" * 100)
    print(f"Total comparison/temporal records scanned: {total_applicable}")
    print(f"Total changed: {total_changed} ({total_changed/total_applicable:.1%})")
    for d in verdict_diffs:
        print(f"\n{d['pipeline']}: n_applicable={d['n_applicable']} n_changed={d['n_changed']} "
              f"phase3_ambiguous={d['n_phase3_ambiguous']} phase4_ambiguous={d['n_phase4_ambiguous']}")

    print("\n" + "=" * 100)
    print(f"RESPONSE STRUCTURE (EXPERIMENTAL) — manual review sample, n={len(response_structure_sample)}, seed={SAMPLE_SEED}")
    print("=" * 100)
    for i, rec in enumerate(response_structure_sample, 1):
        print(f"\n[{i}/{len(response_structure_sample)}] pipeline={rec['pipeline']} qa_id={rec['qa_id']} "
              f"type={rec['question_type']}")
        print(f"  phase3={rec['phase3_response_structure']:22} phase4={rec['phase4_response_structure']}")
        print(f"  answer: {rec['answer'][:280]!r}")


if __name__ == "__main__":
    main()
