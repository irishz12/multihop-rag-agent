#!/usr/bin/env python
"""Phase 3 hardening — OLD vs NEW error analysis (dev-only, offline, ZERO
new LLM/API calls).

Reconstructs the PRE-HARDENING (Phase 2) tier-3 verdict-extraction fallback
exactly as it existed before this phase's fix — inline, below, clearly
labeled — and re-applies it to the same already-persisted development
artifacts this project's Task Success work already uses, so the OLD and
NEW classification for every qa_id can be diffed directly. This does NOT
modify `src/mhrag/eval/task_success.py` (already hardened) — it is a
read-only comparison tool only.

Also diffs abstention_status (UNCHANGED — is_abstention() was never
touched) against the NEW response_structure signal, to quantify how many
already-flagged abstention records the richer classifier reclassifies.

Writes ONLY results/task_success_hardening_error_analysis.json — never
modifies results/task_success_report.json or
results/task_success_report_v2.json.

Usage:
    python scripts/task_success_hardening_error_analysis.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id as compute_qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.task_success import (
    CANONICAL_VERDICTS,
    _resolve_negation,  # reused unchanged — negation resolution itself was never the bug
    _strip_meta_or_phrases,  # reused unchanged — meta-or stripping was never the bug
    classify_response_structure,
    extract_verdict,
    normalize_gold_verdict,
)

BASELINE_RAW_FILE = "results/phase9_hybrid_reranker_raw.json"
AGENTIC_RAW_FILE = "results/phase9_always_agentic_raw.json"
MATCHED_RAW_FILE = "results/phase9_hybrid_reranker_matched_full_raw.json"
DEV_SPLIT_FILE = "dev_subset.json"
OUTPUT_FILE = "results/task_success_hardening_error_analysis.json"  # this script's ONLY write target

_VERDICT_ALTERNATION = "|".join(CANONICAL_VERDICTS)
_OLD_LEADING_RE = re.compile(rf"^\s*(not\s+)?\b({_VERDICT_ALTERNATION})\b", re.IGNORECASE)
_OLD_EXPLICIT_RE = re.compile(
    rf"\b(?:the answer is|it is|overall,?)\s+(not\s+)?\b({_VERDICT_ALTERNATION})\b", re.IGNORECASE
)
_OLD_CLAUSE_SPLIT_RE = re.compile(r"[.;,!?]")
_OLD_NEGATORS = ("not", "n't", "never", "cannot", "can't", "doesn't", "does not", "did not", "didn't")


@dataclass(frozen=True, slots=True)
class OldVerdictExtraction:
    verdict: str | None
    is_ambiguous: bool


def _old_scan_priority_patterns(text: str) -> list[tuple[int, str]]:
    """UNCHANGED between Phase 2 and Phase 3 — reproduced here only so
    this script is self-contained, not because tier 1/2 differ."""
    matches: list[tuple[int, str]] = []
    leading = _OLD_LEADING_RE.match(text)
    if leading:
        resolved = _resolve_negation(leading.group(2).lower(), leading.group(1) is not None)
        if resolved is not None:
            matches.append((leading.start(), resolved))
    for m in _OLD_EXPLICIT_RE.finditer(text):
        resolved = _resolve_negation(m.group(2).lower(), m.group(1) is not None)
        if resolved is not None:
            matches.append((m.start(), resolved))
    matches.sort(key=lambda pair: pair[0])
    return matches


def _old_scan_clause_fallback(text: str) -> list[tuple[int, str]]:
    """PRE-HARDENING (Phase 2) behavior, reconstructed exactly: accepts a
    canonical word ANYWHERE in its clause — no clause-final restriction.
    This is the version of the code that misread "there is no praise for
    the Biden administration" as a "no" verdict. Kept here, clearly
    labeled, ONLY for this diff — the real implementation in
    src/mhrag/eval/task_success.py is already hardened."""
    matches: list[tuple[int, str]] = []
    offset = 0
    for clause in _OLD_CLAUSE_SPLIT_RE.split(text):
        clause_lower = clause.lower()
        for word_match in re.finditer(rf"\b({_VERDICT_ALTERNATION})\b", clause_lower):
            preceding = clause_lower[: word_match.start()]
            negated = any(re.search(rf"\b{re.escape(neg)}\b", preceding) for neg in _OLD_NEGATORS)
            resolved = _resolve_negation(word_match.group(1), negated)
            if resolved is not None:
                matches.append((offset + word_match.start(), resolved))
        offset += len(clause) + 1
    matches.sort(key=lambda pair: pair[0])
    return matches


def old_extract_verdict(text: str) -> OldVerdictExtraction:
    cleaned = _strip_meta_or_phrases(text)
    priority_matches = _old_scan_priority_patterns(cleaned)
    if priority_matches:
        return OldVerdictExtraction(verdict=priority_matches[-1][1], is_ambiguous=False)
    fallback_matches = _old_scan_clause_fallback(cleaned)
    if fallback_matches:
        return OldVerdictExtraction(verdict=fallback_matches[-1][1], is_ambiguous=False)
    return OldVerdictExtraction(verdict=None, is_ambiguous=True)


def old_verdict_correctness(gold_answer: str, generated_answer: str) -> str:
    gold_verdict = normalize_gold_verdict(gold_answer)
    if gold_verdict is None:
        return "not_applicable"
    extraction = old_extract_verdict(generated_answer)
    if extraction.is_ambiguous:
        return "ambiguous"
    return "correct" if extraction.verdict == gold_verdict else "incorrect"


def new_verdict_correctness(gold_answer: str, generated_answer: str) -> str:
    gold_verdict = normalize_gold_verdict(gold_answer)
    if gold_verdict is None:
        return "not_applicable"
    extraction = extract_verdict(generated_answer)
    if extraction.is_ambiguous:
        return "ambiguous"
    return "correct" if extraction.verdict == gold_verdict else "incorrect"


def _load(path: str) -> dict | None:
    p = PROJECT_ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def _gold_answers_by_qa_id() -> dict[str, str]:
    dataset_config = load_config("configs/dataset.yaml")
    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    return {compute_qa_id(r): r.answer for r in load_qa_records(dev_path)}


# --- hand-built regression set (mirrors tests/test_task_success_hardening.py exactly) -----

VERDICT_REGRESSION_SET = [
    # (text, gold, expected_new_correctness_label_or_None_for_just_verdict_check, description)
    ("No, the sources disagree.", "no", "no", "leading tier"),
    ("Yes, they agree.", "yes", "yes", "leading tier"),
    ("The answer is no.", "no", "no", "explicit tier"),
    ("Overall, yes.", "yes", "yes", "explicit tier (overall)"),
    ("There is no evidence that...", None, None, "determiner 'no' — must be ambiguous"),
    ("There is no agreement between the sources.", None, None, "determiner 'no' — must be ambiguous"),
    (
        "I don't think the answer is no; it is yes.",
        "yes",
        "yes",
        "negation trap, still resolved by explicit tier",
    ),
    ("Not no, but yes.", "yes", "yes", "double negative via leading tier"),
    ("It is unclear.", None, None, "no canonical word at all — ambiguous"),
    (
        "I cannot determine whether the answer is yes or no.",
        None,
        None,
        "'X or Y' meta-reference — ambiguous",
    ),
]

ABSTENTION_REGRESSION_SET = [
    ("Insufficient information to answer.", "clean_abstention"),
    ("Insufficient information to answer; however, the answer is Google.", "answer_with_uncertainty"),
    ("I cannot be certain, but the answer is Google.", "answer_with_uncertainty"),
    ("The available evidence is limited, but the sources indicate Google.", "answer_with_uncertainty"),
    ("I don't have enough information to determine this.", "clean_abstention"),
    ("The answer is Google.", "substantive_answer"),
    ("Google, although the evidence is limited.", "answer_with_uncertainty"),
]


def _run_verdict_regression() -> dict:
    results = []
    n_correct = 0
    for text, gold, expected_verdict, description in VERDICT_REGRESSION_SET:
        extraction = extract_verdict(text)
        actual = extraction.verdict
        passed = actual == expected_verdict
        n_correct += int(passed)
        results.append(
            {"text": text, "expected_verdict": expected_verdict, "actual_verdict": actual,
             "is_ambiguous": extraction.is_ambiguous, "description": description, "passed": passed}
        )
    return {
        "n_cases": len(VERDICT_REGRESSION_SET), "n_passed": n_correct,
        "false_positive_rate": 1 - (n_correct / len(VERDICT_REGRESSION_SET)),
        "cases": results,
    }


def _run_abstention_regression() -> dict:
    results = []
    n_correct = 0
    for text, expected in ABSTENTION_REGRESSION_SET:
        actual = classify_response_structure(text)
        passed = actual == expected
        n_correct += int(passed)
        results.append({"text": text, "expected": expected, "actual": actual, "passed": passed})
    return {
        "n_cases": len(ABSTENTION_REGRESSION_SET), "n_passed": n_correct,
        "false_positive_rate": 1 - (n_correct / len(ABSTENTION_REGRESSION_SET)),
        "cases": results,
    }


def _diff_population(name: str, raw_doc: dict, gold_answers: dict[str, str]) -> dict:
    records = raw_doc["records"]
    changed = []
    n_verdict_applicable = 0
    n_old_ambiguous = n_new_ambiguous = 0
    for rec in records:
        if rec["question_type"] not in ("comparison_query", "temporal_query"):
            continue
        gold = gold_answers[rec["qa_id"]]
        if normalize_gold_verdict(gold) is None:
            continue  # gold outside canonical vocabulary — not_applicable either way, not part of this diff
        n_verdict_applicable += 1
        old = old_verdict_correctness(gold, rec["answer"])
        new = new_verdict_correctness(gold, rec["answer"])
        if old == "ambiguous":
            n_old_ambiguous += 1
        if new == "ambiguous":
            n_new_ambiguous += 1
        if old != new:
            changed.append(
                {"qa_id": rec["qa_id"], "question_type": rec["question_type"], "gold_answer": gold,
                 "answer_snippet": rec["answer"][:200], "old_correctness": old, "new_correctness": new}
            )
    return {
        "pipeline": name, "n_verdict_applicable": n_verdict_applicable,
        "n_changed": len(changed), "n_old_ambiguous": n_old_ambiguous, "n_new_ambiguous": n_new_ambiguous,
        "changed_records": changed,
    }


def main() -> None:
    gold_answers = _gold_answers_by_qa_id()

    baseline_raw = _load(BASELINE_RAW_FILE)
    agentic_raw = _load(AGENTIC_RAW_FILE)
    matched_raw = _load(MATCHED_RAW_FILE)

    verdict_regression = _run_verdict_regression()
    abstention_regression = _run_abstention_regression()

    diffs = [
        _diff_population("hybrid_reranker_5chunk", baseline_raw, gold_answers),
        _diff_population("agentic_multi_hop", agentic_raw, gold_answers),
        _diff_population("hybrid_reranker_context_matched", matched_raw, gold_answers),
    ]

    total_changed = sum(d["n_changed"] for d in diffs)
    total_applicable = sum(d["n_verdict_applicable"] for d in diffs)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 3 hardening — OLD (pre-hardening) vs NEW (hardened) verdict-extraction diff, "
                   "plus hand-built regression results, dev-only, offline, zero new LLM calls",
        "verdict_extraction_regression_set": verdict_regression,
        "abstention_structure_regression_set": abstention_regression,
        "population_diffs": diffs,
        "summary": {
            "total_comparison_or_temporal_records_scanned": total_applicable,
            "total_records_whose_deterministic_correctness_changed": total_changed,
            "fraction_changed": (total_changed / total_applicable) if total_applicable else None,
        },
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nVerdict regression: {verdict_regression['n_passed']}/{verdict_regression['n_cases']} passed "
          f"(false-positive rate: {verdict_regression['false_positive_rate']:.1%})")
    print(f"Abstention-structure regression: {abstention_regression['n_passed']}/{abstention_regression['n_cases']} passed "
          f"(false-positive rate: {abstention_regression['false_positive_rate']:.1%})")
    for d in diffs:
        print(f"\n{d['pipeline']}: {d['n_changed']}/{d['n_verdict_applicable']} changed, "
              f"old_ambiguous={d['n_old_ambiguous']}, new_ambiguous={d['n_new_ambiguous']}")
        for c in d["changed_records"][:5]:
            print(f"  qa_id={c['qa_id']} gold={c['gold_answer']!r} old={c['old_correctness']} new={c['new_correctness']}")
            print(f"    answer: {c['answer_snippet']!r}")


if __name__ == "__main__":
    main()
