"""Unit tests for mhrag.eval.task_success — offline, no model call, no I/O.

Covers: abstention 2x2 (all four cells), verdict extraction (leading /
explicit / negation trap / hedge-ambiguous), entity containment
(match + documented paraphrase false-negative), the unsupported flag, and
deterministic repeatability.
"""

from __future__ import annotations

from mhrag.eval.task_success import (
    ABSTENTION_STATUSES,
    DETERMINISTIC_MATCH_TYPES,
    classify_abstention,
    classify_task_success,
    compute_unsupported,
    entity_containment,
    extract_verdict,
    normalize_gold_verdict,
    verdict_match,
)

# --- abstention: all four 2x2 cells ------------------------------------------------------


def test_correct_null_abstention():
    result = classify_abstention("null_query", "The available information is insufficient to answer this question.")
    assert result.is_abstention is True
    assert result.status == "correct_abstention"


def test_incorrect_null_abstention_ie_hallucination():
    """null_query, but the system stated a specific answer instead of abstaining."""
    result = classify_abstention("null_query", "The division most central to the strategy is Azure.")
    assert result.is_abstention is False
    assert result.status == "hallucinated_non_abstention"


def test_incorrect_non_null_abstention():
    """A perfectly answerable question, but the system wrongly declined."""
    result = classify_abstention("comparison_query", "There is not enough information to determine this.")
    assert result.is_abstention is True
    assert result.status == "incorrect_abstention"


def test_normal_non_abstention():
    result = classify_abstention("inference_query", "The individual is Sam Bankman-Fried.")
    assert result.is_abstention is False
    assert result.status == "normal_non_abstention"


def test_all_four_abstention_statuses_are_distinct_and_documented():
    assert len(set(ABSTENTION_STATUSES)) == 4


# --- verdict extraction: comparison_query / temporal_query ------------------------------


def test_correct_comparison_verdict_leading_pattern():
    result = verdict_match("Yes", "Yes, the sources agree on this point.")
    assert result.correctness == "correct"
    assert result.gold_verdict == "yes"
    assert result.extracted_verdict == "yes"
    assert result.is_ambiguous is False


def test_incorrect_comparison_verdict():
    result = verdict_match("Yes", "No, they do not agree.")
    assert result.correctness == "incorrect"
    assert result.gold_verdict == "yes"
    assert result.extracted_verdict == "no"


def test_explicit_the_answer_is_pattern():
    result = verdict_match("yes", "The answer is yes.")
    assert result.correctness == "correct"
    assert result.extracted_verdict == "yes"


def test_hedge_resolved_via_overall_marker():
    """'It is unclear, but overall yes.' must resolve to yes, not ambiguous —
    the explicit 'overall' marker is a high-confidence commitment even
    though the sentence opens with hedge language."""
    result = verdict_match("yes", "It is unclear, but overall yes.")
    assert result.correctness == "correct"
    assert result.extracted_verdict == "yes"
    assert result.is_ambiguous is False


def test_negation_trap_does_not_misclassify_as_no():
    """The single most important extraction test: 'not no' must not be
    read as a negative verdict, and the sentence's final, explicit,
    unnegated clause ('it is yes') must win."""
    extraction = extract_verdict("The answer is not no; it is yes.")
    assert extraction.verdict == "yes"
    assert extraction.is_ambiguous is False

    result = verdict_match("yes", "The answer is not no; it is yes.")
    assert result.correctness == "correct"
    assert result.extracted_verdict == "yes"


def test_negation_flips_a_single_unambiguous_mention():
    """A single negated mention with no later clarifying clause must still
    flip correctly (not just rely on 'last mention wins')."""
    extraction = extract_verdict("The answer is not yes.")
    assert extraction.verdict == "no"
    assert extraction.is_ambiguous is False


def test_hedged_ambiguous_verdict_returns_explicit_ambiguous_state():
    """A genuine hedge that never commits to either option (here, an
    'X or Y' meta-reference to the question's own format) must return
    ambiguous, never a forced guess."""
    extraction = extract_verdict("It's hard to give a simple yes or no on this one.")
    assert extraction.is_ambiguous is True
    assert extraction.verdict is None

    result = verdict_match("yes", "It's hard to give a simple yes or no on this one.")
    assert result.correctness == "ambiguous"
    assert result.gold_verdict == "yes"
    assert result.extracted_verdict is None


def test_genuinely_uncommitted_text_is_ambiguous():
    extraction = extract_verdict("It's really difficult to determine either way; both interpretations seem plausible.")
    assert extraction.is_ambiguous is True
    assert extraction.verdict is None


def test_gold_value_outside_canonical_vocabulary_is_not_applicable():
    """A real, observed dev-split gold value ('different aspects') that
    isn't in the recognized single-token vocabulary — must be
    not_applicable, never forced into the closest-looking token."""
    assert normalize_gold_verdict("different aspects") is None
    result = verdict_match("different aspects", "Yes, they align on this.")
    assert result.correctness == "not_applicable"
    assert result.gold_verdict is None


# --- entity containment: inference_query, secondary signal only -------------------------


def test_correct_inference_entity_containment():
    matched = entity_containment("Google", "The article discusses Google and its recent policy changes.")
    assert matched is True


def test_entity_containment_paraphrase_is_a_documented_false_negative():
    """The known limitation, tested explicitly: a correct paraphrase that
    never states the gold entity verbatim registers as no-match. This
    does NOT mean the answer is wrong — see the function's own docstring
    ('THIS DOES NOT PROVE SEMANTIC CORRECTNESS')."""
    matched = entity_containment("Google", "The tech giant behind the search engine made the announcement.")
    assert matched is False


def test_inference_query_never_sets_deterministic_correctness_from_containment():
    """CRITICAL METHODOLOGY CHECK: even when entity_containment matches,
    deterministic_correctness for inference_query must stay
    'not_applicable' — containment is a secondary signal only, never
    promoted to a correctness claim."""
    result = classify_task_success(
        question_type="inference_query",
        gold_answer="Google",
        generated_answer="The article discusses Google and its recent policy changes.",
    )
    assert result.entity_containment_match is True
    assert result.deterministic_correctness == "not_applicable"
    assert result.deterministic_match_type == "entity_containment_secondary"


# --- unsupported flag: judge grade x evidence coverage ONLY -----------------------------


def test_unsupported_when_judge_correct_but_zero_evidence_coverage():
    assert compute_unsupported("correct", 0.0) is True


def test_not_unsupported_when_evidence_present():
    assert compute_unsupported("correct", 0.6) is False


def test_not_unsupported_when_judge_says_incorrect_regardless_of_coverage():
    """An ungrounded WRONG answer isn't flagged 'unsupported' — being
    ungrounded only matters when the answer otherwise looks right."""
    assert compute_unsupported("incorrect", 0.0) is False


def test_unsupported_is_none_when_inputs_unavailable():
    assert compute_unsupported(None, 0.5) is None
    assert compute_unsupported("correct", None) is None


def test_unsupported_flag_is_independent_of_deterministic_verdict():
    """The unsupported flag must be computable purely from judge_grade +
    evidence_coverage, unaffected by what verdict_match/entity_containment
    found — even when the deterministic check disagrees with the judge."""
    result = classify_task_success(
        question_type="comparison_query",
        gold_answer="yes",
        generated_answer="No, they do not agree.",  # deterministic says incorrect
        judge_grade="correct",  # judge disagrees
        judge_score=1.0,
        evidence_coverage=0.0,
    )
    assert result.deterministic_correctness == "incorrect"
    assert result.judge_grade == "correct"
    assert result.judge_deterministic_agree is False  # disagreement surfaced, not hidden
    assert result.unsupported is True  # judge=correct, coverage=0 -> still computed independently


# --- normal correct / incorrect end-to-end cases -----------------------------------------


def test_normal_correct_case_end_to_end():
    result = classify_task_success(
        question_type="temporal_query",
        gold_answer="no",
        generated_answer="No, the timelines are not consistent.",
        judge_grade="correct",
        judge_score=1.0,
        evidence_coverage=1.0,
    )
    assert result.abstention_status == "normal_non_abstention"
    assert result.deterministic_correctness == "correct"
    assert result.judge_deterministic_agree is True
    assert result.unsupported is False
    assert result.task_success_confident is True


def test_normal_incorrect_case_end_to_end():
    result = classify_task_success(
        question_type="temporal_query",
        gold_answer="no",
        generated_answer="Yes, the timelines are consistent.",
        judge_grade="incorrect",
        judge_score=0.0,
        evidence_coverage=1.0,
    )
    assert result.deterministic_correctness == "incorrect"
    assert result.judge_deterministic_agree is True
    assert result.task_success_confident is True


def test_inference_query_never_confident():
    """Per the explicit methodology rule (entity containment never proves
    correctness), inference_query records must never claim
    task_success_confident=True."""
    result = classify_task_success(
        question_type="inference_query",
        gold_answer="Google",
        generated_answer="The answer is Google.",
        judge_grade="correct",
        judge_score=1.0,
        evidence_coverage=1.0,
    )
    assert result.task_success_confident is False


def test_ambiguous_verdict_is_not_confident():
    result = classify_task_success(
        question_type="comparison_query",
        gold_answer="yes",
        generated_answer="It's really difficult to determine either way.",
        judge_grade="incorrect",
        judge_score=0.0,
        evidence_coverage=0.5,
    )
    assert result.deterministic_correctness == "ambiguous"
    assert result.task_success_confident is False
    assert result.judge_deterministic_agree is None  # not comparable when deterministic is ambiguous


def test_abstention_case_is_always_confident_regardless_of_judge():
    result = classify_task_success(
        question_type="null_query",
        gold_answer="Insufficient information.",
        generated_answer="The information provided is insufficient to answer this question.",
    )
    assert result.abstention_status == "correct_abstention"
    assert result.task_success_confident is True
    assert result.deterministic_match_type == "abstention_only"


# --- determinism ---------------------------------------------------------------------------


def test_deterministic_repeatability():
    kwargs = dict(
        question_type="comparison_query",
        gold_answer="yes",
        generated_answer="The answer is not no; it is yes.",
        judge_grade="correct",
        judge_score=1.0,
        evidence_coverage=0.5,
    )
    first = classify_task_success(**kwargs)
    second = classify_task_success(**kwargs)
    assert first == second


def test_extract_verdict_is_deterministic_across_repeated_calls():
    text = "It is unclear, but overall yes."
    results = [extract_verdict(text) for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_deterministic_match_types_are_exhaustive_and_distinct():
    assert len(set(DETERMINISTIC_MATCH_TYPES)) == 4
