"""Unit tests for mhrag.eval.multihop_success — offline, no model call, no I/O."""

from __future__ import annotations

from mhrag.eval.multihop_success import (
    ExampleCandidate,
    QuestionOutcome,
    added_required_evidence,
    classify_tier,
    coverage,
    select_examples,
)


def _outcome(**overrides) -> QuestionOutcome:
    defaults = dict(
        qa_id="q1",
        question_type="comparison_query",
        stop_reason="max_hops",
        num_agent_hops=2,
        gold_doc_ids=frozenset({"a", "b"}),
        hop1_doc_ids=frozenset({"a"}),
        final_doc_ids=frozenset({"a", "b"}),
        baseline_doc_ids=frozenset({"a"}),
        matched_doc_ids=frozenset({"a"}),
        agentic_grade="correct",
        baseline_grade="incorrect",
        matched_grade="incorrect",
    )
    defaults.update(overrides)
    return QuestionOutcome(**defaults)


# --- coverage --------------------------------------------------------------------------------


def test_coverage_full():
    assert coverage(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0


def test_coverage_partial():
    assert coverage(frozenset({"a"}), frozenset({"a", "b"})) == 0.5


def test_coverage_none_when_gold_empty():
    assert coverage(frozenset({"a"}), frozenset()) is None


def test_coverage_none_when_doc_ids_missing():
    assert coverage(None, frozenset({"a"})) is None


def test_coverage_zero_when_no_overlap():
    assert coverage(frozenset({"z"}), frozenset({"a", "b"})) == 0.0


# --- added_required_evidence -------------------------------------------------------------------


def test_added_required_evidence_detects_new_gold_doc():
    outcome = _outcome(hop1_doc_ids=frozenset({"a"}), final_doc_ids=frozenset({"a", "b"}),
                        gold_doc_ids=frozenset({"a", "b"}))
    assert added_required_evidence(outcome) == frozenset({"b"})


def test_added_required_evidence_empty_when_hop1_already_had_everything():
    outcome = _outcome(hop1_doc_ids=frozenset({"a", "b"}), final_doc_ids=frozenset({"a", "b"}),
                        gold_doc_ids=frozenset({"a", "b"}))
    assert added_required_evidence(outcome) == frozenset()


def test_added_required_evidence_ignores_non_gold_additions():
    """A later hop can add a document that isn't gold-required — that must
    not count as 'added required evidence'."""
    outcome = _outcome(hop1_doc_ids=frozenset({"a"}), final_doc_ids=frozenset({"a", "z"}),
                        gold_doc_ids=frozenset({"a", "b"}))
    assert added_required_evidence(outcome) == frozenset()


# --- classify_tier -----------------------------------------------------------------------------


def test_tier1_when_agentic_correct_and_both_baselines_wrong():
    outcome = _outcome(agentic_grade="correct", baseline_grade="incorrect", matched_grade="incorrect")
    assert classify_tier(outcome) == 1


def test_tier2_when_agentic_correct_and_only_one_baseline_wrong():
    outcome = _outcome(agentic_grade="correct", baseline_grade="incorrect", matched_grade="correct")
    assert classify_tier(outcome) == 2


def test_none_when_agentic_not_correct():
    outcome = _outcome(agentic_grade="incorrect", baseline_grade="incorrect", matched_grade="incorrect")
    assert classify_tier(outcome) is None


def test_none_when_agentic_correct_and_both_baselines_also_correct():
    outcome = _outcome(agentic_grade="correct", baseline_grade="correct", matched_grade="correct")
    assert classify_tier(outcome) is None


def test_none_when_agentic_grade_missing():
    outcome = _outcome(agentic_grade=None, baseline_grade="incorrect", matched_grade="incorrect")
    assert classify_tier(outcome) is None


# --- select_examples -----------------------------------------------------------------------------


def test_select_examples_excludes_known_quirk_qa_id():
    candidates = [
        ExampleCandidate(qa_id="quirky", question_type="comparison_query", stop_reason="max_hops", tier=1),
        ExampleCandidate(qa_id="clean", question_type="inference_query", stop_reason="evidence_sufficient", tier=1),
    ]
    selected = select_examples(candidates, excluded_qa_ids=frozenset({"quirky"}), max_examples=5)
    assert selected == ["clean"]


def test_select_examples_prefers_tier1_over_tier2():
    candidates = [
        ExampleCandidate(qa_id="t2", question_type="comparison_query", stop_reason="max_hops", tier=2),
        ExampleCandidate(qa_id="t1", question_type="comparison_query", stop_reason="max_hops", tier=1),
    ]
    selected = select_examples(candidates, excluded_qa_ids=frozenset(), max_examples=1)
    assert selected == ["t1"]


def test_select_examples_maximizes_facet_diversity():
    candidates = [
        ExampleCandidate(qa_id="c1", question_type="comparison_query", stop_reason="max_hops", tier=1),
        ExampleCandidate(qa_id="c2", question_type="comparison_query", stop_reason="max_hops", tier=1),
        ExampleCandidate(qa_id="c3", question_type="inference_query", stop_reason="evidence_sufficient", tier=1),
    ]
    selected = select_examples(candidates, excluded_qa_ids=frozenset(), max_examples=2)
    # c1 (or c2, tied) picked first by qa_id tie-break, then c3 for its novel
    # (question_type, stop_reason) facets over the remaining same-facet candidate.
    assert selected == ["c1", "c3"]


def test_select_examples_caps_at_max_examples():
    candidates = [
        ExampleCandidate(qa_id=f"q{i}", question_type="comparison_query", stop_reason="max_hops", tier=1)
        for i in range(10)
    ]
    selected = select_examples(candidates, excluded_qa_ids=frozenset(), max_examples=5)
    assert len(selected) == 5


def test_select_examples_deterministic_repeatability():
    candidates = [
        ExampleCandidate(qa_id="c1", question_type="comparison_query", stop_reason="max_hops", tier=1),
        ExampleCandidate(qa_id="c2", question_type="inference_query", stop_reason="evidence_sufficient", tier=1),
        ExampleCandidate(qa_id="c3", question_type="temporal_query", stop_reason="token_budget", tier=2),
    ]
    first = select_examples(candidates, excluded_qa_ids=frozenset(), max_examples=5)
    second = select_examples(candidates, excluded_qa_ids=frozenset(), max_examples=5)
    assert first == second


def test_select_examples_qa_id_tie_break_is_ascending():
    candidates = [
        ExampleCandidate(qa_id="zzz", question_type="comparison_query", stop_reason="max_hops", tier=1),
        ExampleCandidate(qa_id="aaa", question_type="comparison_query", stop_reason="max_hops", tier=1),
    ]
    selected = select_examples(candidates, excluded_qa_ids=frozenset(), max_examples=1)
    assert selected == ["aaa"]
