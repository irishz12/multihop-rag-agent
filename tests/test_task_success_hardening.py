"""Phase 3 hardening regression tests — hand-built fixtures for the two
discovered evaluator weaknesses:

  1. Verdict extraction false positive: "there is no praise for..." was
     read as the verdict "no" (a determiner use, not a commitment).
  2. Abstention false positive: a response opening with an abstention
     phrase but going on to substantively answer was flatly classified
     as abstention with no way to distinguish it from a clean refusal.

Every case explicitly required by the Phase 3 prompt is tested here,
verbatim. Also re-verifies Phase 2 backward compatibility: `is_abstention`/
`classify_abstention`/`ABSTENTION_STATUSES` are untouched (see
tests/test_task_success.py, still passing unmodified), and the negation
trap / leading / explicit tiers from Phase 2 still resolve correctly
(tier 1/2 were never the source of the bug).
"""

from __future__ import annotations

from mhrag.eval.task_success import (
    RESPONSE_STRUCTURE_STATES,
    classify_response_structure,
    extract_verdict,
)

# --- 1. Verdict extraction: the 10 explicitly required cases -----------------------------


def test_leading_no_disagree():
    r = extract_verdict("No, the sources disagree.")
    assert r.verdict == "no" and r.is_ambiguous is False


def test_leading_yes_agree():
    r = extract_verdict("Yes, they agree.")
    assert r.verdict == "yes" and r.is_ambiguous is False


def test_explicit_the_answer_is_no():
    r = extract_verdict("The answer is no.")
    assert r.verdict == "no" and r.is_ambiguous is False


def test_explicit_overall_yes():
    r = extract_verdict("Overall, yes.")
    assert r.verdict == "yes" and r.is_ambiguous is False


def test_determiner_no_evidence_is_not_a_verdict():
    """THE discovered false positive's exact pattern family — 'no' as a
    determiner before a noun, mid-clause, must NOT resolve to a verdict."""
    r = extract_verdict("There is no evidence that...")
    assert r.is_ambiguous is True
    assert r.verdict is None


def test_determiner_no_agreement_is_not_a_verdict():
    """The literal originally-discovered failure case's sibling wording."""
    r = extract_verdict("There is no agreement between the sources.")
    assert r.is_ambiguous is True
    assert r.verdict is None


def test_original_discovered_failure_case_verbatim():
    """The EXACT sentence that motivated this hardening phase."""
    r = extract_verdict(
        "There is no praise for the Biden administration or positive assessment of the "
        "interconnected forces described."
    )
    assert r.is_ambiguous is True
    assert r.verdict is None


def test_negation_trap_still_resolves_correctly_after_hardening():
    """Tier 1/2 (leading/explicit) were never the bug — must be unaffected."""
    r = extract_verdict("I don't think the answer is no; it is yes.")
    assert r.verdict == "yes" and r.is_ambiguous is False


def test_not_no_but_yes_still_resolves_via_leading_tier():
    r = extract_verdict("Not no, but yes.")
    assert r.verdict == "yes" and r.is_ambiguous is False


def test_it_is_unclear_is_ambiguous():
    r = extract_verdict("It is unclear.")
    assert r.is_ambiguous is True
    assert r.verdict is None


def test_yes_or_no_hedge_is_ambiguous():
    r = extract_verdict("I cannot determine whether the answer is yes or no.")
    assert r.is_ambiguous is True
    assert r.verdict is None


def test_clause_final_no_is_still_accepted():
    """A genuine clause-final verdict, phrased differently from the
    project's narrow explicit-pattern set, must still resolve — the
    hardening must not become so strict it stops extracting real
    verdicts entirely."""
    r = extract_verdict("Based on the evidence, the correct answer here is no.")
    assert r.verdict == "no" and r.is_ambiguous is False


# --- 2. Abstention / response structure: the 7 explicitly required cases -----------------


def test_clean_abstention_bare():
    assert classify_response_structure("Insufficient information to answer.") == "clean_abstention"


def test_clean_abstention_with_answer_after_however_is_answer_with_uncertainty():
    result = classify_response_structure(
        "Insufficient information to answer; however, the answer is Google."
    )
    assert result == "answer_with_uncertainty"


def test_hedge_before_answer_is_answer_with_uncertainty():
    result = classify_response_structure("I cannot be certain, but the answer is Google.")
    assert result == "answer_with_uncertainty"


def test_limited_evidence_hedge_with_answer_is_answer_with_uncertainty():
    result = classify_response_structure(
        "The available evidence is limited, but the sources indicate Google."
    )
    assert result == "answer_with_uncertainty"


def test_paraphrased_decline_not_in_old_allowlist_is_still_clean_abstention():
    """'don't have enough information' is NOT a substring of any entry in
    the existing (unmodified) ABSTENTION_PHRASES allowlist — this is
    exactly why a richer classifier was required, not an allowlist
    expansion of that specific list."""
    result = classify_response_structure("I don't have enough information to determine this.")
    assert result == "clean_abstention"


def test_plain_substantive_answer_no_hedge_language():
    assert classify_response_structure("The answer is Google.") == "substantive_answer"


def test_answer_before_hedge_is_a_known_accepted_phase4_false_negative():
    """PHASE 4 REGRESSION, DOCUMENTED AND ACCEPTED — see
    RESPONSE_STRUCTURE_RELIABILITY in task_success.py. The Phase 3
    residue-word classifier caught this case (answer stated BEFORE its
    hedge), but that same permissive word-counting was what produced a
    5/6 false-positive rate on real data (see the Phase 3->4 audit). The
    Phase 4 structural (assertion-marker) classifier requires the marker
    to PRECEDE the named entity, so a bare leading entity name with no
    assertion marker is missed. This was evaluated deliberately: fixing
    it would require treating sentence-initial capitalization as an
    entity signal, which is unreliable (ordinary sentence-initial words
    like "The"/"It"/"This" are also capitalized) and would reopen a
    different false-positive class. Accepted trade-off, not an oversight
    — this test exists to make the regression visible, not hide it."""
    result = classify_response_structure("Google, although the evidence is limited.")
    assert result == "ambiguous"


def test_bare_hedge_no_decline_no_answer_is_ambiguous():
    assert classify_response_structure("It is unclear.") == "ambiguous"


def test_empty_answer_is_ambiguous():
    assert classify_response_structure("   ") == "ambiguous"


def test_response_structure_states_are_exhaustive():
    assert len(set(RESPONSE_STRUCTURE_STATES)) == 4


# --- 3. Original Phase 2 hedge-then-answer discovery, reclassified -----------------------


def test_original_validation_sample_hedge_then_entity_case():
    """The real qa_id 0898e5bf16fa4a76 from Phase 2's --validate run:
    abstention-framed opening, but the model goes on to correctly name
    the entity. response_structure must recognize this as containing a
    substantive answer, unlike the flat old abstention_status field."""
    text = (
        "The available information is insufficient to answer the question. "
        "While Source 1 identifies Travis Kelce as the athlete who helped his team "
        "win Super Bowls in 2020 and 2023, and mentions his connection to Taylor Swift, "
        "there is no mention in any of the provided sources of Kelce being present at one "
        "of Swift's concerts in Buenos Aires in November."
    )
    assert classify_response_structure(text) == "answer_with_uncertainty"


# --- 4. Determinism (same discipline as every other function in this module) -------------


def test_response_structure_is_deterministic():
    text = "Insufficient information to answer; however, the answer is Google."
    results = [classify_response_structure(text) for _ in range(5)]
    assert all(r == results[0] for r in results)
