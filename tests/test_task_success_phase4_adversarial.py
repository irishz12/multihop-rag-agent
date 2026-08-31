"""Phase 4 — substantially stronger adversarial regression suite.

Explicitly NOT limited to the specific sentences that motivated Phase 3/4
(the "no praise"/"climate change" false positives) — this file constructs
new adversarial cases across the categories the Phase 4 audit brief
requires: negation, compound nouns, noun phrases, lists, appositives,
quoted text, hedging, refusal explanations, answer+uncertainty,
uncertainty without an answer, contradictory statements, and multiple
yes/no mentions. Several of these cases were NOT known failures going in
— constructing them is what surfaced the "It's uncertain whether they
align." false positive that this phase's `_UNCERTAINTY_BLOCKERS` fix
addresses (see its own dedicated test below).
"""

from __future__ import annotations

from mhrag.eval.task_success import extract_verdict

# --- the 11 explicitly required cases (verbatim) ------------------------------------------


def test_required_no_sources_disagree():
    assert extract_verdict("No, the sources disagree.").verdict == "no"


def test_required_yes_they_agree():
    assert extract_verdict("Yes, they agree.").verdict == "yes"


def test_required_the_answer_is_no():
    assert extract_verdict("The answer is no.").verdict == "no"


def test_required_overall_yes():
    assert extract_verdict("Overall, yes.").verdict == "yes"


def test_required_there_is_no_evidence():
    r = extract_verdict("There is no evidence that...")
    assert r.is_ambiguous and r.verdict is None


def test_required_there_is_no_agreement():
    r = extract_verdict("There is no agreement...")
    assert r.is_ambiguous and r.verdict is None


def test_required_climate_change_verbatim():
    """The literal example named in the Phase 4 brief."""
    r = extract_verdict("climate change, the modern internet, and authoritarianism")
    assert r.is_ambiguous and r.verdict is None


def test_required_answer_is_unclear():
    r = extract_verdict("The answer is unclear.")
    assert r.is_ambiguous and r.verdict is None


def test_required_yes_or_no_meta():
    r = extract_verdict("I cannot determine whether it is yes or no.")
    assert r.is_ambiguous and r.verdict is None


def test_required_negation_trap():
    assert extract_verdict("I don't think the answer is no; it is yes.").verdict == "yes"


def test_required_not_no_but_yes():
    assert extract_verdict("Not no, but yes.").verdict == "yes"


# --- compound nouns (beyond the one named example) ----------------------------------------


def test_compound_noun_no_change_in_coverage():
    """'no change' as a fixed phrase describing stasis must not be read
    as two independent tokens — the system has no compound-phrase
    semantics, so the safe outcome is ambiguous, not a guessed flip."""
    r = extract_verdict("There was no change in the coverage.")
    assert r.is_ambiguous and r.verdict is None


def test_compound_noun_regime_change_context():
    r = extract_verdict("The article discusses regime change and its consequences for the region.")
    assert r.is_ambiguous and r.verdict is None


# --- noun phrases, two canonical words in one sentence -------------------------------------


def test_noun_phrase_no_indication_that_they_align():
    r = extract_verdict("There is no indication that they align on this point.")
    assert r.is_ambiguous and r.verdict is None


# --- lists -----------------------------------------------------------------------------------


def test_list_of_candidate_verdicts():
    r = extract_verdict(
        "The report considered several verdicts: yes, no, and maybe, before settling on an interpretation."
    )
    assert r.is_ambiguous and r.verdict is None


def test_list_with_and_conjunction_no_final_commitment():
    r = extract_verdict("Sources described the outcome as similar, consistent, and unremarkable overall trends.")
    assert r.is_ambiguous and r.verdict is None


# --- appositives -------------------------------------------------------------------------


def test_appositive_no_friend_of_the_administration():
    r = extract_verdict("The report, no friend of the administration, criticized the policy.")
    assert r.is_ambiguous and r.verdict is None


# --- quoted text -------------------------------------------------------------------------


def test_quoted_no_does_not_leak_but_real_conclusion_does():
    """A quoted 'no' deep inside the sentence must not be extracted; the
    sentence's own unquoted conclusion ('concludes yes') correctly is —
    it really is sentence-final and really is the sentence's own verdict."""
    r = extract_verdict('The article states "there is no way to know," but concludes yes.')
    assert r.verdict == "yes"


# --- hedging, no clean commitment ----------------------------------------------------------


def test_hedge_might_be_either():
    r = extract_verdict("It might be yes, it might be no, hard to say for certain.")
    assert r.is_ambiguous and r.verdict is None


def test_hedge_probably_yes_not_certain():
    """A leaning-but-hedged answer with no clean sentence-final
    commitment — the deliberately conservative outcome is ambiguous, not
    a guessed 'yes'."""
    r = extract_verdict("Probably yes, though I cannot be 100% sure.")
    assert r.is_ambiguous and r.verdict is None


# --- refusal explanations (paraphrased, not the exact known example) ----------------------


def test_refusal_explanation_no_confirmation():
    r = extract_verdict(
        "The sources don't explicitly confirm consistency, so no clear determination can be made."
    )
    assert r.is_ambiguous and r.verdict is None


# --- uncertainty without an answer at all --------------------------------------------------


def test_uncertainty_without_answer_align():
    """THE case adversarial construction discovered during this phase:
    'align' is sentence-final, but 'uncertain whether' explicitly hedges
    it — must not be extracted as a confident verdict. Fixed by
    _UNCERTAINTY_BLOCKERS (new in Phase 4)."""
    r = extract_verdict("It's uncertain whether they align.")
    assert r.is_ambiguous and r.verdict is None


def test_uncertainty_without_answer_unsure():
    r = extract_verdict("I'm unsure whether the reports remain consistent.")
    assert r.is_ambiguous and r.verdict is None


# --- contradictory statements / multiple yes-no mentions -----------------------------------


def test_contradictory_then_resolved_overall():
    """Multiple conflicting mid-text mentions, resolved by an explicit
    final commitment — the sentence-final 'no' after 'Overall' must win,
    not an earlier 'yes'."""
    r = extract_verdict("Some evidence suggests yes, but other evidence suggests no. Overall, the answer remains no.")
    assert r.verdict == "no"


def test_multiple_mentions_no_final_resolution_is_ambiguous():
    r = extract_verdict("One source says yes, another says no, and a third is silent on the matter entirely.")
    assert r.is_ambiguous and r.verdict is None
