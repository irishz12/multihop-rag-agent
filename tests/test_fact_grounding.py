"""Unit tests for mhrag.eval.fact_grounding — offline, no model call, no I/O."""

from __future__ import annotations

from mhrag.eval.fact_grounding import (
    GoldFact,
    compute_question_fact_grounding,
    fact_grounded,
    normalize_fact_text,
)

# --- normalize_fact_text: conservative, never removes meaning -----------------------------


def test_normalize_collapses_whitespace():
    assert normalize_fact_text("The   quick\n\nbrown  fox") == "the quick brown fox"


def test_normalize_unifies_curly_quotes_and_dashes():
    assert normalize_fact_text("Bankman’Fried—the founder") == "bankman'fried-the founder"


def test_normalize_casefolds():
    assert normalize_fact_text("SAM Altman") == "sam altman"


def test_normalize_never_strips_punctuation_or_articles():
    """Deliberately different from answer_metrics.normalize_answer_text —
    a verbatim quote's punctuation/articles must survive normalization."""
    result = normalize_fact_text("The company, a startup, said 'no.'")
    assert "the company, a startup, said 'no.'" == result


# --- fact_grounded: exact substring, single-chunk only -------------------------------------


def test_fact_grounded_exact_match_in_one_chunk():
    fact = "Sam Bankman-Fried made himself out to be a genius."
    chunks = ["Before his fall, Sam Bankman-Fried made himself out to be a genius. He later..."]
    assert fact_grounded(fact, chunks) is True


def test_fact_grounded_case_and_whitespace_insensitive():
    fact = "google  faked   a demo"
    chunks = ["Reports say Google Faked A Demo of its new AI model."]
    assert fact_grounded(fact, chunks) is True


def test_fact_grounded_false_when_absent_from_all_chunks():
    fact = "Sam Bankman-Fried made himself out to be a genius."
    chunks = ["This chunk discusses an entirely unrelated topic about sports betting."]
    assert fact_grounded(fact, chunks) is False


def test_fact_grounded_checks_across_multiple_chunks_independently():
    fact = "the merger was announced on Tuesday"
    chunks = ["An unrelated chunk.", "Sources confirmed the merger was announced on Tuesday afternoon."]
    assert fact_grounded(fact, chunks) is True


def test_fact_grounded_does_not_match_partial_word_fragment():
    """A fact must match as a genuine substring, not accidentally split
    across two unrelated chunks (this module deliberately checks each
    chunk independently, never a concatenation — see module docstring)."""
    fact = "the merger was announced on Tuesday"
    chunks = ["...the merger was announced on", "Tuesday, sources confirmed..."]
    assert fact_grounded(fact, chunks) is False


def test_fact_grounded_empty_fact_never_matches():
    assert fact_grounded("", ["any chunk text"]) is False
    assert fact_grounded("   ", ["any chunk text"]) is False


def test_fact_grounded_empty_context_never_matches():
    assert fact_grounded("a real fact", []) is False


# --- compute_question_fact_grounding: per-question aggregation -----------------------------


def test_all_facts_grounded():
    facts = [GoldFact(fact="Google faked a demo", doc_id="doc1"), GoldFact(fact="Gemini made errors", doc_id="doc2")]
    chunks = ["Reports say Google faked a demo.", "Critics noted Gemini made errors in testing."]
    result = compute_question_fact_grounding("q1", "comparison_query", facts, chunks)
    assert result.n_gold_facts == 2
    assert result.n_grounded_facts == 2
    assert result.fact_grounded_rate == 1.0
    assert result.per_fact_grounded == (True, True)


def test_partial_facts_grounded():
    facts = [GoldFact(fact="Google faked a demo", doc_id="doc1"), GoldFact(fact="Gemini made errors", doc_id="doc2")]
    chunks = ["Reports say Google faked a demo."]  # second fact's doc never retrieved
    result = compute_question_fact_grounding("q2", "comparison_query", facts, chunks)
    assert result.n_gold_facts == 2
    assert result.n_grounded_facts == 1
    assert result.fact_grounded_rate == 0.5
    assert result.per_fact_grounded == (True, False)


def test_zero_facts_grounded():
    facts = [GoldFact(fact="Google faked a demo", doc_id="doc1")]
    chunks = ["This chunk is about something completely different."]
    result = compute_question_fact_grounding("q3", "inference_query", facts, chunks)
    assert result.n_grounded_facts == 0
    assert result.fact_grounded_rate == 0.0


def test_zero_gold_facts_returns_none_rate():
    result = compute_question_fact_grounding("q4", "null_query", [], ["some context"])
    assert result.n_gold_facts == 0
    assert result.fact_grounded_rate is None


def test_gold_fact_doc_ids_preserved_in_order():
    facts = [GoldFact(fact="A", doc_id="doc_a"), GoldFact(fact="B", doc_id="doc_b")]
    result = compute_question_fact_grounding("q5", "comparison_query", facts, [])
    assert result.gold_fact_doc_ids == ("doc_a", "doc_b")


# --- determinism -----------------------------------------------------------------------------


def test_deterministic_repeatability():
    facts = [GoldFact(fact="Google faked a demo", doc_id="doc1"), GoldFact(fact="Gemini made errors", doc_id="doc2")]
    chunks = ["Reports say Google faked a demo.", "Critics noted Gemini made errors."]
    first = compute_question_fact_grounding("q6", "comparison_query", facts, chunks)
    second = compute_question_fact_grounding("q6", "comparison_query", facts, chunks)
    assert first == second


def test_normalize_fact_text_is_deterministic():
    text = "The company, a startup, said ‘no.’"
    results = [normalize_fact_text(text) for _ in range(5)]
    assert all(r == results[0] for r in results)


# --- real-data regression: every fact from the actual dev-split evidence -------------------


def test_real_extractive_fact_matches_its_own_verbatim_sentence():
    """A real gold fact from the dataset (schema.py's own documented
    known-good example) must match when the exact sentence is present."""
    fact = "Before his fall, Bankman-Fried made himself out to be a genius"
    chunk = "Before his fall, Bankman-Fried made himself out to be a genius who could do no wrong."
    assert fact_grounded(fact, [chunk]) is True
