"""Answer-quality string metrics tests — pure functions, fully offline."""

from __future__ import annotations

from mhrag.eval.answer_metrics import exact_match, is_abstention, normalize_answer_text, token_f1

# --- normalize_answer_text --------------------------------------------------------------------


def test_normalize_lowercases_strips_punctuation_and_articles():
    assert normalize_answer_text("The Sam Bankman-Fried, Inc.") == "sam bankmanfried inc"


def test_normalize_collapses_to_empty_for_pure_stopword_punctuation():
    assert normalize_answer_text("The, a an.") == ""


# --- exact_match -------------------------------------------------------------------------------


def test_exact_match_identical_after_normalization():
    assert exact_match("The Sam Bankman-Fried", "sam bankman fried") == 0  # hyphen removal differs from space
    assert exact_match("Sam Bankman Fried", "The Sam Bankman Fried") == 1


def test_exact_match_case_and_article_insensitive():
    assert exact_match("an Apple", "THE APPLE") == 1


def test_exact_match_zero_when_different():
    assert exact_match("Apple", "Orange") == 0


# --- token_f1 -----------------------------------------------------------------------------------


def test_token_f1_perfect_match_is_one():
    assert token_f1("Sam Bankman Fried", "sam bankman fried") == 1.0


def test_token_f1_partial_overlap():
    # pred={sam,bankman}, gold={sam,bankman,fried} -> precision=2/2=1, recall=2/3, f1=2*1*(2/3)/(1+2/3)=0.8
    assert abs(token_f1("Sam Bankman", "Sam Bankman Fried") - 0.8) < 1e-9


def test_token_f1_no_overlap_is_zero():
    assert token_f1("Apple", "Orange") == 0.0


def test_token_f1_both_empty_after_normalization_is_one():
    assert token_f1("the a an", "an the") == 1.0


def test_token_f1_one_empty_is_zero():
    assert token_f1("the a an", "Orange") == 0.0


# --- is_abstention -------------------------------------------------------------------------------


def test_is_abstention_true_for_documented_phrasings():
    assert is_abstention("The available information is insufficient to answer.")
    assert is_abstention("There is not enough information to determine this.")
    assert is_abstention("This cannot be determined from the given context.")
    assert is_abstention("The context does not contain enough information to answer.")


def test_is_abstention_false_for_a_normal_answer():
    assert not is_abstention("The team known for safe handling is the All Blacks.")


def test_is_abstention_case_insensitive():
    assert is_abstention("INSUFFICIENT INFORMATION.")
