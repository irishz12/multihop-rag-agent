"""Deterministic application-side RRF tests: exact score formula, k=60,
equal weighting, deterministic tie-break, no duplicate chunk ids, and that
fusion never mutates its input lists.
"""

from __future__ import annotations

import copy

import pytest

from mhrag.retrieval.rrf import BM25_WEIGHT, DENSE_WEIGHT, RRF_K, rrf_fuse
from mhrag.retrieval.schema import RetrievalResult


def _result(chunk_id: str, doc_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        rank=0,  # overwritten by rrf_fuse; placeholder here
        score=0.0,  # overwritten by rrf_fuse; placeholder here
        method="dense",
        chunk_id=chunk_id,
        doc_id=doc_id or f"doc-{chunk_id}",
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        source="Source",
        category="technology",
        published_at="2024-01-01T00:00:00+00:00",
        text="chunk text",
        position=0,
    )


# --- k=60 is the default, and is actually used in the formula -------------------------


def test_rrf_k_constant_is_60():
    assert RRF_K == 60


def test_rrf_k_default_parameter_matches_the_module_constant():
    dense = [_result("a"), _result("b")]
    bm25 = [_result("b"), _result("a")]
    default_call = rrf_fuse(dense, bm25)
    explicit_call = rrf_fuse(dense, bm25, k=RRF_K)
    assert [(r.chunk_id, r.score) for r in default_call] == [
        (r.chunk_id, r.score) for r in explicit_call
    ]


def test_changing_k_changes_the_fused_score():
    dense = [_result("a"), _result("b")]
    bm25 = [_result("b"), _result("a")]
    scores_k60 = {r.chunk_id: r.score for r in rrf_fuse(dense, bm25, k=60)}
    scores_k2 = {r.chunk_id: r.score for r in rrf_fuse(dense, bm25, k=2)}
    assert scores_k60 != scores_k2
    # k=2 must produce LARGER scores than k=60 (smaller denominator)
    assert scores_k2["a"] > scores_k60["a"]


# --- exact RRF score calculation, 1-based rank -------------------------------------------


def test_exact_score_for_chunk_in_both_lists():
    """chunk "a" is rank 1 in dense, rank 2 in bm25 -> score =
    1/(60+1) + 1/(60+2), using the documented 1-based convention."""
    dense = [_result("a"), _result("b")]
    bm25 = [_result("b"), _result("a")]
    fused = {r.chunk_id: r.score for r in rrf_fuse(dense, bm25, k=60)}
    expected_a = 1 / (60 + 1) + 1 / (60 + 2)
    assert fused["a"] == pytest.approx(expected_a)


def test_exact_score_for_chunk_in_only_one_list():
    """chunk present in dense only (rank 3) contributes just that term —
    not penalized as if absent-from-bm25 were some worst-case rank."""
    dense = [_result("x"), _result("y"), _result("a")]
    bm25 = [_result("x"), _result("y")]  # "a" absent from bm25
    fused = {r.chunk_id: r.score for r in rrf_fuse(dense, bm25, k=60)}
    expected_a = 1 / (60 + 3)  # dense rank 3 only
    assert fused["a"] == pytest.approx(expected_a)


def test_top_ranked_chunk_in_a_list_contributes_one_over_k_plus_one():
    """Precise 1-based-convention check: the #1 chunk in a list contributes
    1/(k+1), NOT 1/k (which would be the 0-based convention — see module
    docstring for why Qdrant's own native RRF differs on this point)."""
    dense = [_result("a")]
    bm25 = []
    fused = {r.chunk_id: r.score for r in rrf_fuse(dense, bm25, k=60)}
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["a"] != pytest.approx(1 / 60)


# --- equal weighting -----------------------------------------------------------------------


def test_weights_are_equal():
    assert DENSE_WEIGHT == BM25_WEIGHT == 1.0


def test_equal_weighting_means_symmetric_contribution():
    """A chunk at rank 1 in dense-only vs. a (different) chunk at rank 1 in
    bm25-only must score identically — proves neither list is weighted more
    than the other."""
    dense = [_result("a")]
    bm25 = [_result("b")]
    fused = {r.chunk_id: r.score for r in rrf_fuse(dense, bm25, k=60)}
    assert fused["a"] == pytest.approx(fused["b"])


# --- deterministic tie-break ----------------------------------------------------------------


def test_tie_break_prefers_better_best_rank_when_scores_tie():
    """Two chunks with an EXACTLY equal fused score but DIFFERENT best
    individual rank must be ordered by best rank (rule 2), not by chunk_id
    (rule 3 only applies once rules 1 and 2 both tie). Uses k=1 purely for
    clean arithmetic — the tie-break logic itself doesn't depend on k.

    "z" (chunk_id sorts AFTER "a"): dense rank 1 only -> score = 1/(1+1) = 0.5, best_rank=1
    "a" (chunk_id sorts BEFORE "z"): dense rank 3 + bm25 rank 3 ->
        score = 1/(1+3) + 1/(1+3) = 0.5 (tied with "z"), best_rank=3

    Despite "a" < "z" lexicographically, "z" must win — its score ties but
    its best_rank (1) beats "a"'s (3).
    """
    dense = [_result("z"), _result("filler-d1"), _result("a")]  # z=rank1, a=rank3
    bm25 = [_result("filler-b1"), _result("filler-b2"), _result("a")]  # a=rank3

    fused = rrf_fuse(dense, bm25, k=1)
    z_score = next(r.score for r in fused if r.chunk_id == "z")
    a_score = next(r.score for r in fused if r.chunk_id == "a")
    assert z_score == pytest.approx(a_score) == pytest.approx(0.5)  # confirmed tie

    z_index = next(i for i, r in enumerate(fused) if r.chunk_id == "z")
    a_index = next(i for i, r in enumerate(fused) if r.chunk_id == "a")
    assert z_index < a_index, "better best_rank must win even though 'a' < 'z' lexicographically"


def test_tie_break_falls_through_to_chunk_id_when_score_and_rank_both_tie():
    """Two chunks appearing in NEITHER shared list, both absent from the
    other side entirely, with identical rank 1 in their respective
    single list -> identical score AND identical best_rank -> must break
    the tie by chunk_id ascending."""
    dense = [_result("zzz")]
    bm25 = [_result("aaa")]
    fused = rrf_fuse(dense, bm25, k=60)
    assert fused[0].chunk_id == "aaa"
    assert fused[1].chunk_id == "zzz"


def test_tie_break_is_stable_across_repeated_calls_with_many_ties():
    """A larger set of same-rank, single-list chunks (all tied on score and
    best_rank) must always sort into the same chunk_id order, every call."""
    dense = [_result(cid) for cid in ["m", "z", "a", "q"]]  # all rank-varying in dense
    bm25: list[RetrievalResult] = []
    first = [r.chunk_id for r in rrf_fuse(dense, bm25, k=60)]
    second = [r.chunk_id for r in rrf_fuse(dense, bm25, k=60)]
    assert first == second
    # dense ranks differ (1,2,3,4) so this is actually rank-ordered, not a
    # tie-break case — but reproducibility must hold regardless.
    assert first == ["m", "z", "a", "q"]


# --- no duplicate chunk ids ------------------------------------------------------------------


def test_no_duplicate_chunk_ids_in_fused_output():
    dense = [_result("a"), _result("b"), _result("c")]
    bm25 = [_result("b"), _result("c"), _result("d")]  # "b","c" overlap with dense
    fused = rrf_fuse(dense, bm25, k=60)
    chunk_ids = [r.chunk_id for r in fused]
    assert len(chunk_ids) == len(set(chunk_ids)) == 4  # a,b,c,d — each once


def test_fused_ranks_are_sequential_from_one():
    dense = [_result("a"), _result("b")]
    bm25 = [_result("c"), _result("d")]
    fused = rrf_fuse(dense, bm25, k=60)
    assert [r.rank for r in fused] == [1, 2, 3, 4]


def test_final_top_k_truncates_after_fusion():
    dense = [_result(f"d{i}") for i in range(10)]
    bm25 = [_result(f"b{i}") for i in range(10)]
    fused = rrf_fuse(dense, bm25, k=60, final_top_k=5)
    assert len(fused) == 5
    assert [r.rank for r in fused] == [1, 2, 3, 4, 5]


# --- identical repeated rankings (pure-function determinism) --------------------------------


def test_repeated_fusion_is_bit_identical():
    dense = [_result(f"d{i}") for i in range(8)]
    bm25 = [_result(f"d{i}") for i in reversed(range(8))]  # overlapping, reordered
    first = rrf_fuse(dense, bm25, k=60)
    second = rrf_fuse(dense, bm25, k=60)
    assert [(r.chunk_id, r.rank, r.score) for r in first] == [
        (r.chunk_id, r.rank, r.score) for r in second
    ]


# --- dense/BM25 inputs remain unchanged ------------------------------------------------------


def test_rrf_fuse_does_not_mutate_input_lists():
    dense = [_result("a"), _result("b")]
    bm25 = [_result("b"), _result("c")]
    dense_before = copy.deepcopy(dense)
    bm25_before = copy.deepcopy(bm25)

    rrf_fuse(dense, bm25, k=60)

    assert dense == dense_before
    assert bm25 == bm25_before


def test_rrf_fuse_does_not_reorder_input_lists():
    dense = [_result("a"), _result("b"), _result("c")]
    bm25 = [_result("c"), _result("b"), _result("a")]
    original_dense_order = [r.chunk_id for r in dense]
    original_bm25_order = [r.chunk_id for r in bm25]

    rrf_fuse(dense, bm25, k=60)

    assert [r.chunk_id for r in dense] == original_dense_order
    assert [r.chunk_id for r in bm25] == original_bm25_order
