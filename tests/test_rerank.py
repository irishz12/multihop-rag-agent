"""Reranking logic tests (offline — a fake scorer stands in for the real
cross-encoder, same pattern as tests/test_chunking.py's fake token counter):
tie-break/ordering, score preservation, metadata preservation, no
duplicates, no input mutation, and top_k truncation.

The real BAAI/bge-reranker-base model is exercised in
tests/test_rerank_integration.py.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mhrag.retrieval.rerank import rerank_results
from mhrag.retrieval.schema import RetrievalResult


class _FakeReranker:
    """Deterministic stand-in for mhrag.retrieval.rerank.Reranker — scores
    each text by a fixed lookup table, so test expectations are exact and
    don't depend on loading the real ~280M-param cross-encoder."""

    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []  # for batching assertions

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        self.calls.append((query, list(texts)))
        return np.array([self.scores_by_text[t] for t in texts])


def _candidate(chunk_id: str, text: str, rrf_rank: int, rrf_score: float) -> RetrievalResult:
    return RetrievalResult(
        rank=rrf_rank,
        score=rrf_score,
        method="hybrid",
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title=f"Title {chunk_id}",
        url=f"https://example.com/{chunk_id}",
        source="Source",
        category="technology",
        published_at="2024-01-01T00:00:00+00:00",
        text=text,
        position=0,
    )


# --- deterministic reranker ordering -------------------------------------------------


def test_rerank_orders_by_descending_score():
    candidates = [
        _candidate("a", "text-a", rrf_rank=1, rrf_score=0.05),
        _candidate("b", "text-b", rrf_rank=2, rrf_score=0.04),
        _candidate("c", "text-c", rrf_rank=3, rrf_score=0.03),
    ]
    # deliberately invert relevance vs. original RRF order
    reranker = _FakeReranker({"text-a": 0.1, "text-b": 0.9, "text-c": 0.5})
    reranked = rerank_results("query", candidates, reranker)
    assert [r.chunk_id for r in reranked] == ["b", "c", "a"]


def test_rerank_ranks_are_sequential_from_one():
    candidates = [_candidate(cid, f"text-{cid}", i + 1, 1 / (60 + i + 1)) for i, cid in enumerate("abcd")]
    reranker = _FakeReranker({f"text-{cid}": float(i) for i, cid in enumerate("abcd")})
    reranked = rerank_results("query", candidates, reranker)
    assert [r.rank for r in reranked] == [1, 2, 3, 4]


# --- reranker score preservation -----------------------------------------------------


def test_rrf_score_and_rerank_score_both_preserved():
    candidates = [_candidate("a", "text-a", rrf_rank=1, rrf_score=0.0164)]
    reranker = _FakeReranker({"text-a": 7.5})
    reranked = rerank_results("query", candidates, reranker)
    r = reranked[0]
    assert r.rrf_score == pytest.approx(0.0164)  # original RRF score preserved
    assert r.rerank_score == pytest.approx(7.5)  # new cross-encoder score
    assert r.score == pytest.approx(7.5)  # `score` reflects the new ranking score
    assert r.method == "hybrid_reranked"


def test_non_reranked_results_have_none_rrf_and_rerank_score():
    """Dense/bm25/hybrid results (constructed the normal way, not through
    rerank_results) must be unaffected by these new optional fields."""
    r = _candidate("a", "text-a", rrf_rank=1, rrf_score=0.5)
    assert r.rrf_score is None
    assert r.rerank_score is None


# --- metadata preservation ------------------------------------------------------------


def test_metadata_preserved_through_reranking():
    original = _candidate("a", "some chunk text", rrf_rank=1, rrf_score=0.5)
    reranker = _FakeReranker({"some chunk text": 3.0})
    reranked = rerank_results("query", [original], reranker)[0]
    assert reranked.chunk_id == original.chunk_id
    assert reranked.doc_id == original.doc_id
    assert reranked.title == original.title
    assert reranked.url == original.url
    assert reranked.source == original.source
    assert reranked.category == original.category
    assert reranked.published_at == original.published_at
    assert reranked.text == original.text
    assert reranked.position == original.position


# --- no duplicate chunk ids -----------------------------------------------------------


def test_no_duplicate_chunk_ids_in_reranked_output():
    candidates = [_candidate(cid, f"text-{cid}", i + 1, 1 / (60 + i + 1)) for i, cid in enumerate("abcde")]
    reranker = _FakeReranker({f"text-{cid}": float(i) for i, cid in enumerate("abcde")})
    reranked = rerank_results("query", candidates, reranker)
    chunk_ids = [r.chunk_id for r in reranked]
    assert len(chunk_ids) == len(set(chunk_ids)) == 5


# --- candidate count / top_k respected -------------------------------------------------


def test_top_k_truncates_after_reranking():
    candidates = [_candidate(cid, f"text-{cid}", i + 1, 1 / (60 + i + 1)) for i, cid in enumerate("abcdefgh")]
    reranker = _FakeReranker({f"text-{cid}": float(i) for i, cid in enumerate("abcdefgh")})
    reranked = rerank_results("query", candidates, reranker, top_k=3)
    assert len(reranked) == 3
    assert [r.rank for r in reranked] == [1, 2, 3]


def test_no_top_k_returns_all_candidates():
    candidates = [_candidate(cid, f"text-{cid}", i + 1, 1 / (60 + i + 1)) for i, cid in enumerate("abcde")]
    reranker = _FakeReranker({f"text-{cid}": float(i) for i, cid in enumerate("abcde")})
    reranked = rerank_results("query", candidates, reranker)
    assert len(reranked) == 5


def test_empty_candidates_returns_empty_list():
    reranker = _FakeReranker({})
    assert rerank_results("query", [], reranker) == []


# --- original hybrid ranking is not mutated ---------------------------------------------


def test_rerank_does_not_mutate_input_candidates():
    candidates = [_candidate(cid, f"text-{cid}", i + 1, 1 / (60 + i + 1)) for i, cid in enumerate("abc")]
    before = copy.deepcopy(candidates)
    reranker = _FakeReranker({"text-a": 0.9, "text-b": 0.1, "text-c": 0.5})

    rerank_results("query", candidates, reranker)

    assert candidates == before


def test_rerank_does_not_reorder_input_list():
    candidates = [_candidate(cid, f"text-{cid}", i + 1, 1 / (60 + i + 1)) for i, cid in enumerate("abc")]
    original_order = [c.chunk_id for c in candidates]
    reranker = _FakeReranker({"text-a": 0.9, "text-b": 0.1, "text-c": 0.5})

    rerank_results("query", candidates, reranker)

    assert [c.chunk_id for c in candidates] == original_order


# --- deterministic tie-break -----------------------------------------------------------


def test_tie_break_falls_through_to_original_rrf_rank_then_chunk_id():
    candidates = [
        _candidate("z", "text-z", rrf_rank=1, rrf_score=0.05),  # better RRF rank
        _candidate("a", "text-a", rrf_rank=5, rrf_score=0.01),  # worse RRF rank
    ]
    # tied reranker score -> must fall back to original RRF rank ascending
    reranker = _FakeReranker({"text-z": 2.0, "text-a": 2.0})
    reranked = rerank_results("query", candidates, reranker)
    assert [r.chunk_id for r in reranked] == ["z", "a"]  # z's better RRF rank wins the tie
