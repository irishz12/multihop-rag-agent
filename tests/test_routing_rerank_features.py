"""Reranker signal extraction tests — offline, pure function, no models."""

from __future__ import annotations

import inspect

from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.rerank_features import extract_rerank_signals


def _pool_result(chunk_id, doc_id, rank, score=0.05) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=score, method="hybrid", chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text="text", position=0,
    )


def _reranked_result(chunk_id, doc_id, rank, score) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=score, method="hybrid_reranked", chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text="text", position=0,
        rrf_score=0.03, rerank_score=score,
    )


def test_no_reranking_change_gives_zero_rank_change_and_full_overlap():
    fused_pool = [_pool_result(f"c{i}", f"doc{i}", i + 1) for i in range(20)]
    # Reranker keeps the exact same order for the top 5.
    reranked_top5 = [_reranked_result(f"c{i}", f"doc{i}", i + 1, 1.0 - i * 0.1) for i in range(5)]
    signals = extract_rerank_signals(fused_pool, reranked_top5)
    assert signals.rank_change_mean_abs == 0.0
    assert signals.top5_overlap_with_hybrid == 1.0
    assert signals.num_docs_new_in_rerank_top5 == 0


def test_full_reshuffle_gives_nonzero_rank_change_and_no_overlap():
    fused_pool = [_pool_result(f"c{i}", f"doc{i}", i + 1) for i in range(20)]
    # Reranker promotes docs 15-19 (originally ranks 16-20) into the new top 5.
    reranked_top5 = [_reranked_result(f"c{15 + i}", f"doc{15 + i}", i + 1, 1.0 - i * 0.1) for i in range(5)]
    signals = extract_rerank_signals(fused_pool, reranked_top5)
    assert signals.rank_change_mean_abs > 0.0
    assert signals.top5_overlap_with_hybrid == 0.0
    assert signals.num_docs_new_in_rerank_top5 == 5


def test_rank_change_mean_abs_exact_value():
    fused_pool = [_pool_result("c0", "doc0", 1), _pool_result("c1", "doc1", 6)]
    # c1 moves from pool rank 6 to reranked rank 1 -> |6-1|=5; c0 stays rank1->rank2 -> |1-2|=1
    reranked_top5 = [_reranked_result("c1", "doc1", 1, 0.9), _reranked_result("c0", "doc0", 2, 0.5)]
    signals = extract_rerank_signals(fused_pool, reranked_top5)
    assert signals.rank_change_mean_abs == (5 + 1) / 2


def test_score_gaps_computed_from_reranked_scores():
    fused_pool = [_pool_result(f"c{i}", f"doc{i}", i + 1) for i in range(5)]
    reranked_top5 = [_reranked_result(f"c{i}", f"doc{i}", i + 1, 0.9 - i * 0.1) for i in range(5)]
    signals = extract_rerank_signals(fused_pool, reranked_top5)
    assert signals.rerank_top1_score == 0.9
    assert abs(signals.rerank_score_gap_top1_top2 - 0.1) < 1e-9
    assert abs(signals.rerank_score_gap_top1_top5 - 0.4) < 1e-9


def test_empty_reranked_top5_returns_zeros():
    signals = extract_rerank_signals([], [])
    assert signals.rerank_top1_score == 0.0
    assert signals.rank_change_mean_abs == 0.0
    assert signals.top5_overlap_with_hybrid == 0.0
    assert signals.num_docs_new_in_rerank_top5 == 0


def test_extract_rerank_signals_signature_has_no_gold_parameter():
    params = list(inspect.signature(extract_rerank_signals).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label"}
    assert not (forbidden & set(params))
    assert params == ["fused_pool", "reranked_top5"]
