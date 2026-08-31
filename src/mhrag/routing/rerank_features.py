"""RUNTIME reranker-derived signals — Phase 8A.2's Stage 2 features, on
top of Stage 1's `mhrag.routing.features.RetrievalSignals`. Pure function
of already-computed `RetrievalResult` lists (the fused Hybrid pool the
reranker scored, and its own top-5 output) — no gold, no live call here
(the live reranking call itself is the frozen, unmodified
`mhrag.retrieval.rerank.rerank_results`; this module only reads its
output).
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.retrieval.schema import RetrievalResult


@dataclass(frozen=True, slots=True)
class RerankSignals:
    rerank_top1_score: float
    rerank_top5_mean_score: float
    rerank_score_gap_top1_top2: float
    rerank_score_gap_top1_top5: float
    rank_change_mean_abs: float  # mean |pre-rerank pool rank - post-rerank rank| over reranked top-5 docs
    top5_overlap_with_hybrid: float  # fraction (0..1) of reranked top-5 docs also in the pre-rerank top-5
    num_docs_new_in_rerank_top5: int  # 5 - (that overlap count)


def extract_rerank_signals(
    fused_pool: list[RetrievalResult],
    reranked_top5: list[RetrievalResult],
) -> RerankSignals:
    """`fused_pool` is the pre-rerank Hybrid RRF candidate pool the
    reranker actually scored (its `.rank` there is each chunk's ORIGINAL
    hybrid-pool position); `reranked_top5` is the reranker's own top-5
    output (`.rank` there is the NEW post-rerank position, 1-5)."""
    pre_rerank_rank_by_chunk = {r.chunk_id: r.rank for r in fused_pool}
    pre_rerank_top5_docs = {r.doc_id for r in fused_pool[:5]}

    scores = [r.score for r in reranked_top5]
    top1_score = scores[0] if scores else 0.0
    top5_mean_score = sum(scores) / len(scores) if scores else 0.0
    gap_1_2 = scores[0] - scores[1] if len(scores) >= 2 else 0.0
    gap_1_5 = scores[0] - scores[-1] if len(scores) >= 2 else 0.0

    rank_diffs = []
    reranked_top5_docs = set()
    for r in reranked_top5:
        reranked_top5_docs.add(r.doc_id)
        original_rank = pre_rerank_rank_by_chunk.get(r.chunk_id, len(fused_pool) + 1)
        rank_diffs.append(abs(original_rank - r.rank))
    rank_change_mean_abs = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0.0

    overlap_docs = reranked_top5_docs & pre_rerank_top5_docs
    overlap_fraction = len(overlap_docs) / len(reranked_top5_docs) if reranked_top5_docs else 0.0
    num_new_docs = len(reranked_top5_docs) - len(overlap_docs)

    return RerankSignals(
        rerank_top1_score=top1_score,
        rerank_top5_mean_score=top5_mean_score,
        rerank_score_gap_top1_top2=gap_1_2,
        rerank_score_gap_top1_top5=gap_1_5,
        rank_change_mean_abs=rank_change_mean_abs,
        top5_overlap_with_hybrid=overlap_fraction,
        num_docs_new_in_rerank_top5=num_new_docs,
    )
