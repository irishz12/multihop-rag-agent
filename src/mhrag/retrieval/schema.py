"""Typed schema for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk returned for a query, from any retrieval method.

    `doc_id` is carried through unchanged from the indexed chunk's source
    document, so results can always be traced back to the original
    document/evidence ground truth used in later evaluation.

    The same schema is shared across dense, BM25, hybrid, and (since Phase 5)
    reranked retrieval (`method` records which) so their outputs are
    directly comparable — but `score` is NOT comparable across methods:
    it's cosine similarity (bounded ~[0,1] for normalized vectors) for
    "dense", an unbounded IDF-weighted BM25 relevance score for "bm25", an
    RRF fusion score for "hybrid" — as of Phase 4.1, the canonical hybrid
    score is the deterministic, application-side RRF@k=60 computed by
    `mhrag.retrieval.rrf.rrf_fuse` (see that module for the exact formula
    and rank convention); Qdrant's native server-side fusion
    (`mhrag.retrieval.hybrid.hybrid_search`, unchanged) is kept only as a
    reference implementation and is no longer used for the benchmark — and
    a sigmoid-bounded ~[0,1] cross-encoder relevance score for
    "hybrid_reranked" (see `mhrag.retrieval.rerank`). Only `rank` is safe
    to compare across methods.

    `rrf_score`/`rerank_score` (Phase 5): populated only on
    "hybrid_reranked" results, so a reranked result still carries the RRF
    score/rank it arrived with alongside its new reranker score — both
    `None` for every other method's results (dense/bm25/hybrid never touch
    these fields; unaffected by this addition).
    """

    rank: int  # 1-indexed within this query's result list
    score: float  # meaning depends on `method` — see class docstring
    method: str  # "dense" | "bm25" | "hybrid" | "hybrid_reranked"
    chunk_id: str
    doc_id: str
    title: str
    url: str
    source: str
    category: str
    published_at: str
    text: str
    position: int
    rrf_score: float | None = None  # Phase 5: original hybrid RRF score, preserved when reranked
    rerank_score: float | None = None  # Phase 5: cross-encoder score (== `score` when method="hybrid_reranked")
