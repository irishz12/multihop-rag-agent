"""Cross-encoder reranking on top of the deterministic Hybrid RRF baseline.

    Dense (top-20) + BM25 (top-20)
              ↓
    deterministic RRF, k=60 (mhrag.retrieval.rrf — unmodified)
              ↓
       fused top-20 chunks
              ↓
    cross-encoder reranker: score(query, chunk_text)
              ↓
       reranked top `final_top_k`

`Reranker` wraps sentence-transformers' `CrossEncoder` around
`BAAI/bge-reranker-base`, run locally (CPU by default — no external API
call, no new dependency: `CrossEncoder` ships in the same
`sentence-transformers` package already used for the dense embedding
model). It scores `(query, candidate_chunk_text)` pairs only — ground-truth
`answer`/`evidence_list` are never passed to it (see `rerank_results`,
which only ever reads `.text` off already-retrieved `RetrievalResult`
objects, the same isolation contract as `dense_search`/`bm25_search`).

This module does NOT modify `mhrag.retrieval.{dense,bm25,rrf}` — reranking
reads their already-correct output and re-scores it; it never changes how
the fused candidate pool itself is produced.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.rrf import deterministic_hybrid_search
from mhrag.retrieval.schema import RetrievalResult

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Fixed per Phase 5 spec ("keep the existing top-20 candidate depth fixed
# so this is a controlled experiment" / "do NOT tune candidate depth in
# this phase") — not a knob meant to be swept.
RERANK_CANDIDATE_DEPTH = 20


class Reranker:
    """Local cross-encoder reranker. Reusable across queries — the model is
    loaded once at construction and every `score`/`rerank` call reuses it."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Score `(query, text)` pairs for every text in `texts`, batched
        via CrossEncoder's own `batch_size` handling. Returns cross-encoder
        relevance scores (higher = more relevant); the reranked ORDER is
        what matters here, not the score's absolute scale. Verified
        empirically: `sentence-transformers`' `CrossEncoder` applies
        `BAAI/bge-reranker-base`'s configured activation automatically, so
        scores come back sigmoid-bounded in ~[0,1] (e.g. 0.9999 for a
        clearly relevant pair, 0.00004 for a clearly irrelevant one) —
        not raw unbounded logits."""
        if not texts:
            return np.array([])
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=float)


def rerank_results(
    query: str,
    candidates: list[RetrievalResult],
    reranker: Reranker,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Rerank `candidates` (e.g. the fused Hybrid RRF top-20) with a
    cross-encoder.

    Preserves, per result: `chunk_id`, `doc_id`, and all other metadata
    (title/url/source/category/published_at/text/position) unchanged;
    `rrf_score` is set to the candidate's incoming `score` (its RRF score —
    candidates here are always Hybrid RRF output); `rerank_score`/`score`
    are set to the new cross-encoder score; `rank` is the new 1-based
    position after reranking; `method` becomes "hybrid_reranked".

    Does not mutate `candidates` — reads `.text`/`.rank`/`.chunk_id`/
    `.score` off each element and returns an entirely new list.

    Deterministic tie-break when reranker scores land on the exact same
    value (rare for a continuous cross-encoder score, but not impossible —
    e.g. two near-duplicate chunks): (1) reranker score descending, (2)
    original RRF rank ascending, (3) chunk_id ascending — the same
    escalating-specificity style as `mhrag.retrieval.rrf.rrf_fuse`'s
    tie-break.
    """
    if not candidates:
        return []

    texts = [c.text for c in candidates]
    scores = reranker.score(query, texts)

    scored = list(zip(candidates, (float(s) for s in scores)))
    scored.sort(key=lambda pair: (-pair[1], pair[0].rank, pair[0].chunk_id))

    if top_k is not None:
        scored = scored[:top_k]

    reranked: list[RetrievalResult] = []
    for new_rank, (original, rerank_score) in enumerate(scored, start=1):
        reranked.append(
            replace(
                original,
                rank=new_rank,
                score=rerank_score,
                method="hybrid_reranked",
                rrf_score=original.score,
                rerank_score=rerank_score,
            )
        )
    return reranked


def rerank_hybrid_search(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    reranker: Reranker,
    dense_top_k: int = RERANK_CANDIDATE_DEPTH,
    bm25_top_k: int = RERANK_CANDIDATE_DEPTH,
    final_top_k: int = 5,
) -> list[RetrievalResult]:
    """Full Phase 5 pipeline: Dense(top-20) + BM25(top-20) -> deterministic
    RRF(k=60) -> cross-encoder rerank -> top `final_top_k`.

    `dense_top_k`/`bm25_top_k` default to the fixed candidate depth
    (`RERANK_CANDIDATE_DEPTH = 20`) — the fused pool handed to the reranker
    is exactly this many chunks (RRF's own `final_top_k` is set to the same
    value here, so no candidates are dropped between fusion and reranking).
    Calls the existing, unmodified `deterministic_hybrid_search` for the
    fusion stage — this function only adds the reranking step on top.
    """
    fused_candidates = deterministic_hybrid_search(
        query,
        client,
        collection_name,
        embedding_model,
        bm25_model,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=RERANK_CANDIDATE_DEPTH,
    )
    return rerank_results(query, fused_candidates, reranker, top_k=final_top_k)
