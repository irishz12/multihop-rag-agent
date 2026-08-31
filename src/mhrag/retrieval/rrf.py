"""Deterministic, application-side Reciprocal Rank Fusion (RRF).

This is now the CANONICAL hybrid retrieval path for the benchmark — see the
"correction" note below for why. It fuses two already-ranked chunk lists
(dense, BM25 — both unchanged, see `mhrag.retrieval.dense`/`mhrag.retrieval.bm25`)
with the textbook RRF formula:

    score(chunk) = Σ_{list ∈ {dense, bm25}} weight_list / (K + rank_list(chunk))

summed over every list the chunk appears in (a chunk missing from a list
contributes 0 for that list's term — not a penalty, not the max rank).

**Rank convention — 1-based, documented explicitly per the correction spec:**
`rank_list(chunk)` is the chunk's 1-based position within that list — the
top-ranked candidate has rank 1, the second has rank 2, etc. This is the
convention used in the original RRF paper (Cormack, Clarke & Büttcher,
"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning
Methods", SIGIR 2009) and in most RRF implementations outside Qdrant (e.g.
Elasticsearch's `rank_rrf`, LangChain's `EnsembleRetriever`). Concretely: a
chunk ranked #1 in a list contributes `1/(K+1)`, not `1/K`.

**K = 60**, the approved experimental baseline, and the constant most
commonly cited alongside "RRF" in IR literature and the implementations
above. **Weights are equal** (1.0 each) — the classic, unweighted RRF from
the original paper; not tunable here by design (Phase 3's "do NOT tune
weights yet" still applies).

**Deterministic tie-break**, applied whenever two chunks land on the exact
same fused score (this is common — RRF scores are sums of one or two simple
reciprocals, so exact ties happen often, and untreated ties were the exact
cause of the Phase 4 hybrid-search nondeterminism this module fixes):

    1. fused RRF score, descending  (higher score wins)
    2. best individual rank, ascending  (min(dense_rank, bm25_rank), missing
       list = +inf) — a chunk highly ranked by at least one method outranks
       one that was merely decent in both
    3. chunk_id, ascending (lexicographic)  — final, always-decisive
       tie-break; `chunk_id` is unique per chunk, so this guarantees a
       total order regardless of scores or ranks

---

**CORRECTION (Phase 4.1)** — discovered while investigating a report that
Phase 3/4 had used Qdrant's *default* RRF, not the approved k=60 baseline.
Verified directly against the installed `qdrant-client==1.19.x` /
`qdrant/qdrant:v1.19.0`, with controlled fixtures (not assumed):

- `qdrant_client.http.models` exposes an `RrfQuery`/`Rrf` pair — `Rrf` has
  `k: Optional[int]` and `weights: Optional[List[float]]` fields — DISTINCT
  from `FusionQuery` (which has only a `fusion` field: `Fusion.RRF` or
  `Fusion.DBSF`, no k/weights). Phase 3's `mhrag.retrieval.hybrid.hybrid_search`
  uses `FusionQuery(fusion=Fusion.RRF)`, i.e. Qdrant's SERVER-CHOSEN default,
  not a configurable k. This was missed in Phase 3's own verification (which
  found `FusionQuery` had no k/weight fields and stopped there without
  searching for a differently-named model that did).
- Confirmed empirically that `FusionQuery(fusion=Fusion.RRF)` is
  bit-identical, on identical prefetch candidates, to `RrfQuery(rrf=Rrf(k=2))`
  — i.e. Qdrant's default k really is 2, not 60.
- Confirmed empirically, with a fixture giving one chunk a known rank in
  each list, that Qdrant's OWN native RRF (`RrfQuery`) uses 0-BASED rank
  internally (a chunk ranked #1 in both lists at k=60 scored exactly
  `2/60`, matching the 0-based prediction `1/(60+0)+1/(60+0)`, not the
  1-based prediction `1/(60+1)+1/(60+1)`). So even calling `RrfQuery(rrf=
  Rrf(k=60))` would not reproduce this module's numbers exactly — the rank
  convention differs too, which is why this module states its own
  convention explicitly rather than silently inheriting Qdrant's.
- `mhrag.retrieval.hybrid.hybrid_search` is left completely unmodified and
  kept as an optional REFERENCE implementation (per the correction spec);
  the benchmark hybrid baseline now calls `deterministic_hybrid_search`
  (this module) exclusively.
"""

from __future__ import annotations

from dataclasses import replace

from qdrant_client import QdrantClient

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.schema import RetrievalResult

RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0


def _rank_map(results: list[RetrievalResult]) -> dict[str, int]:
    """1-based rank per chunk_id within one already rank-ordered list."""
    return {r.chunk_id: i + 1 for i, r in enumerate(results)}


def rrf_fuse(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    k: int = RRF_K,
    final_top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse two ranked chunk lists with deterministic RRF@k.

    Does not mutate or reorder `dense_results`/`bm25_results` — both are
    only read (via `_rank_map` and dict lookups), so the caller's original
    lists are unaffected and reusable afterward (e.g. for standalone
    dense/BM25 metrics on the very same retrieval call).

    Returns a new list of `RetrievalResult`, re-ranked and re-scored
    (`method="hybrid"`, `score`=fused RRF score, `rank`=1-based fused
    position), truncated to `final_top_k` if given. No chunk_id appears
    twice in the output — the fusion key is chunk_id, and each chunk_id
    contributes exactly one row.
    """
    dense_rank = _rank_map(dense_results)
    bm25_rank = _rank_map(bm25_results)

    chunk_by_id: dict[str, RetrievalResult] = {}
    for r in dense_results:
        chunk_by_id.setdefault(r.chunk_id, r)
    for r in bm25_results:
        chunk_by_id.setdefault(r.chunk_id, r)

    all_chunk_ids = set(dense_rank) | set(bm25_rank)

    scored: list[tuple[str, float, float]] = []  # (chunk_id, rrf_score, best_rank)
    for chunk_id in all_chunk_ids:
        score = 0.0
        best_rank = float("inf")
        if chunk_id in dense_rank:
            rank = dense_rank[chunk_id]
            score += DENSE_WEIGHT / (k + rank)
            best_rank = min(best_rank, rank)
        if chunk_id in bm25_rank:
            rank = bm25_rank[chunk_id]
            score += BM25_WEIGHT / (k + rank)
            best_rank = min(best_rank, rank)
        scored.append((chunk_id, score, best_rank))

    # Deterministic tie-break: score desc, best_rank asc, chunk_id asc.
    scored.sort(key=lambda t: (-t[1], t[2], t[0]))

    if final_top_k is not None:
        scored = scored[:final_top_k]

    fused: list[RetrievalResult] = []
    for rank, (chunk_id, score, _best_rank) in enumerate(scored, start=1):
        source = chunk_by_id[chunk_id]
        fused.append(replace(source, rank=rank, score=score, method="hybrid"))
    return fused


def deterministic_hybrid_search(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    dense_top_k: int = 20,
    bm25_top_k: int = 20,
    final_top_k: int = 5,
    k: int = RRF_K,
) -> list[RetrievalResult]:
    """Drop-in replacement for `mhrag.retrieval.hybrid.hybrid_search`, using
    deterministic application-side RRF (this module) instead of Qdrant's
    server-side fusion. Same call shape and same `dense_top_k`/`bm25_top_k`/
    `final_top_k` semantics — candidates fetched per method before fusion,
    and final fused results returned, respectively. Defaults
    (dense_top_k=20, bm25_top_k=20) match the production hybrid config in
    configs/retrieval.yaml, unchanged since Phase 3.

    Calls the existing, unmodified `dense_search`/`bm25_search` — this
    function changes only how their two already-correct candidate lists are
    fused, not how either candidate list is produced.
    """
    dense_results = dense_search(
        query, client, collection_name, embedding_model, top_k=dense_top_k
    )
    bm25_results = bm25_search(query, client, collection_name, bm25_model, top_k=bm25_top_k)
    return rrf_fuse(dense_results, bm25_results, k=k, final_top_k=final_top_k)
