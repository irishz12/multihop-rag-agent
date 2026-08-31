"""Hybrid retrieval: dense + BM25, fused server-side by Qdrant with RRF.

    query
     ├── dense search   (prefetch, top `dense_top_k`)
     └── BM25 search    (prefetch, top `bm25_top_k`)
            ↓
       Qdrant RRF fusion
            ↓
       top `final_top_k` ranked results

RRF configuration actually supported by the pinned client/server
(qdrant-client==1.19.x, qdrant/qdrant:v1.19.0) — verified empirically, not
assumed: `qdrant_client.http.models.FusionQuery` has exactly one field,
`fusion` (`Fusion.RRF` or `Fusion.DBSF`). There is no exposed RRF
weight-per-retriever or k-constant parameter in this client version — Qdrant
runs its own fixed, equal-weight RRF internally once given the prefetch
lists. This is a good match for the Phase 3 requirement to use "equal-weight
RRF... do NOT tune weights yet": there is currently nothing to tune even if
we wanted to. The two knobs that ARE exposed and used here are each
prefetch's `limit` (dense_top_k / bm25_top_k — how many candidates each
retriever contributes before fusion) and the outer `limit` (final_top_k —
how many fused results are returned).

KNOWN LIMITATION (discovered in Phase 4, running the real evaluation twice
back-to-back): repeated identical hybrid queries are not always
bit-identical at the tail of the ranking. Two independent 265-question
evaluation runs against the real 5,721-chunk collection produced identical
`dense_search`/`bm25_search` results in every case, but differed in
`hybrid_search`'s returned order for 73/265 (27.5%) queries — always a
swap among documents tied on RRF score (RRF scores are simple reciprocal-
rank sums, so exact ties are common, especially near the tail of a 50-deep
prefetch), never a change in which documents were in the fused set. This is
Qdrant server-side tie-breaking behavior, not something this module
controls, and its effect on aggregate metrics was small (3 of 9 metrics
moved by <0.4% between the two runs; every "which method wins" conclusion
was unchanged). Per the Phase 4 spec ("do NOT change... RRF implementation"
— this phase is measurement, not optimization), no fix (e.g. an
application-side deterministic secondary sort key) was applied here; this
is left as a known caveat for a future phase to address if it matters.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import BM25_VECTOR_NAME, DENSE_VECTOR_NAME
from mhrag.retrieval.schema import RetrievalResult


def hybrid_search(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    dense_top_k: int = 20,
    bm25_top_k: int = 20,
    final_top_k: int = 5,
) -> list[RetrievalResult]:
    query_vector = embedding_model.embed_query(query)
    sparse_query = bm25_model.embed_query(query)

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            qmodels.Prefetch(
                query=query_vector.tolist(), using=DENSE_VECTOR_NAME, limit=dense_top_k
            ),
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=sparse_query.indices, values=sparse_query.values
                ),
                using=BM25_VECTOR_NAME,
                limit=bm25_top_k,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=final_top_k,
        with_payload=True,
    )

    results: list[RetrievalResult] = []
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload
        results.append(
            RetrievalResult(
                rank=rank,
                score=point.score,
                method="hybrid",
                chunk_id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                title=payload["title"],
                url=payload["url"],
                source=payload["source"],
                category=payload["category"],
                published_at=payload["published_at"],
                text=payload["text"],
                position=payload["position"],
            )
        )
    return results
