"""Standalone BM25 sparse retrieval: query -> BM25 sparse query -> Qdrant -> ranked chunks.

Only the query text is ever embedded or sent to Qdrant here — same
ground-truth-isolation contract as `mhrag.retrieval.dense.dense_search`.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.retrieval.qdrant_store import BM25_VECTOR_NAME
from mhrag.retrieval.schema import RetrievalResult


def bm25_search(
    query: str,
    client: QdrantClient,
    collection_name: str,
    bm25_model: Bm25Model,
    top_k: int = 5,
) -> list[RetrievalResult]:
    sparse_query = bm25_model.embed_query(query)
    response = client.query_points(
        collection_name=collection_name,
        query=qmodels.SparseVector(indices=sparse_query.indices, values=sparse_query.values),
        using=BM25_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    results: list[RetrievalResult] = []
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload
        results.append(
            RetrievalResult(
                rank=rank,
                score=point.score,
                method="bm25",
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
