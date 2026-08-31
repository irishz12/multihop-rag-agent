"""Dense top-k retrieval: query -> embedding -> Qdrant -> ranked chunks.

Only the query text is ever embedded or sent to Qdrant here. Ground-truth
`answer`/`evidence_list` fields (when a caller has a QARecord in hand) are
never passed into this function — see scripts/retrieval_sanity_check.py,
which uses them only for console display, after retrieval has already run.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import DENSE_VECTOR_NAME
from mhrag.retrieval.schema import RetrievalResult


def dense_search(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
) -> list[RetrievalResult]:
    query_vector = embedding_model.embed_query(query)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        using=DENSE_VECTOR_NAME,
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
                method="dense",
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
