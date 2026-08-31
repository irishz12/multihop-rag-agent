"""Qdrant collection setup and indexing.

One collection, holding a named dense vector ("dense") and, since Phase 3, a
named BM25 sparse vector ("bm25") — named vectors (rather than the
collection's anonymous default vector) are what let both live on the same
points instead of needing two collections to keep in sync.

Indexing always recreates the collection from scratch (delete-if-exists,
then create) rather than incrementally upserting into whatever is already
there — that's what makes "build the index" deterministic: same corpus +
same config in, same collection contents out, regardless of what was
indexed before.

CORRECTION (Phase 3): the Phase 2 version of this docstring claimed a named
vector would let "a later phase add a named sparse/BM25 vector to the same
collection and points, without a schema migration". That turned out to be
wrong — verified empirically against the pinned client/server
(qdrant-client==1.19.x, qdrant/qdrant:v1.19.0): a collection's set of named
vectors is fixed at creation time. `update_collection(sparse_vectors_config=
{...})` can only tune params (e.g. the IDF modifier, HNSW config) of a
sparse vector that's already in the schema; asked to add a brand-new name,
it returns `400 Wrong input: Not existing vector name error: <name>`. There
is no in-place "add a vector to an existing collection" operation in this
version. See `recreate_hybrid_collection` and `scripts/build_hybrid_index.py`
for how Phase 3 actually extends the Phase 2 collection: the schema is
recreated, but each point's already-computed dense vector is fetched
straight off the existing collection and reused unchanged — it is the
collection recreation (cheap, seconds) that's repeated, not the expensive
part (re-running the dense embedding model).
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from mhrag.ingestion.bm25 import SparseVector
from mhrag.ingestion.chunking import Chunk

DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"

REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "chunk_id",
        "doc_id",
        "title",
        "url",
        "source",
        "category",
        "published_at",
        "text",
        "position",
        "token_count",
    }
)


@dataclass(frozen=True)
class CollectionConfig:
    name: str
    vector_size: int
    distance: str = "Cosine"


def get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def chunk_id_to_point_id(chunk_id: str) -> int:
    """Qdrant point ids must be an unsigned int or UUID; chunk_id is a
    16-hex-char sha1 prefix (64 bits), so it converts losslessly to an int.
    The string chunk_id itself is also stored in the payload, so callers
    never need to reverse this conversion."""
    return int(chunk_id, 16)


def recreate_collection(client: QdrantClient, config: CollectionConfig) -> None:
    if client.collection_exists(config.name):
        client.delete_collection(config.name)
    distance = qmodels.Distance[config.distance.upper()]
    client.create_collection(
        collection_name=config.name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(size=config.vector_size, distance=distance)
        },
    )


def chunk_to_point(chunk: Chunk, vector: list[float]) -> qmodels.PointStruct:
    payload = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "url": chunk.url,
        "source": chunk.source,
        "category": chunk.category,
        "published_at": chunk.published_at,
        "text": chunk.text,
        "position": chunk.position,
        "token_count": chunk.token_count,
    }
    return qmodels.PointStruct(
        id=chunk_id_to_point_id(chunk.chunk_id),
        vector={DENSE_VECTOR_NAME: vector},
        payload=payload,
    )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    vectors,  # np.ndarray, shape (len(chunks), vector_size)
    batch_size: int = 128,
) -> None:
    points = [chunk_to_point(chunk, vectors[i].tolist()) for i, chunk in enumerate(chunks)]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])


# --- Phase 3: hybrid (dense + BM25) collection support --------------------------


@dataclass(frozen=True)
class HybridCollectionConfig:
    name: str
    dense_vector_size: int
    dense_distance: str = "Cosine"


def recreate_hybrid_collection(client: QdrantClient, config: HybridCollectionConfig) -> None:
    """Recreate the collection with BOTH the dense vector and a named BM25
    sparse vector (IDF modifier) declared in the schema from the start —
    see the module docstring for why this can't be done in-place on an
    existing collection."""
    if client.collection_exists(config.name):
        client.delete_collection(config.name)
    distance = qmodels.Distance[config.dense_distance.upper()]
    client.create_collection(
        collection_name=config.name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(size=config.dense_vector_size, distance=distance)
        },
        sparse_vectors_config={
            BM25_VECTOR_NAME: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
        },
    )


def fetch_all_points(
    client: QdrantClient, collection_name: str, batch_size: int = 256
) -> list[qmodels.Record]:
    """Scroll through every point in a collection, including vectors and
    payload. Used to carry a collection's already-computed dense vectors
    into a recreated collection without recomputing them."""
    points: list[qmodels.Record] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def point_to_hybrid_point(
    point: qmodels.Record, sparse_vector: SparseVector
) -> qmodels.PointStruct:
    """Rebuild a point for the hybrid collection: reuse its existing dense
    vector unchanged, add the new BM25 sparse vector, keep the payload as-is."""
    dense_vector = point.vector[DENSE_VECTOR_NAME]
    return qmodels.PointStruct(
        id=point.id,
        vector={
            DENSE_VECTOR_NAME: dense_vector,
            BM25_VECTOR_NAME: qmodels.SparseVector(
                indices=sparse_vector.indices, values=sparse_vector.values
            ),
        },
        payload=point.payload,
    )


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: list[qmodels.PointStruct],
    batch_size: int = 128,
) -> None:
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])


def verify_hybrid_points(
    client: QdrantClient, collection_name: str, expected_count: int
) -> dict:
    """Verify every point in the collection has both a dense and a BM25
    sparse vector, and the total count matches `expected_count`.

    Raises AssertionError on any mismatch; returns a small report dict
    otherwise (checked count and the two mismatch counts, both 0).
    """
    info = client.get_collection(collection_name)
    if info.points_count != expected_count:
        raise AssertionError(
            f"collection '{collection_name}' has {info.points_count} points, expected {expected_count}"
        )

    missing_dense = 0
    missing_sparse = 0
    checked = 0
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=True,
        )
        for point in batch:
            checked += 1
            vector = point.vector or {}
            if DENSE_VECTOR_NAME not in vector:
                missing_dense += 1
            if BM25_VECTOR_NAME not in vector:
                missing_sparse += 1
        if offset is None:
            break

    if missing_dense or missing_sparse:
        raise AssertionError(
            f"{missing_dense} points missing '{DENSE_VECTOR_NAME}', "
            f"{missing_sparse} missing '{BM25_VECTOR_NAME}' (of {checked} checked)"
        )
    return {"checked": checked, "missing_dense": missing_dense, "missing_sparse": missing_sparse}
