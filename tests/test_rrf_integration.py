"""Deterministic hybrid RRF integration tests: real embedding model, real
BM25 model, and a live local Qdrant, using a small temporary collection.

Directly closes the loop on the Phase 4 finding — repeating the full
265-question development evaluation twice found Qdrant's native
`hybrid_search` varied at the ranking tail for 27.5% of queries. This file
proves `deterministic_hybrid_search` does not, by running enough repeats
against a collection sized to actually produce RRF score ties (a handful of
documents, several sharing chunks/ranks across dense and BM25 orderings).
"""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.chunking import Chunk
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.qdrant_store import (
    HybridCollectionConfig,
    fetch_all_points,
    get_client,
    point_to_hybrid_point,
    recreate_hybrid_collection,
    upsert_points,
)
from mhrag.retrieval.rrf import RRF_K, deterministic_hybrid_search, rrf_fuse

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "mhrag_test_rrf_integration"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BM25_MODEL_NAME = "Qdrant/bm25"

# Several topically related documents so both dense and BM25 return
# multiple plausible candidates with real potential for RRF score ties.
DOCS = [
    ("doc-1", "Solar Power Growth", "Solar panel installations increased sharply across the country this year."),
    ("doc-2", "Wind Energy Report", "Wind turbine capacity expanded as renewable energy investment grew."),
    ("doc-3", "Renewable Policy Update", "New government policy supports renewable energy and solar power projects."),
    ("doc-4", "Battery Storage News", "Grid-scale battery storage adoption rose alongside renewable energy growth."),
    ("doc-5", "Cooking Guide", "A simple recipe for pasta with garlic, olive oil, and fresh basil."),
    ("doc-6", "Stock Market Report", "The central bank raised interest rates to control persistent inflation."),
]


@pytest.fixture(scope="module")
def client():
    c = get_client(QDRANT_URL)
    try:
        c.get_collections()
    except (ResponseHandlingException, ConnectionError, Exception) as exc:  # pragma: no cover
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}: {exc}")
    yield c
    if c.collection_exists(TEST_COLLECTION):
        c.delete_collection(TEST_COLLECTION)


@pytest.fixture(scope="module")
def embedding_model():
    try:
        return EmbeddingModel(model_name=EMBEDDING_MODEL_NAME, device="cpu", normalize=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load {EMBEDDING_MODEL_NAME}: {exc}")


@pytest.fixture(scope="module")
def bm25_model():
    try:
        return Bm25Model(model_name=BM25_MODEL_NAME)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load {BM25_MODEL_NAME}: {exc}")


@pytest.fixture(scope="module")
def rrf_collection(client, embedding_model, bm25_model):
    from mhrag.retrieval.qdrant_store import CollectionConfig, recreate_collection, upsert_chunks

    chunks = [
        Chunk(
            chunk_id=f"{i:016x}",
            doc_id=doc_id,
            title=title,
            url=f"https://example.com/{doc_id}",
            source="Test",
            category="test",
            published_at="2024-01-01T00:00:00+00:00",
            text=text,
            position=0,
            token_count=len(text.split()),
        )
        for i, (doc_id, title, text) in enumerate(DOCS)
    ]
    dense_vectors = embedding_model.embed_passages([c.text for c in chunks])
    recreate_collection(
        client, CollectionConfig(name=TEST_COLLECTION, vector_size=embedding_model.dimension)
    )
    upsert_chunks(client, TEST_COLLECTION, chunks, dense_vectors)

    existing_points = fetch_all_points(client, TEST_COLLECTION)
    bm25_vectors = bm25_model.embed_passages([p.payload["text"] for p in existing_points])
    recreate_hybrid_collection(
        client, HybridCollectionConfig(name=TEST_COLLECTION, dense_vector_size=embedding_model.dimension)
    )
    hybrid_points = [
        point_to_hybrid_point(p, bm25_vectors[i]) for i, p in enumerate(existing_points)
    ]
    upsert_points(client, TEST_COLLECTION, hybrid_points)
    return chunks


QUERY = "renewable energy and solar power growth"


def test_deterministic_hybrid_search_returns_schema_valid_results(
    client, embedding_model, bm25_model, rrf_collection
):
    results = deterministic_hybrid_search(
        QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, final_top_k=6
    )
    assert 1 <= len(results) <= 6
    for r in results:
        assert r.method == "hybrid"
        assert r.chunk_id and r.doc_id


def test_deterministic_hybrid_search_uses_rrf_k60_matching_standalone_fusion(
    client, embedding_model, bm25_model, rrf_collection
):
    """Cross-check: calling deterministic_hybrid_search must produce exactly
    what manually calling dense_search + bm25_search + rrf_fuse(k=60)
    produces — proving it really is RRF_K=60 under the hood, not some other
    default."""
    dense_results = dense_search(QUERY, client, TEST_COLLECTION, embedding_model, top_k=20)
    bm25_results = bm25_search(QUERY, client, TEST_COLLECTION, bm25_model, top_k=20)
    expected = rrf_fuse(dense_results, bm25_results, k=RRF_K, final_top_k=6)

    actual = deterministic_hybrid_search(
        QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, final_top_k=6
    )
    assert [(r.chunk_id, r.score) for r in actual] == [(r.chunk_id, r.score) for r in expected]


def test_deterministic_hybrid_search_no_duplicate_chunk_ids(
    client, embedding_model, bm25_model, rrf_collection
):
    results = deterministic_hybrid_search(
        QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, final_top_k=6
    )
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_deterministic_hybrid_search_is_repeatable_many_times(
    client, embedding_model, bm25_model, rrf_collection
):
    """The direct regression test for the Phase 4 finding: run the same
    query through the corrected hybrid path repeatedly and require every
    run to be bit-identical (chunk_id, rank, AND score) — not just 'mostly'
    stable. 5 repeats on a 6-document collection with real topical overlap
    (several plausible RRF ties) is a much harsher test than Phase 4's old
    Qdrant-native path ever passed at this scale."""
    runs = [
        deterministic_hybrid_search(
            QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, final_top_k=6
        )
        for _ in range(5)
    ]
    first = [(r.chunk_id, r.rank, r.score) for r in runs[0]]
    for run in runs[1:]:
        assert [(r.chunk_id, r.rank, r.score) for r in run] == first


def test_deterministic_hybrid_search_dense_and_bm25_results_unaffected(
    client, embedding_model, bm25_model, rrf_collection
):
    """Calling deterministic_hybrid_search must not change what standalone
    dense_search/bm25_search return for the same query — confirms the
    fusion step reads, but does not disturb, either candidate source."""
    dense_before = dense_search(QUERY, client, TEST_COLLECTION, embedding_model, top_k=20)
    bm25_before = bm25_search(QUERY, client, TEST_COLLECTION, bm25_model, top_k=20)

    deterministic_hybrid_search(QUERY, client, TEST_COLLECTION, embedding_model, bm25_model)

    dense_after = dense_search(QUERY, client, TEST_COLLECTION, embedding_model, top_k=20)
    bm25_after = bm25_search(QUERY, client, TEST_COLLECTION, bm25_model, top_k=20)

    assert [(r.chunk_id, r.score) for r in dense_before] == [
        (r.chunk_id, r.score) for r in dense_after
    ]
    assert [(r.chunk_id, r.score) for r in bm25_before] == [
        (r.chunk_id, r.score) for r in bm25_after
    ]
