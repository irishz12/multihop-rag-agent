"""Hybrid (dense + BM25 RRF) integration tests: real embedding model, real
BM25 model, and a live local Qdrant, using a small temporary collection —
never the real 5,721-chunk corpus collection.

Exercises the actual migration functions used by scripts/build_hybrid_index.py
(fetch_all_points, recreate_hybrid_collection, point_to_hybrid_point,
upsert_points, verify_hybrid_points), not a reimplementation, so these tests
validate the real migration path.

Skips gracefully if Qdrant isn't reachable, same pattern as
tests/test_retrieval_integration.py (which this file leaves untouched).
"""

from __future__ import annotations

import pytest
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.chunking import Chunk
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.hybrid import hybrid_search
from mhrag.retrieval.qdrant_store import (
    BM25_VECTOR_NAME,
    CollectionConfig,
    DENSE_VECTOR_NAME,
    HybridCollectionConfig,
    fetch_all_points,
    get_client,
    point_to_hybrid_point,
    recreate_collection,
    recreate_hybrid_collection,
    upsert_chunks,
    upsert_points,
    verify_hybrid_points,
)
from mhrag.retrieval.schema import RetrievalResult

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "mhrag_test_hybrid_retrieval"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BM25_MODEL_NAME = "Qdrant/bm25"

# Deliberately topically distinct, plus one doc with a rare, distinctive
# proper noun ("Zylquartz") to exercise BM25's exact-term / IDF strength on
# a query dense similarity alone might not rank first.
DOCS = [
    ("doc-space", "Space Exploration", "NASA launched a new rover to explore the surface of Mars."),
    ("doc-cooking", "Home Cooking", "This recipe uses fresh basil, garlic, and olive oil for a simple pasta sauce."),
    ("doc-finance", "Stock Market Update", "The central bank raised interest rates to control inflation this quarter."),
    ("doc-rare", "Obscure Product Launch", "The new device is called Zylquartz and it ships next month."),
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
def hybrid_collection(client, embedding_model, bm25_model):
    """Build a dense-only collection (mimicking Phase 2), record its dense
    vectors, then migrate it to dense+BM25 via the real migration functions
    — mirroring scripts/build_hybrid_index.py exactly."""
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

    # Snapshot dense vectors + count BEFORE migration, for the "unchanged"/
    # "all present" checks below.
    pre_migration_points = fetch_all_points(client, TEST_COLLECTION)
    pre_migration_dense = {p.id: p.vector[DENSE_VECTOR_NAME] for p in pre_migration_points}

    bm25_vectors = bm25_model.embed_passages([c.text for c in chunks])

    recreate_hybrid_collection(
        client,
        HybridCollectionConfig(name=TEST_COLLECTION, dense_vector_size=embedding_model.dimension),
    )
    hybrid_points = [
        point_to_hybrid_point(p, bm25_vectors[i]) for i, p in enumerate(pre_migration_points)
    ]
    upsert_points(client, TEST_COLLECTION, hybrid_points)

    return {
        "chunks": chunks,
        "pre_migration_count": len(pre_migration_points),
        "pre_migration_dense": pre_migration_dense,
    }


# --- migration correctness ---------------------------------------------------------


def test_bm25_sparse_vector_config_uses_idf(client, hybrid_collection):
    info = client.get_collection(TEST_COLLECTION)
    sparse_config = info.config.params.sparse_vectors[BM25_VECTOR_NAME]
    assert sparse_config.modifier == qmodels.Modifier.IDF


def test_all_indexed_chunks_remain_present_after_migration(client, hybrid_collection):
    info = client.get_collection(TEST_COLLECTION)
    assert info.points_count == hybrid_collection["pre_migration_count"] == len(DOCS)


def test_every_point_has_dense_and_sparse_vectors(client, hybrid_collection):
    report = verify_hybrid_points(client, TEST_COLLECTION, expected_count=len(DOCS))
    assert report == {"checked": len(DOCS), "missing_dense": 0, "missing_sparse": 0}


def test_dense_vectors_remain_unchanged_after_migration(client, hybrid_collection):
    post_migration_points = fetch_all_points(client, TEST_COLLECTION)
    post_migration_dense = {p.id: p.vector[DENSE_VECTOR_NAME] for p in post_migration_points}
    assert post_migration_dense == hybrid_collection["pre_migration_dense"]


# --- standalone BM25 -----------------------------------------------------------------


def test_standalone_bm25_returns_valid_ranked_results(client, bm25_model, hybrid_collection):
    results = bm25_search("pasta sauce recipe", client, TEST_COLLECTION, bm25_model, top_k=3)
    assert 1 <= len(results) <= 3
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.method == "bm25"
        assert r.chunk_id and r.doc_id


def test_bm25_finds_rare_term_document(client, bm25_model, hybrid_collection):
    """Exact-term / IDF strength: a query using the rare, distinctive term
    should surface doc-rare via lexical match."""
    results = bm25_search("Zylquartz", client, TEST_COLLECTION, bm25_model, top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "doc-rare"


# --- hybrid ------------------------------------------------------------------------


def test_hybrid_retrieval_returns_valid_ranked_results(
    client, embedding_model, bm25_model, hybrid_collection
):
    results = hybrid_search(
        "central bank interest rates",
        client,
        TEST_COLLECTION,
        embedding_model,
        bm25_model,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=3,
    )
    assert 1 <= len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.method == "hybrid"


def test_hybrid_top_result_is_topically_correct(
    client, embedding_model, bm25_model, hybrid_collection
):
    results = hybrid_search(
        "What ingredients go into a good pasta sauce?",
        client,
        TEST_COLLECTION,
        embedding_model,
        bm25_model,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=3,
    )
    assert results[0].doc_id == "doc-cooking"


def test_no_duplicate_chunk_ids_in_hybrid_results(
    client, embedding_model, bm25_model, hybrid_collection
):
    results = hybrid_search(
        "Mars rover mission",
        client,
        TEST_COLLECTION,
        embedding_model,
        bm25_model,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=4,
    )
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


# --- determinism ---------------------------------------------------------------------


def test_dense_search_is_deterministic_for_repeated_query(
    client, embedding_model, hybrid_collection
):
    a = dense_search("Mars rover mission", client, TEST_COLLECTION, embedding_model, top_k=3)
    b = dense_search("Mars rover mission", client, TEST_COLLECTION, embedding_model, top_k=3)
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    assert [r.score for r in a] == [r.score for r in b]


def test_bm25_search_is_deterministic_for_repeated_query(client, bm25_model, hybrid_collection):
    a = bm25_search("pasta sauce recipe", client, TEST_COLLECTION, bm25_model, top_k=3)
    b = bm25_search("pasta sauce recipe", client, TEST_COLLECTION, bm25_model, top_k=3)
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    assert [r.score for r in a] == [r.score for r in b]


def test_hybrid_search_is_deterministic_for_repeated_query(
    client, embedding_model, bm25_model, hybrid_collection
):
    a = hybrid_search(
        "interest rate policy", client, TEST_COLLECTION, embedding_model, bm25_model
    )
    b = hybrid_search(
        "interest rate policy", client, TEST_COLLECTION, embedding_model, bm25_model
    )
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    assert [r.score for r in a] == [r.score for r in b]


# --- metadata / document mapping ------------------------------------------------------


def test_metadata_and_document_mapping_intact_across_all_methods(
    client, embedding_model, bm25_model, hybrid_collection
):
    chunks_by_id = {c.chunk_id: c for c in hybrid_collection["chunks"]}

    dense_results = dense_search(
        "stock market interest rates", client, TEST_COLLECTION, embedding_model, top_k=4
    )
    bm25_results = bm25_search(
        "stock market interest rates", client, TEST_COLLECTION, bm25_model, top_k=4
    )
    hybrid_results = hybrid_search(
        "stock market interest rates", client, TEST_COLLECTION, embedding_model, bm25_model
    )

    for results in (dense_results, bm25_results, hybrid_results):
        for r in results:
            source_chunk = chunks_by_id[r.chunk_id]
            assert r.doc_id == source_chunk.doc_id
            assert r.title == source_chunk.title
            assert r.url == source_chunk.url
            assert r.source == source_chunk.source


def test_phase2_dense_search_signature_unchanged(client, embedding_model, hybrid_collection):
    """dense_search's call signature and RetrievalResult population must be
    unaffected by Phase 3 — same call, same shape (with the new `method`
    field now included)."""
    results = dense_search("Mars rover mission", client, TEST_COLLECTION, embedding_model, top_k=1)
    assert len(results) == 1
    r = results[0]
    assert r.method == "dense"
    assert r.rank == 1
