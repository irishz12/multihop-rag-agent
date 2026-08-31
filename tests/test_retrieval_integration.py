"""Dense retrieval integration tests: real embedding model + a live local
Qdrant instance, using a small temporary collection (never the real
609-document corpus collection, so this stays fast and doesn't depend on
scripts/build_index.py having been run).

Skips gracefully if Qdrant isn't reachable at configs/retrieval.yaml's URL —
this suite is meant to run against `docker compose up -d` locally, not to
require Qdrant in every environment.
"""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from mhrag.ingestion.chunking import Chunk
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.qdrant_store import (
    CollectionConfig,
    get_client,
    recreate_collection,
    upsert_chunks,
)
from mhrag.retrieval.schema import RetrievalResult

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "mhrag_test_dense_retrieval"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

DOCS = [
    # (doc_id, title, text) — deliberately topically distinct so ranking is unambiguous.
    ("doc-space", "Space Exploration", "NASA launched a new rover to explore the surface of Mars."),
    ("doc-cooking", "Home Cooking", "This recipe uses fresh basil, garlic, and olive oil for a simple pasta sauce."),
    ("doc-finance", "Stock Market Update", "The central bank raised interest rates to control inflation this quarter."),
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
        return EmbeddingModel(model_name=MODEL_NAME, device="cpu", normalize=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")


@pytest.fixture(scope="module")
def indexed_collection(client, embedding_model):
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
    vectors = embedding_model.embed_passages([c.text for c in chunks])

    recreate_collection(
        client,
        CollectionConfig(name=TEST_COLLECTION, vector_size=embedding_model.dimension),
    )
    upsert_chunks(client, TEST_COLLECTION, chunks, vectors)
    return chunks


def test_dense_search_returns_schema_valid_results(client, embedding_model, indexed_collection):
    results = dense_search("Tell me about a mission to Mars", client, TEST_COLLECTION, embedding_model, top_k=3)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert isinstance(r.chunk_id, str) and r.chunk_id
        assert isinstance(r.doc_id, str) and r.doc_id
        assert isinstance(r.score, float)
        assert isinstance(r.rank, int)


def test_dense_search_ranks_are_sequential_from_one(client, embedding_model, indexed_collection):
    results = dense_search("basil and garlic pasta recipe", client, TEST_COLLECTION, embedding_model, top_k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_dense_search_scores_are_descending(client, embedding_model, indexed_collection):
    results = dense_search("interest rate policy and inflation", client, TEST_COLLECTION, embedding_model, top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_search_top_result_is_topically_correct(client, embedding_model, indexed_collection):
    """The clearest possible correctness check: a query about cooking must
    rank the cooking document first, not space or finance."""
    results = dense_search(
        "What ingredients go into a good pasta sauce?",
        client,
        TEST_COLLECTION,
        embedding_model,
        top_k=3,
    )
    assert results[0].doc_id == "doc-cooking"


def test_dense_search_top_k_of_one_returns_single_result(client, embedding_model, indexed_collection):
    results = dense_search("Mars rover mission", client, TEST_COLLECTION, embedding_model, top_k=1)
    assert len(results) == 1
    assert results[0].rank == 1


def test_retrieval_result_maps_back_to_source_document(client, embedding_model, indexed_collection):
    results = dense_search("central bank raises rates", client, TEST_COLLECTION, embedding_model, top_k=1)
    top = results[0]
    matching_doc = next(c for c in indexed_collection if c.chunk_id == top.chunk_id)
    assert top.doc_id == matching_doc.doc_id
    assert top.title == matching_doc.title
