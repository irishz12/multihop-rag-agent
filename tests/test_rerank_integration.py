"""Reranker integration tests: the real BAAI/bge-reranker-base model, real
embedding/BM25 models, and a live local Qdrant, using a small temporary
collection — never the real 5,721-chunk corpus collection.

Covers what tests/test_rerank.py's fake-scorer tests cannot: whether the
real model's batching is numerically consistent with single-item scoring,
and whether the full pipeline is deterministic end to end under repeated
identical queries.
"""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.chunking import Chunk
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import (
    CollectionConfig,
    HybridCollectionConfig,
    fetch_all_points,
    get_client,
    point_to_hybrid_point,
    recreate_collection,
    recreate_hybrid_collection,
    upsert_chunks,
    upsert_points,
)
from mhrag.retrieval.rerank import RERANK_CANDIDATE_DEPTH, Reranker, rerank_hybrid_search, rerank_results
from mhrag.retrieval.rrf import deterministic_hybrid_search

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "mhrag_test_rerank_integration"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BM25_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

DOCS = [
    ("doc-1", "Solar Power Growth", "Solar panel installations increased sharply across the country this year."),
    ("doc-2", "Wind Energy Report", "Wind turbine capacity expanded as renewable energy investment grew."),
    ("doc-3", "Renewable Policy Update", "New government policy supports renewable energy and solar power projects."),
    ("doc-4", "Battery Storage News", "Grid-scale battery storage adoption rose alongside renewable energy growth."),
    ("doc-5", "Cooking Guide", "A simple recipe for pasta with garlic, olive oil, and fresh basil."),
    ("doc-6", "Stock Market Report", "The central bank raised interest rates to control persistent inflation."),
]

QUERY = "renewable energy and solar power growth"


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
def reranker():
    try:
        return Reranker(model_name=RERANKER_MODEL_NAME, device="cpu", batch_size=32)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load {RERANKER_MODEL_NAME}: {exc}")


@pytest.fixture(scope="module")
def rerank_collection(client, embedding_model, bm25_model):
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


# --- deterministic reranker ordering / repeated identical query -----------------------


def test_rerank_hybrid_search_is_deterministic_across_repeated_calls(
    client, embedding_model, bm25_model, reranker, rerank_collection
):
    runs = [
        rerank_hybrid_search(
            QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, reranker, final_top_k=6
        )
        for _ in range(3)
    ]
    first = [(r.chunk_id, r.rank, r.score) for r in runs[0]]
    for run in runs[1:]:
        assert [(r.chunk_id, r.rank, r.score) for r in run] == first


def test_rerank_hybrid_search_returns_schema_valid_results(
    client, embedding_model, bm25_model, reranker, rerank_collection
):
    results = rerank_hybrid_search(
        QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, reranker, final_top_k=6
    )
    assert 1 <= len(results) <= 6
    for r in results:
        assert r.method == "hybrid_reranked"
        assert r.rrf_score is not None
        assert r.rerank_score is not None
        assert r.chunk_id and r.doc_id


def test_rerank_hybrid_search_top_result_is_topically_correct(
    client, embedding_model, bm25_model, reranker, rerank_collection
):
    results = rerank_hybrid_search(
        "What ingredients go into a good pasta sauce?",
        client,
        TEST_COLLECTION,
        embedding_model,
        bm25_model,
        reranker,
        final_top_k=3,
    )
    assert results[0].doc_id == "doc-5"


# --- candidate depth respected (fixed at RERANK_CANDIDATE_DEPTH) ----------------------


def test_rerank_hybrid_search_reranks_the_fixed_candidate_depth(
    client, embedding_model, bm25_model, reranker, rerank_collection
):
    """With only 6 documents in the collection, the fused pool (top-20) is
    smaller than RERANK_CANDIDATE_DEPTH — confirms the reranker receives
    every fused candidate (<=20, capped by what actually exists), not some
    other, silently-different number."""
    fused = deterministic_hybrid_search(
        QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, final_top_k=RERANK_CANDIDATE_DEPTH
    )
    reranked_all = rerank_hybrid_search(
        QUERY, client, TEST_COLLECTION, embedding_model, bm25_model, reranker, final_top_k=len(fused)
    )
    assert len(reranked_all) == len(fused) <= RERANK_CANDIDATE_DEPTH
    assert {r.chunk_id for r in reranked_all} == {r.chunk_id for r in fused}


# --- batching produces same result as single-item scoring -----------------------------


def test_batched_scoring_matches_single_item_scoring(reranker):
    texts = [text for _, _, text in DOCS]
    batched_scores = reranker.score(QUERY, texts)
    single_scores = [reranker.score(QUERY, [t])[0] for t in texts]

    assert len(batched_scores) == len(single_scores)
    for batched, single in zip(batched_scores, single_scores):
        assert batched == pytest.approx(single, abs=1e-4)

    # And the resulting ORDER (what actually matters downstream) must match exactly.
    batched_order = sorted(range(len(texts)), key=lambda i: -batched_scores[i])
    single_order = sorted(range(len(texts)), key=lambda i: -single_scores[i])
    assert batched_order == single_order


def test_rerank_results_batched_call_matches_looped_single_calls(reranker):
    """End-to-end version of the batching check, through rerank_results
    itself rather than calling .score() directly."""

    def _candidate(i, doc_id, text):
        from mhrag.retrieval.schema import RetrievalResult

        return RetrievalResult(
            rank=i + 1,
            score=1 / (60 + i + 1),
            method="hybrid",
            chunk_id=f"{i:016x}",
            doc_id=doc_id,
            title="t",
            url=f"https://example.com/{doc_id}",
            source="s",
            category="c",
            published_at="2024-01-01T00:00:00+00:00",
            text=text,
            position=0,
        )

    candidates = [_candidate(i, doc_id, text) for i, (doc_id, _, text) in enumerate(DOCS)]

    batched = rerank_results(QUERY, candidates, reranker)

    # Score one-by-one manually and re-derive the same ranking logic rerank_results uses.
    single_scores = {c.chunk_id: float(reranker.score(QUERY, [c.text])[0]) for c in candidates}
    expected_order = sorted(candidates, key=lambda c: (-single_scores[c.chunk_id], c.rank, c.chunk_id))

    assert [r.chunk_id for r in batched] == [c.chunk_id for c in expected_order]


# --- reranking latency is measurable and non-trivial -----------------------------------


def test_reranker_score_call_completes_and_returns_finite_scores(reranker):
    import math

    scores = reranker.score(QUERY, [text for _, _, text in DOCS])
    assert len(scores) == len(DOCS)
    assert all(math.isfinite(s) for s in scores)
