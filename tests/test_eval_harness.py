"""Evaluation harness tests:
- structural guard against final_holdout access
- deterministic repeated evaluation, end-to-end against a live Qdrant
  collection (real embedding model + real BM25 model + real retrieval
  functions + real metric computation)
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from mhrag.data.schema import Evidence, QARecord, doc_id_from_url
from mhrag.eval.ground_truth import gold_doc_ids
from mhrag.eval.metrics import collapse_to_unique_documents, compute_all_metrics
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.chunking import Chunk
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.hybrid import hybrid_search
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

QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "mhrag_test_eval_harness"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BM25_MODEL_NAME = "Qdrant/bm25"

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_retrieval_eval.py"

DOCS = [
    ("doc-a", "Renewable Energy Report", "Solar and wind power capacity grew significantly this year."),
    ("doc-b", "Cooking Basics", "A simple guide to sautéing vegetables with olive oil."),
    ("doc-c", "Central Bank Policy", "The bank raised interest rates to fight inflation."),
]


# --- structural guard: no final_holdout access ------------------------------------


def _load_run_retrieval_eval_module():
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_split_file_constant_is_the_development_split():
    module = _load_run_retrieval_eval_module()
    assert module.DEV_SPLIT_FILE == "dev_subset.json"


def test_script_source_never_references_final_holdout_outside_documentation():
    """Documentation (docstrings and `#` comments) is allowed to explain the
    guarantee in prose (it names "final_holdout.json" to describe what's
    excluded); no actual code — no variable value, no path, no CLI choice —
    may reference it."""
    source = SCRIPT_PATH.read_text()
    without_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    without_comments = re.sub(r"#.*", "", without_docstrings)
    assert "final_holdout" not in without_comments


def test_script_has_no_cli_flag_that_can_select_a_different_split():
    source = SCRIPT_PATH.read_text()
    assert "--split" not in source, (
        "run_retrieval_eval.py must not expose a --split flag — the "
        "development split is a hardcoded constant, not a runtime choice"
    )


# --- deterministic repeated evaluation, end to end ---------------------------------


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
def eval_collection(client, embedding_model, bm25_model):
    # doc_id is the URL hash — matching production (CorpusDocument.doc_id /
    # chunk_document) exactly, since this fixture exists specifically to
    # test the URL-based ground-truth-to-index mapping end to end. Using
    # the raw label here (as e.g. test_hybrid_retrieval_integration.py's
    # DOCS fixture does, harmlessly, since that file never cross-references
    # evidence) would silently make gold_doc_ids() never match anything.
    chunks = [
        Chunk(
            chunk_id=f"{i:016x}",
            doc_id=doc_id_from_url(f"https://example.com/{label}"),
            title=title,
            url=f"https://example.com/{label}",
            source="Test",
            category="test",
            published_at="2024-01-01T00:00:00+00:00",
            text=text,
            position=0,
            token_count=len(text.split()),
        )
        for i, (label, title, text) in enumerate(DOCS)
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


def _synthetic_question() -> QARecord:
    """A question whose gold evidence is exactly doc-a and doc-c — its urls
    match two of the indexed test chunks above via the same url-hash scheme
    used for the real corpus."""
    return QARecord(
        query="How did renewable energy and central bank policy change this year?",
        answer="placeholder",
        question_type="comparison_query",
        evidence_list=(
            Evidence(
                title="Renewable Energy Report",
                author=None,
                url="https://example.com/doc-a",
                source="Test",
                category="test",
                published_at="2024-01-01T00:00:00+00:00",
                fact="Solar and wind power capacity grew.",
            ),
            Evidence(
                title="Central Bank Policy",
                author=None,
                url="https://example.com/doc-c",
                source="Test",
                category="test",
                published_at="2024-01-01T00:00:00+00:00",
                fact="The bank raised interest rates.",
            ),
        ),
    )


def _evaluate_once(client, embedding_model, bm25_model, record) -> dict:
    gold = gold_doc_ids(record)
    dense_results = dense_search(record.query, client, TEST_COLLECTION, embedding_model, top_k=10)
    hybrid_results = hybrid_search(
        record.query, client, TEST_COLLECTION, embedding_model, bm25_model
    )
    return {
        "dense": compute_all_metrics(collapse_to_unique_documents(dense_results), gold),
        "hybrid": compute_all_metrics(collapse_to_unique_documents(hybrid_results), gold),
    }


def test_repeated_evaluation_is_deterministic(client, embedding_model, bm25_model, eval_collection):
    """On this small (3-document) fixture, dense, bm25, and hybrid are all
    exactly reproducible. NOTE: at the real corpus's scale (5,721 chunks),
    two full 265-question evaluation runs found hybrid's RRF-fused order to
    vary at the tail for 27.5% of queries (Qdrant server-side tie-breaking
    among equal RRF scores — see mhrag.retrieval.hybrid module docstring),
    with a small (<0.4%) effect on 3 of 9 aggregate metrics. This test
    documents the property the code aims for; it does not reproduce that
    tie-prone scale."""
    record = _synthetic_question()
    first = _evaluate_once(client, embedding_model, bm25_model, record)
    second = _evaluate_once(client, embedding_model, bm25_model, record)
    assert first == second


def test_evaluation_correctly_scores_a_known_gold_set(
    client, embedding_model, bm25_model, eval_collection
):
    """End-to-end sanity: with only 3 indexed documents and 2 gold docs
    among them, both gold docs should be trivially findable within the
    small candidate pool, so Recall@10 and Complete-Evidence@10 should be
    1.0 for hybrid (the strongest of the two signals combined)."""
    record = _synthetic_question()
    result = _evaluate_once(client, embedding_model, bm25_model, record)
    assert result["hybrid"]["recall@10"] == pytest.approx(1.0)
    assert result["hybrid"]["complete_evidence@10"] == 1.0
