"""BM25 embedding wrapper tests: the real FastEmbed `Qdrant/bm25` model.

Skips gracefully if the model can't be loaded (offline environment), same
pattern as tests/test_embedding.py.
"""

from __future__ import annotations

import pytest

from mhrag.ingestion.bm25 import Bm25Model, SparseVector

MODEL_NAME = "Qdrant/bm25"


@pytest.fixture(scope="module")
def bm25_model() -> Bm25Model:
    try:
        return Bm25Model(model_name=MODEL_NAME)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")


def test_embed_passages_returns_sparse_vectors(bm25_model):
    vectors = bm25_model.embed_passages(["the quick brown fox", "a different sentence entirely"])
    assert len(vectors) == 2
    for v in vectors:
        assert isinstance(v, SparseVector)
        assert len(v.indices) == len(v.values)
        assert len(v.indices) > 0


def test_embed_query_returns_a_single_sparse_vector(bm25_model):
    v = bm25_model.embed_query("quick fox")
    assert isinstance(v, SparseVector)
    assert len(v.indices) == len(v.values)
    assert len(v.indices) > 0


def test_embed_passages_is_deterministic(bm25_model):
    a = bm25_model.embed_passages(["a repeatable sentence for testing"])
    b = bm25_model.embed_passages(["a repeatable sentence for testing"])
    assert a[0].indices == b[0].indices
    assert a[0].values == b[0].values


def test_embed_query_is_deterministic(bm25_model):
    a = bm25_model.embed_query("a repeatable query")
    b = bm25_model.embed_query("a repeatable query")
    assert a.indices == b.indices
    assert a.values == b.values


def test_query_and_passage_embeddings_use_different_value_schemes(bm25_model):
    """Per the model card (`requires_idf: True`), passage vectors carry
    TF-saturation weights while query vectors carry simple term-presence
    weights — so embedding the same short text as a query vs. a passage
    should not produce identical values (this is not a instruction-prefix
    difference like BGE's; it's two genuinely different weighting schemes)."""
    text = "a short repeatable phrase"
    query_vec = bm25_model.embed_query(text)
    passage_vec = bm25_model.embed_passages([text])[0]
    assert set(query_vec.indices) == set(passage_vec.indices), "same terms should appear in both"
    assert query_vec.values != passage_vec.values


def test_indices_are_distinct_within_a_single_vector(bm25_model):
    """No term should be hashed to the same sparse index twice within one
    embedding (would silently corrupt the sparse vector)."""
    v = bm25_model.embed_passages(["one two three four five six seven distinct words here"])[0]
    assert len(v.indices) == len(set(v.indices))
