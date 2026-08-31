"""Embedding model tests: dimensionality and normalization, against the real
BAAI/bge-base-en-v1.5 model (configs/retrieval.yaml).

Requires the model weights to be available locally or downloadable (first
run caches to ~/.cache/huggingface). Skips gracefully if the model can't be
loaded, rather than failing the whole suite in an offline environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from mhrag.ingestion.embedding import EmbeddingModel

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EXPECTED_DIMENSION = 768


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    try:
        return EmbeddingModel(
            model_name=MODEL_NAME,
            device="cpu",
            normalize=True,
            query_instruction="Represent this sentence for searching relevant passages: ",
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")


def test_embedding_dimension_matches_model_card(embedding_model):
    assert embedding_model.dimension == EXPECTED_DIMENSION


def test_passage_embedding_shape_and_dimension(embedding_model):
    vectors = embedding_model.embed_passages(["a short passage", "another passage of text"])
    assert vectors.shape == (2, EXPECTED_DIMENSION)


def test_passage_embeddings_are_normalized(embedding_model):
    vectors = embedding_model.embed_passages(["some passage text", "a different passage"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_query_embedding_is_normalized_and_correct_dimension(embedding_model):
    vec = embedding_model.embed_query("Who founded the company?")
    assert vec.shape == (EXPECTED_DIMENSION,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_query_instruction_prefix_changes_the_embedding(embedding_model):
    """embed_query must actually apply the instruction prefix (not silently
    embed the raw query) — verified by confirming it differs from embedding
    the same text as a passage (no prefix)."""
    text = "What year was the company founded?"
    query_vec = embedding_model.embed_query(text)
    passage_vec = embedding_model.embed_passages([text])[0]
    assert not np.allclose(query_vec, passage_vec)


def test_token_counter_matches_model_tokenizer(embedding_model):
    count_tokens = embedding_model.build_token_counter()
    assert count_tokens("hello world") == 2
    assert count_tokens("") == 0
    assert count_tokens("supercalifragilisticexpialidocious") >= 1
