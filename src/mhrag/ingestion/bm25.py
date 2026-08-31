"""BM25 sparse embedding wrapper around FastEmbed's `Qdrant/bm25` model.

`Qdrant/bm25` produces raw term-frequency sparse vectors (BM25's TF
saturation term — see https://qdrant.tech/documentation/fastembed/bm25/); it
does NOT bake in IDF (`list_supported_models()` reports
`requires_idf: True`). IDF weighting is applied by Qdrant itself at query
time via the sparse vector's `Modifier.IDF` (see
`mhrag.retrieval.qdrant_store.BM25_VECTOR_NAME`'s config), computed from
corpus-wide term statistics Qdrant tracks internally once indexed. This is
why passage and query embedding are two different methods with different
value schemes, not just an instruction-prefix difference like the dense
model: `passage_embed` weights terms by TF-saturation, `query_embed` weights
terms by simple presence — Qdrant's IDF modifier completes the BM25 score by
combining both with corpus IDF at query time.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastembed import SparseTextEmbedding

BM25_MODEL_NAME = "Qdrant/bm25"


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A generic sparse vector — term index -> weight — decoupled from any
    specific client library's type, mirroring how EmbeddingModel returns a
    plain np.ndarray rather than a sentence-transformers-specific type."""

    indices: list[int]
    values: list[float]


class Bm25Model:
    def __init__(self, model_name: str = BM25_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> list[SparseVector]:
        """Embed chunk/passage text — TF-saturation weights, no IDF."""
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.passage_embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Embed a search query — term-presence weights, no IDF."""
        embedding = next(iter(self._model.query_embed([text])))
        return SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )
