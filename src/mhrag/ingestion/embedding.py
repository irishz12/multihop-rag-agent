"""Embedding model wrapper around sentence-transformers.

Passages (chunk text) and queries are embedded differently, per BGE's
documented usage: queries get an instruction prefix, passages do not. Both
paths return L2-normalized vectors when `normalize=True`, which is required
for cosine similarity in Qdrant to behave as expected.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from mhrag.ingestion.chunking import TokenCounter


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        normalize: bool = True,
        query_instruction: str = "",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed chunk/passage text — no instruction prefix."""
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query — BGE instruction prefix applied, per model card."""
        prefixed = f"{self.query_instruction}{text}" if self.query_instruction else text
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0]

    def build_token_counter(self) -> TokenCounter:
        """Token counter backed by this model's own tokenizer, for chunking
        chunk sizes that are meaningful relative to what actually gets
        embedded (and to respect the model's input length limits)."""
        tokenizer = self._model.tokenizer

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        return count_tokens
