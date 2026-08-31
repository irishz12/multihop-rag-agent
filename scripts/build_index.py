#!/usr/bin/env python
"""Build the dense Vector RAG index: chunk the corpus, embed every chunk,
and index it into Qdrant.

Only `corpus.json` is read. QA records (queries, ground-truth answers,
evidence) are never touched by this script.

Usage:
    python scripts/build_index.py [--config configs/dataset.yaml]
                                   [--retrieval-config configs/retrieval.yaml]
"""

from __future__ import annotations

import argparse
import statistics
import time

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.loader import load_corpus
from mhrag.ingestion.chunking import ChunkingConfig, chunk_corpus
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import CollectionConfig, get_client, recreate_collection, upsert_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    args = parser.parse_args()

    dataset_config = load_config(args.config)
    retrieval_config = load_config(args.retrieval_config)

    raw_dir = PROJECT_ROOT / dataset_config["paths"]["raw_dir"]
    corpus_path = raw_dir / dataset_config["source"]["corpus_file"]

    print(f"Loading corpus from {corpus_path} ...")
    documents = load_corpus(corpus_path)
    print(f"Corpus documents: {len(documents)}")

    print(f"\nLoading embedding model {retrieval_config['embedding']['model_name']} ...")
    embedding_model = EmbeddingModel(
        model_name=retrieval_config["embedding"]["model_name"],
        device=retrieval_config["embedding"].get("device"),
        normalize=retrieval_config["embedding"]["normalize"],
        query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
        batch_size=retrieval_config["embedding"]["batch_size"],
    )
    print(f"Embedding dimension: {embedding_model.dimension}")

    chunk_cfg = retrieval_config["chunking"]
    chunking_config = ChunkingConfig(
        target_tokens=chunk_cfg["target_tokens"],
        max_tokens=chunk_cfg["max_tokens"],
        overlap_tokens=chunk_cfg["overlap_tokens"],
    )
    count_tokens = embedding_model.build_token_counter()

    print("\nChunking corpus (paragraph-aware, per-document) ...")
    t0 = time.monotonic()
    chunks = chunk_corpus(documents, count_tokens, chunking_config)
    chunk_secs = time.monotonic() - t0
    print(f"Total chunks: {len(chunks)}  ({chunk_secs:.1f}s)")

    token_counts = [c.token_count for c in chunks]
    print("\nChunk-size statistics (tokens):")
    print(f"  min:    {min(token_counts)}")
    print(f"  max:    {max(token_counts)}")
    print(f"  mean:   {statistics.mean(token_counts):.1f}")
    print(f"  median: {statistics.median(token_counts)}")
    print(f"  stdev:  {statistics.stdev(token_counts):.1f}")

    per_doc = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    chunks_per_doc = list(per_doc.values())
    print("\nChunks per document:")
    print(f"  min:    {min(chunks_per_doc)}")
    print(f"  max:    {max(chunks_per_doc)}")
    print(f"  mean:   {statistics.mean(chunks_per_doc):.1f}")
    print(f"  docs represented: {len(per_doc)} / {len(documents)}")

    print("\nEmbedding chunks ...")
    t0 = time.monotonic()
    texts = [c.text for c in chunks]
    vectors = embedding_model.embed_passages(texts)
    embed_secs = time.monotonic() - t0
    print(f"Embedded {len(chunks)} chunks in {embed_secs:.1f}s -> shape {vectors.shape}")

    collection_name = retrieval_config["qdrant"]["collection_name"]
    print(f"\nRecreating Qdrant collection '{collection_name}' ...")
    client = get_client(retrieval_config["qdrant"]["url"])
    recreate_collection(
        client,
        CollectionConfig(
            name=collection_name,
            vector_size=embedding_model.dimension,
            distance=retrieval_config["qdrant"]["distance"],
        ),
    )

    print("Indexing chunks into Qdrant ...")
    t0 = time.monotonic()
    upsert_chunks(client, collection_name, chunks, vectors)
    index_secs = time.monotonic() - t0
    print(f"Indexed {len(chunks)} points in {index_secs:.1f}s")

    info = client.get_collection(collection_name)
    print("\nQdrant collection details:")
    print(f"  name:          {collection_name}")
    print(f"  points_count:  {info.points_count}")
    print(f"  status:        {info.status}")
    print(f"  vector config: {info.config.params.vectors}")


if __name__ == "__main__":
    main()
