#!/usr/bin/env python
"""Extend the Phase 2 dense collection with a named BM25 sparse vector.

Qdrant's vector schema is fixed at collection-creation time (verified
empirically — see mhrag.retrieval.qdrant_store module docstring), so this
recreates the collection with both vector types declared, but reuses each
point's already-computed dense vector fetched straight off the existing
collection — the dense embedding model is never re-run.

Only the chunk `text` already stored in each point's payload (indexed in
Phase 2) is used to compute BM25 vectors — the raw corpus is not re-read or
re-chunked, per "use the same chunk text and metadata already indexed".

Usage:
    python scripts/build_hybrid_index.py [--retrieval-config configs/retrieval.yaml]
"""

from __future__ import annotations

import argparse
import time

from mhrag.config import load_config
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.retrieval.qdrant_store import (
    HybridCollectionConfig,
    fetch_all_points,
    get_client,
    point_to_hybrid_point,
    recreate_hybrid_collection,
    upsert_points,
    verify_hybrid_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    args = parser.parse_args()

    retrieval_config = load_config(args.retrieval_config)
    collection_name = retrieval_config["qdrant"]["collection_name"]
    client = get_client(retrieval_config["qdrant"]["url"])

    if not client.collection_exists(collection_name):
        raise SystemExit(
            f"Collection '{collection_name}' does not exist — run scripts/build_index.py first."
        )

    print(f"Fetching existing points (dense vectors + payload) from '{collection_name}' ...")
    t0 = time.monotonic()
    existing_points = fetch_all_points(client, collection_name)
    fetch_secs = time.monotonic() - t0
    print(f"Fetched {len(existing_points)} points in {fetch_secs:.1f}s")

    for p in existing_points:
        if "text" not in (p.payload or {}):
            raise SystemExit(f"Point {p.id} has no 'text' payload field — cannot compute BM25.")

    dense_vector_size = len(next(iter(existing_points)).vector["dense"])
    print(f"Dense vector size (reused, unchanged): {dense_vector_size}")

    print(f"\nLoading BM25 model {retrieval_config['bm25']['model_name']} ...")
    bm25_model = Bm25Model(model_name=retrieval_config["bm25"]["model_name"])

    print("Computing BM25 sparse vectors for existing chunk text ...")
    t0 = time.monotonic()
    texts = [p.payload["text"] for p in existing_points]
    sparse_vectors = bm25_model.embed_passages(texts)
    bm25_secs = time.monotonic() - t0
    print(f"Computed {len(sparse_vectors)} sparse vectors in {bm25_secs:.1f}s")

    print(f"\nRecreating collection '{collection_name}' with dense + BM25 (IDF) schema ...")
    recreate_hybrid_collection(
        client,
        HybridCollectionConfig(
            name=collection_name,
            dense_vector_size=dense_vector_size,
            dense_distance=retrieval_config["qdrant"]["distance"],
        ),
    )

    print("Re-upserting points: existing dense vector reused + new BM25 sparse vector ...")
    t0 = time.monotonic()
    hybrid_points = [
        point_to_hybrid_point(point, sparse_vectors[i]) for i, point in enumerate(existing_points)
    ]
    upsert_points(client, collection_name, hybrid_points)
    upsert_secs = time.monotonic() - t0
    print(f"Re-indexed {len(hybrid_points)} points in {upsert_secs:.1f}s")

    print("\nVerifying every point has both dense and BM25 representations ...")
    report = verify_hybrid_points(client, collection_name, expected_count=len(existing_points))
    print(f"Verified {report['checked']} points: "
          f"missing_dense={report['missing_dense']}, missing_sparse={report['missing_sparse']}")

    info = client.get_collection(collection_name)
    print("\nQdrant collection details:")
    print(f"  name:          {collection_name}")
    print(f"  points_count:  {info.points_count}")
    print(f"  status:        {info.status}")
    print(f"  dense config:  {info.config.params.vectors}")
    print(f"  sparse config: {info.config.params.sparse_vectors}")


if __name__ == "__main__":
    main()
