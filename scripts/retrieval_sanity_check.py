#!/usr/bin/env python
"""Manual retrieval sanity checks: dense vs BM25 vs Hybrid RRF vs Hybrid+Reranker.

Only ever reads from the `smoke` or `development` benchmark splits — the
`--split` flag intentionally has no `final_holdout` choice, so this script
cannot be used, even by accident, to inspect or tune against the held-out
evaluation set.

Ground-truth `answer`/`evidence_list` fields are read only to print a
"[has evidence]" label alongside each query in this script's console
output — they are never passed to any embedding model, retriever, or the
reranker (see mhrag.retrieval.{dense,bm25,rrf,rerank}, which only ever
take `query` text).

`--method hybrid` uses the canonical deterministic RRF@k=60
(mhrag.retrieval.rrf.deterministic_hybrid_search, since Phase 4.1) — NOT
mhrag.retrieval.hybrid.hybrid_search (Qdrant-native, kept only as a
reference implementation elsewhere in this codebase).

Usage:
    python scripts/retrieval_sanity_check.py --split smoke --top-k 5 --n 3
    python scripts/retrieval_sanity_check.py --split smoke --method hybrid
    python scripts/retrieval_sanity_check.py --split smoke --method hybrid_reranker
"""

from __future__ import annotations

import argparse
import json

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker, rerank_hybrid_search
from mhrag.retrieval.rrf import RRF_K, deterministic_hybrid_search

# final_holdout is intentionally NOT in this mapping — see module docstring.
SPLIT_FILES = {
    "smoke": "smoke_subset.json",
    "development": "dev_subset.json",
}

METHODS = ["dense", "bm25", "hybrid", "hybrid_reranker"]


def _print_results(label: str, results, top_k: int) -> None:
    print(f"  -- {label} --")
    for r in results[:top_k]:
        extra = ""
        if r.rrf_score is not None:
            extra = f"  (rrf_score={r.rrf_score:.4f}, rerank_score={r.rerank_score:.4f})"
        print(f"    #{r.rank}  score={r.score:.4f}  {r.title!r}  ({r.source}){extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="smoke")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n", type=int, default=3, help="number of queries to sample")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    args = parser.parse_args()

    dataset_config = load_config(args.config)
    retrieval_config = load_config(args.retrieval_config)

    split_path = (
        PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / SPLIT_FILES[args.split]
    )
    records = json.loads(split_path.read_text())[: args.n]

    methods = METHODS if args.method == "all" else [args.method]
    needs_dense = any(m in methods for m in ("dense", "hybrid", "hybrid_reranker"))
    needs_bm25 = any(m in methods for m in ("bm25", "hybrid", "hybrid_reranker"))
    needs_reranker = "hybrid_reranker" in methods

    embedding_model = None
    bm25_model = None
    reranker = None
    if needs_dense:
        print(f"Loading embedding model {retrieval_config['embedding']['model_name']} ...")
        embedding_model = EmbeddingModel(
            model_name=retrieval_config["embedding"]["model_name"],
            device=retrieval_config["embedding"].get("device"),
            normalize=retrieval_config["embedding"]["normalize"],
            query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
            batch_size=retrieval_config["embedding"]["batch_size"],
        )
    if needs_bm25:
        print(f"Loading BM25 model {retrieval_config['bm25']['model_name']} ...")
        bm25_model = Bm25Model(model_name=retrieval_config["bm25"]["model_name"])
    if needs_reranker:
        print(f"Loading reranker model {retrieval_config['reranker']['model_name']} ...")
        reranker = Reranker(
            model_name=retrieval_config["reranker"]["model_name"],
            device=retrieval_config["reranker"].get("device"),
            batch_size=retrieval_config["reranker"]["batch_size"],
        )

    client = get_client(retrieval_config["qdrant"]["url"])
    collection_name = retrieval_config["qdrant"]["collection_name"]
    hybrid_cfg = retrieval_config["hybrid"]

    for rec in records:
        has_evidence = "[has evidence]" if rec["evidence_list"] else "[no evidence expected]"
        print(f"\nQuery [{rec['question_type']}] {has_evidence}: {rec['query']}")

        if "dense" in methods:
            results = dense_search(
                rec["query"], client, collection_name, embedding_model, top_k=args.top_k
            )
            _print_results("dense", results, args.top_k)

        if "bm25" in methods:
            results = bm25_search(
                rec["query"], client, collection_name, bm25_model, top_k=args.top_k
            )
            _print_results("bm25", results, args.top_k)

        if "hybrid" in methods:
            results = deterministic_hybrid_search(
                rec["query"],
                client,
                collection_name,
                embedding_model,
                bm25_model,
                dense_top_k=hybrid_cfg["dense_top_k"],
                bm25_top_k=hybrid_cfg["bm25_top_k"],
                final_top_k=args.top_k,
                k=RRF_K,
            )
            _print_results("hybrid RRF (k=60, deterministic)", results, args.top_k)

        if "hybrid_reranker" in methods:
            results = rerank_hybrid_search(
                rec["query"],
                client,
                collection_name,
                embedding_model,
                bm25_model,
                reranker,
                final_top_k=args.top_k,
            )
            _print_results("hybrid + reranker (bge-reranker-base)", results, args.top_k)


if __name__ == "__main__":
    main()
