#!/usr/bin/env python
"""Retrieval evaluation harness: dense vs BM25 vs hybrid vs hybrid+reranker,
scored against the DEVELOPMENT benchmark split only.

Reads ONLY data/processed/dev_subset.json — this module has no code path
that can reach final_holdout.json (no CLI flag, no config option; the split
file is a hardcoded module constant). smoke_subset.json is likewise never
read here — it remains available for manual debugging via
scripts/retrieval_sanity_check.py, not for scored evaluation.

`evidence_list` (ground truth) is used only via mhrag.eval.ground_truth,
AFTER retrieval has already returned results for a query's `query` text —
never passed to any embedding model, retriever, or the reranker.

null_query questions (no gold evidence) are counted and reported separately;
no retrieval ground truth is invented for them, and they are excluded from
every metric.

CORRECTION (Phase 4.1): the "hybrid" method calls
`mhrag.retrieval.rrf.deterministic_hybrid_search` (application-side RRF@k=60,
1-based rank, explicit deterministic tie-break) instead of the original
`mhrag.retrieval.hybrid.hybrid_search` (Qdrant server-side fusion, which
defaulted to k=2 and had nondeterministic tie-breaking). See
`mhrag.retrieval.rrf` module docstring.

PHASE 5: adds a 4th method, "hybrid_reranker" — the same deterministic
Hybrid RRF pipeline, cross-encoder reranked
(`mhrag.retrieval.rerank.rerank_hybrid_search`, BAAI/bge-reranker-base, run
locally). Its candidate depth is FIXED at
`mhrag.retrieval.rerank.RERANK_CANDIDATE_DEPTH` (20) per the Phase 5 spec
("keep the existing top-20 candidate depth fixed... do NOT tune candidate
depth in this phase") — NOT the same as the other three methods'
evaluation-only `RAW_CANDIDATE_POOL_SIZE` (50, unchanged from Phase 4).
This is an intentional asymmetry, not an oversight: hybrid_reranker's
Recall@10/Hit@10/NDCG@10/Complete-Evidence@10 have structurally less
candidate headroom (up to 20 chunks, vs. 50 for the other three) — flagged
explicitly in the artifact's config and in the printed headroom check
below, so it isn't mistaken for an apples-to-apples pool size.

Retrieval and reranking latency are measured separately, per query, for
the hybrid and hybrid_reranker methods (P50/P95 reported at the end). No
LLM cost is estimated — no Bedrock Mantle calls exist in this phase.

Usage:
    python scripts/run_retrieval_eval.py [--output results/retrieval_eval_development.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import qa_id
from mhrag.data.loader import load_qa_records
from mhrag.eval.ground_truth import gold_doc_ids, hop_count
from mhrag.eval.metrics import collapse_to_unique_documents, compute_all_metrics, mean_metrics
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import RERANK_CANDIDATE_DEPTH, Reranker, rerank_results
from mhrag.retrieval.rrf import RRF_K, deterministic_hybrid_search

# Hardcoded to the development split ONLY — no CLI flag, no config option,
# so there is no code path in this script that can reach final_holdout.json.
DEV_SPLIT_FILE = "dev_subset.json"

METHODS = ["dense", "bm25", "hybrid", "hybrid_reranker"]

# Raw chunk/candidate pool size used ONLY by this evaluation harness, for
# dense/bm25/hybrid (NOT hybrid_reranker — see module docstring). Applied
# identically to those three methods for fairness. Chosen for headroom —
# large enough that collapsing to unique documents still reliably yields
# >= 10 unique docs to score Recall@10/Hit@10/NDCG@10/Complete-Evidence@10 —
# NOT tuned against results, and distinct from any method's production
# default top_k (retrieval.default_top_k=5, hybrid.final_top_k=5 in
# configs/retrieval.yaml — both untouched).
RAW_CANDIDATE_POOL_SIZE = 50

QUESTION_TYPES_EVALUATED = ["inference_query", "comparison_query", "temporal_query"]
HOP_COUNTS_EVALUATED = [2, 3, 4]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p=0.5 -> P50/median, p=0.95 -> P95) — sort,
    take the value at ceil(p * n) - 1. Simple, standard, no interpolation
    ambiguity to document."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    idx = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[idx]


def aggregate(entries: list[dict], method: str) -> dict:
    return mean_metrics([e["methods"][method]["metrics"] for e in entries])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--output", default="results/retrieval_eval_development.json")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index into the non-null question list (infra-only chunking "
        "for slow/constrained environments; does not change methodology — "
        "default 0 evaluates from the start).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of non-null questions to evaluate, starting at --offset "
        "(infra-only chunking; default None evaluates all remaining). A "
        "partial run (offset>0 or limit set) skips the manifest completeness "
        "check and writes only the evaluated slice — chunks are meant to be "
        "merged afterward, not treated as a final artifact on their own.",
    )
    args = parser.parse_args()

    dataset_config = load_config(args.config)
    retrieval_config = load_config(args.retrieval_config)

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    records = load_qa_records(dev_path)
    print(f"Loaded {len(records)} development questions from {dev_path}")

    null_records = [r for r in records if r.question_type == "null_query"]
    non_null_records_full = [r for r in records if r.question_type != "null_query"]
    print(f"  non-null (evaluated): {len(non_null_records_full)}")
    print(f"  null_query (reported separately, no metrics): {len(null_records)}")

    is_partial_run = args.offset != 0 or args.limit is not None
    end = None if args.limit is None else args.offset + args.limit
    non_null_records = non_null_records_full[args.offset : end]
    if is_partial_run:
        print(
            f"  PARTIAL RUN: evaluating records [{args.offset}:{end if end is not None else len(non_null_records_full)}] "
            f"({len(non_null_records)} of {len(non_null_records_full)}) — chunk for later merging"
        )

    print(f"\nLoading embedding model {retrieval_config['embedding']['model_name']} ...")
    embedding_model = EmbeddingModel(
        model_name=retrieval_config["embedding"]["model_name"],
        device=retrieval_config["embedding"].get("device"),
        normalize=retrieval_config["embedding"]["normalize"],
        query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
        batch_size=retrieval_config["embedding"]["batch_size"],
    )
    print(f"Loading BM25 model {retrieval_config['bm25']['model_name']} ...")
    bm25_model = Bm25Model(model_name=retrieval_config["bm25"]["model_name"])
    print(f"Loading reranker model {retrieval_config['reranker']['model_name']} ...")
    reranker = Reranker(
        model_name=retrieval_config["reranker"]["model_name"],
        device=retrieval_config["reranker"].get("device"),
        batch_size=retrieval_config["reranker"]["batch_size"],
    )

    client = get_client(retrieval_config["qdrant"]["url"])
    collection_name = retrieval_config["qdrant"]["collection_name"]
    collection_info = client.get_collection(collection_name)

    per_query = []
    latencies_ms: dict[str, list[float]] = {
        "hybrid_retrieval": [],
        "hybrid_reranker_retrieval": [],
        "hybrid_reranker_rerank": [],
        "hybrid_reranker_total": [],
    }

    t0 = time.monotonic()
    for i, record in enumerate(non_null_records):
        gold = gold_doc_ids(record)
        entry = {
            "qa_id": qa_id(record),
            "question_type": record.question_type,
            "hop_count": hop_count(record),
            "num_gold_docs": len(gold),
            "gold_doc_ids": sorted(gold),
            "methods": {},
        }

        def _record(method: str, results) -> None:
            unique_doc_ids = collapse_to_unique_documents(results)
            entry["methods"][method] = {
                "metrics": compute_all_metrics(unique_doc_ids, gold),
                "unique_doc_ids_top10": unique_doc_ids[:10],
                "num_unique_docs_retrieved": len(unique_doc_ids),
            }

        # dense
        _record(
            "dense",
            dense_search(
                record.query, client, collection_name, embedding_model, top_k=RAW_CANDIDATE_POOL_SIZE
            ),
        )

        # bm25
        _record(
            "bm25",
            bm25_search(record.query, client, collection_name, bm25_model, top_k=RAW_CANDIDATE_POOL_SIZE),
        )

        # hybrid (deterministic RRF@k=60, eval pool=50 — unchanged from Phase 4/4.1)
        t_start = time.perf_counter()
        hybrid_results = deterministic_hybrid_search(
            record.query,
            client,
            collection_name,
            embedding_model,
            bm25_model,
            dense_top_k=RAW_CANDIDATE_POOL_SIZE,
            bm25_top_k=RAW_CANDIDATE_POOL_SIZE,
            final_top_k=RAW_CANDIDATE_POOL_SIZE,
            k=RRF_K,
        )
        latencies_ms["hybrid_retrieval"].append((time.perf_counter() - t_start) * 1000)
        _record("hybrid", hybrid_results)

        # hybrid_reranker (fixed candidate depth=20, per Phase 5 spec)
        t_start = time.perf_counter()
        fused_20 = deterministic_hybrid_search(
            record.query,
            client,
            collection_name,
            embedding_model,
            bm25_model,
            dense_top_k=RERANK_CANDIDATE_DEPTH,
            bm25_top_k=RERANK_CANDIDATE_DEPTH,
            final_top_k=RERANK_CANDIDATE_DEPTH,
            k=RRF_K,
        )
        retrieval_latency = (time.perf_counter() - t_start) * 1000
        t_start = time.perf_counter()
        reranked = rerank_results(record.query, fused_20, reranker, top_k=RERANK_CANDIDATE_DEPTH)
        rerank_latency = (time.perf_counter() - t_start) * 1000
        latencies_ms["hybrid_reranker_retrieval"].append(retrieval_latency)
        latencies_ms["hybrid_reranker_rerank"].append(rerank_latency)
        latencies_ms["hybrid_reranker_total"].append(retrieval_latency + rerank_latency)
        _record("hybrid_reranker", reranked)

        per_query.append(entry)
        if (i + 1) % 10 == 0:
            elapsed = time.monotonic() - t0
            print(f"  evaluated {i + 1}/{len(non_null_records)} ({elapsed:.0f}s elapsed)", flush=True)

    elapsed = time.monotonic() - t0
    print(f"Evaluated {len(non_null_records)} questions in {elapsed:.1f}s")

    print("\nCandidate-pool headroom check:")
    for method in METHODS:
        counts = [e["methods"][method]["num_unique_docs_retrieved"] for e in per_query]
        below_10 = sum(1 for c in counts if c < 10)
        pool = RERANK_CANDIDATE_DEPTH if method == "hybrid_reranker" else RAW_CANDIDATE_POOL_SIZE
        print(
            f"  [{method}] min unique docs/query: {min(counts)}; "
            f"{below_10}/{len(counts)} queries returned <10 unique docs "
            f"from a pool of {pool}"
        )

    print("\nLatency (ms/query):")
    latency_summary = {}
    for stage, values in latencies_ms.items():
        summary = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }
        latency_summary[stage] = summary
        print(f"  {stage}: mean={summary['mean']:.1f}  p50={summary['p50']:.1f}  p95={summary['p95']:.1f}")

    overall = {method: aggregate(per_query, method) for method in METHODS}

    by_question_type = {}
    for qtype in QUESTION_TYPES_EVALUATED:
        subset = [e for e in per_query if e["question_type"] == qtype]
        if subset:
            by_question_type[qtype] = {
                "n": len(subset),
                "metrics": {method: aggregate(subset, method) for method in METHODS},
            }

    by_hop_count = {}
    for hops in HOP_COUNTS_EVALUATED:
        subset = [e for e in per_query if e["hop_count"] == hops]
        if subset:
            by_hop_count[str(hops)] = {
                "n": len(subset),
                "metrics": {method: aggregate(subset, method) for method in METHODS},
            }

    manifest_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and not is_partial_run:
        manifest_dev_ids = set(manifest["splits"]["development"]["qa_ids"])
        evaluated_ids = {e["qa_id"] for e in per_query} | {qa_id(r) for r in null_records}
        if manifest_dev_ids != evaluated_ids:
            raise SystemExit(
                "dev_subset.json does not match the persisted benchmark manifest — "
                "re-run scripts/build_benchmark.py before evaluating."
            )

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": "development",
        "partial_run": is_partial_run,
        "config": {
            "dataset_source": manifest["dataset_source"] if manifest else None,
            "raw_file_sha256": manifest["raw_file_sha256"] if manifest else None,
            "development_seed": manifest["seeds"]["development"] if manifest else None,
            "embedding_model": retrieval_config["embedding"]["model_name"],
            "embedding_dimension": embedding_model.dimension,
            "bm25_model": retrieval_config["bm25"]["model_name"],
            "reranker_model": retrieval_config["reranker"]["model_name"],
            "chunking": retrieval_config["chunking"],
            "hybrid_production_config": retrieval_config["hybrid"],
            "qdrant_collection": collection_name,
            "qdrant_points_count": collection_info.points_count,
            "eval_raw_candidate_pool_size": RAW_CANDIDATE_POOL_SIZE,
            "hybrid_reranker_candidate_depth": RERANK_CANDIDATE_DEPTH,
            "hybrid_reranker_pool_note": (
                "hybrid_reranker uses a FIXED candidate depth of "
                f"{RERANK_CANDIDATE_DEPTH} chunks (production depth, per Phase 5 spec), "
                f"not the {RAW_CANDIDATE_POOL_SIZE}-chunk eval-only pool used for "
                "dense/bm25/hybrid — its Recall@10/Hit@10/NDCG@10/Complete-Evidence@10 "
                "have structurally less headroom; not an apples-to-apples pool size."
            ),
            "hybrid_fusion": {
                "implementation": "deterministic_app_side_rrf",
                "module": "mhrag.retrieval.rrf.deterministic_hybrid_search",
                "rrf_k": RRF_K,
                "rank_convention": "1-based",
                "weights": {"dense": 1.0, "bm25": 1.0},
                "tie_break": ["score_desc", "best_individual_rank_asc", "chunk_id_asc"],
                "supersedes": (
                    "Qdrant server-side FusionQuery(fusion=Fusion.RRF) — verified to be "
                    "Qdrant's default k=2, 0-based rank, with nondeterministic tie-breaking; "
                    "see results/retrieval_eval_development_superseded_qdrant_native_rrf.json"
                ),
            },
            "metric_ks": {
                "recall": [4, 5, 10],
                "hit": [4, 10],
                "complete_evidence": [4, 10],
                "mrr": 10,
                "ndcg": 10,
            },
        },
        "counts": {
            "total_development_questions": len(records),
            "non_null_evaluated": len(non_null_records),
            "null_query_excluded": len(null_records),
        },
        "latency_ms": latency_summary,
        "aggregate": overall,
        "breakdown_by_question_type": by_question_type,
        "breakdown_by_hop_count": by_hop_count,
        "per_query": per_query,
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote evaluation artifact to {out_path}")

    print(f"\nOverall (non-null, n={len(non_null_records)}):")
    for method in METHODS:
        print(f"  {method}: {overall[method]}")


if __name__ == "__main__":
    main()
