#!/usr/bin/env python
"""LIVE Amazon Bedrock Mantle smoke check.

Makes REAL API calls to Mantle and incurs REAL (small) cost. This is
connectivity/instrumentation validation, NOT answer-quality evaluation —
not run by pytest, not part of `pytest`'s default collection, run
explicitly:

    python scripts/mantle_smoke_check.py --n 5

Reads ONLY data/processed/smoke_subset.json — this module has no code path
that can reach final_holdout.json (no CLI flag, no config option; the split
file is a hardcoded module constant, same guard pattern as
scripts/run_retrieval_eval.py and scripts/retrieval_sanity_check.py). Does
NOT run against the full development benchmark — `--n` defaults to 5 and
is meant to stay small (3-5 questions).

For each question: retrieve with the existing Hybrid+Reranker pipeline,
assemble context, generate an answer through Mantle, and record real
usage/cost/latency. Only `query` text and retrieved chunk text are ever
sent to Mantle — never the gold `answer`, `evidence_list`, or
`question_type` (see mhrag.generation.answer module docstring for why this
is structural, not just a convention followed here).

Requires `$OPENAI_API_KEY` in the environment (see .env.example). The key
is never printed, logged, or written to any output file by this script or
any of the mhrag.generation modules it uses.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.generation.answer import generate_answer
from mhrag.generation.context import approximate_token_count
from mhrag.generation.mantle_client import MantleClient, MantleConfigError
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.qdrant_store import get_client
from mhrag.retrieval.rerank import Reranker, rerank_hybrid_search

# Hardcoded to the smoke split ONLY — no CLI flag, no config option, so
# there is no code path in this script that can reach final_holdout.json.
SMOKE_SPLIT_FILE = "smoke_subset.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="number of smoke questions (3-5 recommended)")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--mantle-config", default="configs/mantle.yaml")
    parser.add_argument("--output", default="results/mantle_smoke_check.json")
    args = parser.parse_args()

    if args.n > 5:
        print(f"Warning: --n={args.n} exceeds the recommended 3-5 smoke questions for this phase.")

    dataset_config = load_config(args.config)
    retrieval_config = load_config(args.retrieval_config)
    mantle_config = load_config(args.mantle_config)

    smoke_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / SMOKE_SPLIT_FILE
    records = json.loads(smoke_path.read_text())[: args.n]
    print(f"Loaded {len(records)} smoke questions from {smoke_path}")

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

    qdrant_client = get_client(retrieval_config["qdrant"]["url"])
    collection_name = retrieval_config["qdrant"]["collection_name"]

    print(f"\nConnecting to Mantle: model={mantle_config['generation']['model_id']}")
    try:
        mantle_client = MantleClient(
            model_id=mantle_config["generation"]["model_id"],
            base_url_env=mantle_config["client"]["base_url_env"],
            default_base_url=mantle_config["client"]["default_base_url"],
            api_key_env=mantle_config["client"]["api_key_env"],
            timeout_seconds=mantle_config["client"]["timeout_seconds"],
            temperature=mantle_config["generation"]["temperature"],
            max_output_tokens=mantle_config["generation"]["max_output_tokens"],
            max_retries=mantle_config["client"]["max_retries"],
            retry_base_delay_seconds=mantle_config["client"]["retry_base_delay_seconds"],
        )
    except MantleConfigError as exc:
        raise SystemExit(f"Cannot run live Mantle smoke check: {exc}") from exc

    pricing = mantle_config["pricing"]
    top_k = mantle_config["context"]["top_k_chunks"]
    max_context_tokens = mantle_config["context"]["max_context_tokens"]

    results = []
    total_cost = 0.0
    n_succeeded = 0
    n_failed = 0

    for i, rec in enumerate(records):
        print(f"\n[{i + 1}/{len(records)}] {rec['query']}")
        retrieved = rerank_hybrid_search(
            rec["query"], qdrant_client, collection_name, embedding_model, bm25_model, reranker,
            final_top_k=top_k,
        )
        gen = generate_answer(
            rec["query"],
            retrieved,
            mantle_client,
            approximate_token_count,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
            input_price_per_million=pricing["input_per_million_tokens"],
            output_price_per_million=pricing["output_per_million_tokens"],
            prompt_version=mantle_config["generation"]["prompt_version"],
        )

        resp = gen.mantle_response
        if resp.success:
            n_succeeded += 1
            print(f"  answer: {gen.answer[:200]!r}")
        else:
            n_failed += 1
            print(f"  FAILED: {resp.error}")
        print(
            f"  tokens: in={resp.usage.input_tokens} out={resp.usage.output_tokens} "
            f"total={resp.usage.total_tokens}"
        )
        print(f"  cost: ${gen.cost.total_cost_usd if gen.cost.total_cost_usd is not None else 'unknown'}")
        print(f"  latency: llm={resp.llm_latency_ms:.0f}ms total={resp.total_latency_ms:.0f}ms")
        print(
            f"  context: {len(gen.context.chunks_included)} chunks included, "
            f"{len(gen.context.chunks_dropped)} dropped, {gen.context.total_token_count} tokens (approx), "
            f"{len(gen.context.source_doc_ids)} source docs"
        )

        if gen.cost.total_cost_usd is not None:
            total_cost += gen.cost.total_cost_usd

        results.append(
            {
                "query": rec["query"],
                "answer": gen.answer,
                "success": resp.success,
                "error": resp.error,
                "retry_count": resp.retry_count,
                "model": resp.model,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
                "cost_usd": {
                    "input": gen.cost.input_cost_usd,
                    "output": gen.cost.output_cost_usd,
                    "total": gen.cost.total_cost_usd,
                },
                "latency_ms": {
                    "llm": resp.llm_latency_ms,
                    "total": resp.total_latency_ms,
                },
                "context": {
                    "chunks_included": len(gen.context.chunks_included),
                    "chunks_dropped": len(gen.context.chunks_dropped),
                    "token_count": gen.context.total_token_count,
                    "source_doc_ids": list(gen.context.source_doc_ids),
                    "chunk_ids_used": [c.chunk_id for c in gen.context.chunks_included],
                },
                "prompt_version": gen.prompt_version,
            }
        )

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "connectivity/instrumentation smoke validation — NOT answer-quality evaluation",
        "split": "smoke",
        "model": mantle_config["generation"]["model_id"],
        "region": mantle_config["client"]["region"],
        "n_questions": len(records),
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "total_cost_usd": total_cost,
        "pricing": pricing,
        "results": results,
    }
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Succeeded: {n_succeeded}/{len(records)}  Failed: {n_failed}/{len(records)}")
    print(f"Total smoke-test cost: ${total_cost:.6f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
