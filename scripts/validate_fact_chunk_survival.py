#!/usr/bin/env python
"""Phase 5A, STEP 1 (dev-only, offline, ZERO LLM/API calls, VALIDATION ONLY):
does a gold Evidence.fact survive intact inside the chunks the existing,
UNMODIFIED chunking pipeline actually produces?

Mirrors scripts/build_index.py's exact production chunking invocation
(same load_corpus, same EmbeddingModel token counter, same ChunkingConfig
from configs/retrieval.yaml, same chunk_corpus call) so the chunks
reconstructed here are the same ones the real index was built from — this
script never modifies mhrag.ingestion.chunking, mhrag.ingestion.embedding,
or any config; it only calls them exactly as scripts/build_index.py
already does, read-only.

For every one of the 722 gold facts across the 265 non-null DEVELOPMENT
questions (data/processed/dev_subset.json — never final_holdout.json),
classifies:
  - "intact_in_one_chunk": the normalized fact is a substring of exactly
    one reconstructed chunk's normalized text
  - "cross_chunk": not found in any single chunk, but found in the
    concatenation of two adjacent chunks from the same document (position
    i, i+1) — the chunk-boundary-split case
  - "absent": found in neither

Normalization is whitespace-collapse + straight-quote/dash unification
only (never lowercasing or token-level normalization) — Evidence.fact is
a verbatim, case-sensitive extractive quote (see Phase 5 audit: 722/722
exact substring matches against corpus.json), so the only expected
divergence between corpus.json's raw body text and chunked text is
paragraph-join whitespace, not casing or wording.

Writes ONLY results/fact_grounding_chunk_survival.json — never modifies
data/raw/corpus.json, data/processed/dev_subset.json, or any results/*.json.

Usage:
    python scripts/validate_fact_chunk_survival.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.loader import load_corpus, load_qa_records
from mhrag.data.schema import doc_id_from_url
from mhrag.ingestion.chunking import ChunkingConfig, chunk_corpus
from mhrag.ingestion.embedding import EmbeddingModel

DEV_SPLIT_FILE = "dev_subset.json"  # hardcoded — no CLI flag, cannot reach final_holdout.json
OUTPUT_FILE = "results/fact_grounding_chunk_survival.json"  # this script's ONLY write target

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTE_TABLE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    """Whitespace-collapse + curly-quote/dash unification only — NEVER
    lowercasing or word-level normalization, since Evidence.fact is a
    case-sensitive verbatim quote (see module docstring)."""
    return _WHITESPACE_RE.sub(" ", text.translate(_QUOTE_TABLE)).strip()


def main() -> None:
    dataset_config = load_config("configs/dataset.yaml")
    retrieval_config = load_config("configs/retrieval.yaml")

    raw_dir = PROJECT_ROOT / dataset_config["paths"]["raw_dir"]
    corpus_path = raw_dir / dataset_config["source"]["corpus_file"]
    print(f"Loading corpus from {corpus_path} ...")
    documents = load_corpus(corpus_path)
    print(f"Corpus documents: {len(documents)}")

    print(f"Loading embedding model {retrieval_config['embedding']['model_name']} (for its tokenizer only) ...")
    embedding_model = EmbeddingModel(
        model_name=retrieval_config["embedding"]["model_name"],
        device=retrieval_config["embedding"].get("device"),
        normalize=retrieval_config["embedding"]["normalize"],
        query_instruction=retrieval_config["embedding"].get("query_instruction", ""),
        batch_size=retrieval_config["embedding"]["batch_size"],
    )
    count_tokens = embedding_model.build_token_counter()

    chunk_cfg = retrieval_config["chunking"]
    chunking_config = ChunkingConfig(
        target_tokens=chunk_cfg["target_tokens"], max_tokens=chunk_cfg["max_tokens"],
        overlap_tokens=chunk_cfg["overlap_tokens"],
    )
    print("Chunking corpus (paragraph-aware, per-document, unmodified production logic) ...")
    chunks = chunk_corpus(documents, count_tokens, chunking_config)
    print(f"Total chunks: {len(chunks)}")

    chunks_by_doc: dict[str, list] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c.doc_id].append(c)
    for doc_id in chunks_by_doc:
        chunks_by_doc[doc_id].sort(key=lambda c: c.position)

    dev_path = PROJECT_ROOT / dataset_config["paths"]["processed_dir"] / DEV_SPLIT_FILE
    dev_records = load_qa_records(dev_path)
    non_null = [r for r in dev_records if r.question_type != "null_query"]
    print(f"Non-null development questions: {len(non_null)}")

    results = []
    counts = {"intact_in_one_chunk": 0, "cross_chunk": 0, "absent": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"intact_in_one_chunk": 0, "cross_chunk": 0, "absent": 0})
    examples: dict[str, list] = {"intact_in_one_chunk": [], "cross_chunk": [], "absent": []}

    for record in non_null:
        for evidence in record.evidence_list:
            doc_id = doc_id_from_url(evidence.url)
            fact_norm = normalize(evidence.fact)
            doc_chunks = chunks_by_doc.get(doc_id, [])

            category = "absent"
            matched_chunk_ids: list[str] = []

            for c in doc_chunks:
                if fact_norm in normalize(c.text):
                    category = "intact_in_one_chunk"
                    matched_chunk_ids = [c.chunk_id]
                    break

            if category == "absent":
                for i in range(len(doc_chunks) - 1):
                    combined = normalize(doc_chunks[i].text) + " " + normalize(doc_chunks[i + 1].text)
                    if fact_norm in combined:
                        category = "cross_chunk"
                        matched_chunk_ids = [doc_chunks[i].chunk_id, doc_chunks[i + 1].chunk_id]
                        break

            counts[category] += 1
            by_type[record.question_type][category] += 1
            record_entry = {
                "question_type": record.question_type, "doc_id": doc_id, "url": evidence.url,
                "fact": evidence.fact, "matched_chunk_ids": matched_chunk_ids,
                "n_chunks_in_doc": len(doc_chunks),
            }
            results.append({**record_entry, "category": category})
            if len(examples[category]) < 5:
                examples[category].append(record_entry)

    total = sum(counts.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 5A STEP 1 (validation only) — corpus fact chunk-survival check, dev-only, "
                   "zero LLM/API calls, unmodified production chunking pipeline",
        "total_facts": total,
        "counts": counts,
        "rates": {k: v / total for k, v in counts.items()} if total else {},
        "breakdown_by_question_type": {
            qt: {"counts": dict(c), "rates": {k: v / sum(c.values()) for k, v in c.items()}}
            for qt, c in by_type.items()
        },
        "examples_by_category": examples,
        "all_records": results,
    }
    out_path = PROJECT_ROOT / OUTPUT_FILE
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nTotal facts: {total}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v/total:.2%})")
    print("\nBy question_type:")
    for qt, c in by_type.items():
        print(f"  {qt}: {dict(c)}")


if __name__ == "__main__":
    main()
