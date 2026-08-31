#!/usr/bin/env python
"""Download and validate the MultiHop-RAG dataset.

Usage:
    python scripts/download_dataset.py [--force] [--config configs/dataset.yaml]
"""

from __future__ import annotations

import argparse
import os

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.download import download_dataset
from mhrag.data.loader import load_corpus, load_qa_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]

    print(f"Downloading MultiHop-RAG into {raw_dir} ...")
    results = download_dataset(
        raw_dir,
        base_url=config["source"]["base_url"],
        force=args.force,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    for name, result in results.items():
        cache_note = " (cached)" if result.from_cache else ""
        print(
            f"  {name}: {result.path.name} — {result.num_bytes:,} bytes, "
            f"sha256={result.sha256[:16]}...{cache_note}"
        )

    qa_path = raw_dir / config["source"]["qa_file"]
    corpus_path = raw_dir / config["source"]["corpus_file"]

    print("\nValidating schema and loading records ...")
    qa = load_qa_records(qa_path)
    corpus = load_corpus(corpus_path)

    print(f"\nQA records:        {len(qa):,}")
    print(f"Corpus documents:  {len(corpus):,}")

    type_counts: dict[str, int] = {}
    evidence_hop_counts: dict[int, int] = {}
    for r in qa:
        type_counts[r.question_type] = type_counts.get(r.question_type, 0) + 1
        n_ev = len(r.evidence_list)
        evidence_hop_counts[n_ev] = evidence_hop_counts.get(n_ev, 0) + 1

    print("\nQuestion type distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype:<20} {count:>5}  ({count / len(qa):.1%})")

    print("\nEvidence count (hops) distribution:")
    for n_ev, count in sorted(evidence_hop_counts.items()):
        print(f"  {n_ev} evidence doc(s): {count:>5}  ({count / len(qa):.1%})")


if __name__ == "__main__":
    main()
