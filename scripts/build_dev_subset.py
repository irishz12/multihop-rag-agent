#!/usr/bin/env python
"""Build a deterministic, stratified development/evaluation subset.

Run after scripts/download_dataset.py. Writes a JSON file with the same
record shape as MultiHopRAG.json (query, answer, question_type,
evidence_list) — evidence is preserved for later evaluation, never stripped.

Usage:
    python scripts/build_dev_subset.py [--config configs/dataset.yaml]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.loader import load_qa_records
from mhrag.data.sampling import SubsetSpec, stratified_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    qa_path = raw_dir / config["source"]["qa_file"]

    records = load_qa_records(qa_path)
    spec = SubsetSpec(size=config["subset"]["size"], seed=config["subset"]["seed"])
    subset = stratified_sample(records, spec)

    out_path = PROJECT_ROOT / config["subset"]["output_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "query": r.query,
            "answer": r.answer,
            "question_type": r.question_type,
            "evidence_list": [asdict(e) for e in r.evidence_list],
        }
        for r in subset
    ]
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(subset)} records (seed={spec.seed}) to {out_path}")
    print(f"\nSource population: {len(records)} records")

    def distribution(recs) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in recs:
            counts[r.question_type] = counts.get(r.question_type, 0) + 1
        return counts

    pop_dist = distribution(records)
    sub_dist = distribution(subset)

    print("\nquestion_type         population        subset")
    for qtype in sorted(pop_dist):
        p = pop_dist[qtype]
        s = sub_dist.get(qtype, 0)
        print(
            f"  {qtype:<20} {p:>5} ({p / len(records):.1%})   "
            f"{s:>5} ({s / len(subset):.1%})"
        )


if __name__ == "__main__":
    main()
