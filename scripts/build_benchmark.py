#!/usr/bin/env python
"""Build the hardened benchmark: development / final-holdout / smoke splits,
plus a persisted manifest.

Supersedes scripts/build_dev_subset.py for benchmark purposes (that script
is left as-is; this one additionally builds the final holdout, smoke subset,
and manifest, reusing the same `subset.seed`/`subset.size` config for
development so both scripts produce an identical development split).

Usage:
    python scripts/build_benchmark.py [--config configs/dataset.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict

from mhrag.config import PROJECT_ROOT, load_config
from mhrag.data.benchmark import build_benchmark_splits, build_manifest, qa_id
from mhrag.data.loader import load_qa_records
from mhrag.data.sampling import SubsetSpec
from mhrag.data.schema import QARecord


def _record_to_dict(r: QARecord) -> dict:
    return {
        "query": r.query,
        "answer": r.answer,
        "question_type": r.question_type,
        "evidence_list": [asdict(e) for e in r.evidence_list],
    }


def _write_split(records: list[QARecord], out_file: str) -> None:
    out_path = PROJECT_ROOT / out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([_record_to_dict(r) for r in records], indent=2))


def _print_distribution(name: str, records: list[QARecord]) -> None:
    dist: dict[str, int] = {}
    for r in records:
        dist[r.question_type] = dist.get(r.question_type, 0) + 1
    print(f"  {name} (n={len(records)}):")
    for qtype, count in sorted(dist.items()):
        print(f"    {qtype:<20} {count:>4}  ({count / len(records):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    qa_path = raw_dir / config["source"]["qa_file"]
    corpus_path = raw_dir / config["source"]["corpus_file"]

    population = load_qa_records(qa_path)

    dev_spec = SubsetSpec(size=config["subset"]["size"], seed=config["subset"]["seed"])
    final_cfg = config["benchmark"]["final_holdout"]
    smoke_cfg = config["benchmark"]["smoke"]
    final_spec = SubsetSpec(size=final_cfg["size"], seed=final_cfg["seed"])
    smoke_spec = SubsetSpec(size=smoke_cfg["size"], seed=smoke_cfg["seed"])

    print(f"Population: {len(population)} QA records")
    print(
        f"Seeds — development: {dev_spec.seed}, final_holdout: {final_spec.seed}, "
        f"smoke: {smoke_spec.seed}\n"
    )

    splits = build_benchmark_splits(population, dev_spec, final_spec, smoke_spec)

    _write_split(splits.development, config["subset"]["output_file"])
    _write_split(splits.final_holdout, final_cfg["output_file"])
    _write_split(splits.smoke, smoke_cfg["output_file"])

    raw_file_sha256 = {
        config["source"]["corpus_file"]: hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        config["source"]["qa_file"]: hashlib.sha256(qa_path.read_bytes()).hexdigest(),
    }

    manifest = build_manifest(
        dataset_source={
            "hf_dataset": config["source"]["hf_dataset"],
            "base_url": config["source"]["base_url"],
        },
        population=population,
        splits=splits,
        seeds={
            "development": dev_spec.seed,
            "final_holdout": final_spec.seed,
            "smoke": smoke_spec.seed,
        },
        raw_file_sha256=raw_file_sha256,
    )
    manifest_path = PROJECT_ROOT / config["benchmark"]["manifest_file"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("Split sizes:")
    _print_distribution("development", splits.development)
    _print_distribution("final_holdout", splits.final_holdout)
    _print_distribution("smoke", splits.smoke)

    dev_ids = {qa_id(r) for r in splits.development}
    final_ids = {qa_id(r) for r in splits.final_holdout}
    smoke_ids = {qa_id(r) for r in splits.smoke}

    print("\nOverlap checks:")
    print(f"  development ∩ final_holdout: {len(dev_ids & final_ids)} (must be 0)")
    print(f"  smoke ⊆ development:         {smoke_ids <= dev_ids}")

    print(f"\nManifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
