#!/usr/bin/env python
"""FINAL HOLDOUT evaluation — mark final_holdout.json as CONSUMED.

Writes a permanent, separate marker file, results/final_holdout_consumed
.json — never mutates results/final_evaluation_manifest.json itself (that
file stays an immutable pre-access snapshot; see its own guard test,
tests/test_freeze_final_evaluation_manifest_guard.py, which asserts its
`final_holdout_access_status` field stays "NOT_YET_ACCESSED" forever).

Run this ONLY after scripts/analyze_phase9_holdout.py has produced
results/phase9_holdout_report.json with `integrity_check: "PASSED"` —
this script checks for exactly that before writing the marker, and refuses
otherwise (SystemExit), so the "consumed" marker can never be written
before the evaluation actually finished cleanly.

Downstream: any future script attempting a NEW final_holdout access should
check for this marker first and refuse — no such check exists yet
elsewhere in the codebase (this is Phase 9's one-time evaluation; a
future phase reusing final_holdout for anything else is explicitly out of
scope and would need its own deliberate decision, not an automatic one).

Usage:
    python scripts/mark_final_holdout_consumed.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT

REPORT_PATH = "results/phase9_holdout_report.json"
OUTPUT_PATH = "results/final_holdout_consumed.json"


def main() -> None:
    report_path = PROJECT_ROOT / REPORT_PATH
    if not report_path.exists():
        raise SystemExit(f"{report_path} does not exist — run scripts/analyze_phase9_holdout.py first")
    report = json.loads(report_path.read_text())
    if report.get("integrity_check") != "PASSED — all frozen files unchanged since pre-access manifest":
        raise SystemExit(
            f"Refusing to mark final_holdout consumed — {REPORT_PATH}'s integrity_check field is "
            f"{report.get('integrity_check')!r}, not the expected PASSED value"
        )

    marker = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "FINAL HOLDOUT evaluation — CONSUMED marker. final_holdout.json has been used for "
                   "its one-time Phase 9 evaluation (Adaptive RAG vs Agentic Multi-Hop RAG, 50-question "
                   "stratified sample). No further tuning is permitted based on these results.",
        "status": "CONSUMED",
        "sample_seed": report["sample_seed"],
        "sample_size": report["sample_size"],
        "pre_access_manifest_generated_at": report["pre_access_manifest_generated_at"],
        "holdout_report_generated_at": report["generated_at"],
        "holdout_report_path": REPORT_PATH,
        "integrity_check": report["integrity_check"],
    }
    out_path = PROJECT_ROOT / OUTPUT_PATH
    out_path.write_text(json.dumps(marker, indent=2))
    print(f"final_holdout.json marked CONSUMED — wrote {out_path}")


if __name__ == "__main__":
    main()
