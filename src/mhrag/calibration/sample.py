"""Deterministic calibration sample selection.

DEVELOPMENT split ONLY. This function itself is split-agnostic (it takes
an already-loaded `list[QARecord]`, never reads a file) — the hard
guarantee that only `dev_subset.json` is ever loaded for calibration lives
in `scripts/agentic_budget_calibration.py` (CALIBRATION_SPLIT_FILE is a
hardcoded module constant there, no CLI flag, same guard pattern as every
other script in this project; see
tests/test_agentic_budget_calibration_guard.py).

`null_query` is excluded (no gold evidence to calibrate against). The
sample is balanced across `question_type` as evenly as possible, and
within each `question_type`, balanced across `hop_count` (number of unique
gold documents) as evenly as possible.

DATA CONSTRAINT (verified empirically, not assumed): the development split
has ZERO 4-hop `comparison_query` or `temporal_query` questions — cross-
tabulating `question_type` x `hop_count` over all 265 non-null development
questions found 4-hop questions only among `inference_query`. `CELL_TARGETS`
below reflects this: 9 questions per question_type (27 total), hop_count
distribution as balanced as that real constraint allows (13x 2-hop, 11x
3-hop, 3x 4-hop — the 4-hop questions can only come from inference_query).
"""

from __future__ import annotations

import random

from mhrag.data.schema import QARecord
from mhrag.eval.ground_truth import hop_count

CALIBRATION_SEED = 42

# (question_type, hop_count) -> how many to select. See module docstring
# for why comparison_query/temporal_query have no hop_count=4 entries.
CELL_TARGETS: dict[tuple[str, int], int] = {
    ("inference_query", 2): 3,
    ("inference_query", 3): 3,
    ("inference_query", 4): 3,
    ("comparison_query", 2): 5,
    ("comparison_query", 3): 4,
    ("temporal_query", 2): 5,
    ("temporal_query", 3): 4,
}


def select_calibration_sample(
    records: list[QARecord],
    seed: int = CALIBRATION_SEED,
    cell_targets: dict[tuple[str, int], int] | None = None,
) -> list[QARecord]:
    """Deterministically select the calibration sample.

    Excludes `null_query`. Raises `ValueError` if any target cell has
    fewer candidates than requested, rather than silently under-filling
    (a silent shortfall would quietly unbalance the sample).
    """
    cell_targets = cell_targets if cell_targets is not None else CELL_TARGETS
    non_null = [r for r in records if r.question_type != "null_query"]

    cells: dict[tuple[str, int], list[QARecord]] = {}
    for r in non_null:
        key = (r.question_type, hop_count(r))
        cells.setdefault(key, []).append(r)

    rng = random.Random(seed)
    selected: list[QARecord] = []
    for key in sorted(cell_targets):
        target = cell_targets[key]
        pool = sorted(cells.get(key, []), key=lambda r: r.query)  # deterministic order before sampling
        if len(pool) < target:
            raise ValueError(
                f"calibration cell {key} has only {len(pool)} candidate(s) in the "
                f"given records, need {target}"
            )
        selected.extend(rng.sample(pool, k=target))

    selected.sort(key=lambda r: (r.question_type, hop_count(r), r.query))  # deterministic output order
    return selected
