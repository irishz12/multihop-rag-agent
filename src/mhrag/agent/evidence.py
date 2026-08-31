"""Deterministic evidence merging across agentic-loop hops.

Chunks are deduplicated by `chunk_id`. A chunk already in the pool from an
earlier hop is recorded as a duplicate for the later hop that re-surfaced
it — it is never moved, re-ranked, or re-scored; the pool retains its
existing position from when it was first added.

Merge ORDER is hop-ascending, then each hop's own (reranked) rank order —
deliberately NOT re-sorted by score across hops, because each hop's
reranker score measures relevance to THAT hop's query (the original
question for hop 1, a controller-generated follow-up query for every hop
after), so scores from different hops are not on a directly comparable
scale. Treating hop order as primary keeps the merge simple, deterministic,
and avoids quietly conflating "relevant to a follow-up query" with
"relevant to the original question."
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.retrieval.schema import RetrievalResult


@dataclass(frozen=True, slots=True)
class MergeResult:
    pool: tuple[RetrievalResult, ...]
    new_chunk_ids: tuple[str, ...]
    duplicate_chunk_ids: tuple[str, ...]


def merge_evidence(
    pool: list[RetrievalResult], new_results: list[RetrievalResult]
) -> MergeResult:
    """Merge `new_results` (one hop's reranked chunks) into `pool` (the
    evidence accumulated so far), deduplicated by `chunk_id`.

    Does not mutate `pool` or `new_results` — returns a new `MergeResult`.
    """
    existing_ids = {r.chunk_id for r in pool}
    updated = list(pool)
    new_ids: list[str] = []
    duplicate_ids: list[str] = []

    for r in new_results:
        if r.chunk_id in existing_ids:
            duplicate_ids.append(r.chunk_id)
            continue
        updated.append(r)
        existing_ids.add(r.chunk_id)
        new_ids.append(r.chunk_id)

    return MergeResult(
        pool=tuple(updated),
        new_chunk_ids=tuple(new_ids),
        duplicate_chunk_ids=tuple(duplicate_ids),
    )
