"""Evidence merge tests: deterministic dedup by chunk_id, doc_id retention,
and no input mutation."""

from __future__ import annotations

import copy

from mhrag.agent.evidence import merge_evidence
from mhrag.retrieval.schema import RetrievalResult


def _result(chunk_id: str, doc_id: str, text: str = "text") -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=1.0, method="hybrid_reranked", chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text=text, position=0,
    )


def test_merge_into_empty_pool_adds_all_as_new():
    new = [_result("a", "doc-a"), _result("b", "doc-b")]
    result = merge_evidence([], new)
    assert [c.chunk_id for c in result.pool] == ["a", "b"]
    assert result.new_chunk_ids == ("a", "b")
    assert result.duplicate_chunk_ids == ()


def test_merge_deduplicates_by_chunk_id():
    pool = [_result("a", "doc-a")]
    new = [_result("a", "doc-a"), _result("b", "doc-b")]  # "a" repeated
    result = merge_evidence(pool, new)
    assert [c.chunk_id for c in result.pool] == ["a", "b"]  # "a" not duplicated in pool
    assert result.new_chunk_ids == ("b",)
    assert result.duplicate_chunk_ids == ("a",)


def test_merge_preserves_doc_id_of_retained_chunks():
    pool = []
    new = [_result("a", "doc-xyz")]
    result = merge_evidence(pool, new)
    assert result.pool[0].doc_id == "doc-xyz"


def test_merge_keeps_original_position_for_duplicate_not_re_added():
    """A duplicate chunk must not move to the end / be re-inserted — the
    pool's existing order is untouched aside from appending genuinely new
    chunks."""
    pool = [_result("a", "doc-a"), _result("b", "doc-b")]
    new = [_result("b", "doc-b"), _result("c", "doc-c")]  # "b" duplicate, "c" new
    result = merge_evidence(pool, new)
    assert [c.chunk_id for c in result.pool] == ["a", "b", "c"]


def test_merge_across_three_hops_is_deterministic_and_cumulative():
    pool: list[RetrievalResult] = []
    hop1 = [_result("a", "doc-a"), _result("b", "doc-b")]
    hop2 = [_result("b", "doc-b"), _result("c", "doc-c")]  # "b" dup
    hop3 = [_result("a", "doc-a"), _result("d", "doc-d")]  # "a" dup

    r1 = merge_evidence(pool, hop1)
    r2 = merge_evidence(list(r1.pool), hop2)
    r3 = merge_evidence(list(r2.pool), hop3)

    assert [c.chunk_id for c in r3.pool] == ["a", "b", "c", "d"]
    assert r1.duplicate_chunk_ids == ()
    assert r2.duplicate_chunk_ids == ("b",)
    assert r3.duplicate_chunk_ids == ("a",)


def test_merge_does_not_mutate_inputs():
    pool = [_result("a", "doc-a")]
    new = [_result("b", "doc-b")]
    pool_before = copy.deepcopy(pool)
    new_before = copy.deepcopy(new)

    merge_evidence(pool, new)

    assert pool == pool_before
    assert new == new_before


def test_merge_with_no_new_results_returns_pool_unchanged():
    pool = [_result("a", "doc-a")]
    result = merge_evidence(pool, [])
    assert [c.chunk_id for c in result.pool] == ["a"]
    assert result.new_chunk_ids == ()
    assert result.duplicate_chunk_ids == ()


def test_same_chunk_id_from_different_docs_still_counts_as_duplicate():
    """chunk_id is the sole dedup key — this documents that behavior
    explicitly (chunk_ids are content-hash-derived per-document in
    production, so this scenario shouldn't arise for real data, but the
    dedup logic itself is chunk_id-only by design)."""
    pool = [_result("shared-id", "doc-a")]
    new = [_result("shared-id", "doc-b")]  # same chunk_id, different doc_id
    result = merge_evidence(pool, new)
    assert len(result.pool) == 1
    assert result.pool[0].doc_id == "doc-a"  # first-seen wins
    assert result.duplicate_chunk_ids == ("shared-id",)
