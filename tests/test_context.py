"""Context assembler tests: deterministic assembly, budget enforcement, and
chunk/doc id + metadata preservation."""

from __future__ import annotations

from mhrag.generation.context import approximate_token_count, assemble_context
from mhrag.retrieval.schema import RetrievalResult


def _result(rank: int, chunk_id: str, doc_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=1.0 / rank,
        method="hybrid_reranked",
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=f"Title {doc_id}",
        url=f"https://example.com/{doc_id}",
        source="Source",
        category="technology",
        published_at="2024-01-01T00:00:00+00:00",
        text=text,
        position=0,
    )


def _word_count(text: str) -> int:
    return len(text.split())


# --- deterministic context assembly -----------------------------------------------------


def test_assembly_is_deterministic_across_repeated_calls():
    results = [_result(1, "a", "doc-a", "some text here"), _result(2, "b", "doc-b", "more text")]
    first = assemble_context(results, _word_count, top_k=5, max_context_tokens=1000)
    second = assemble_context(results, _word_count, top_k=5, max_context_tokens=1000)
    assert first == second


def test_assembly_preserves_rank_order():
    results = [
        _result(1, "a", "doc-a", "first chunk"),
        _result(2, "b", "doc-b", "second chunk"),
        _result(3, "c", "doc-c", "third chunk"),
    ]
    context = assemble_context(results, _word_count, top_k=5, max_context_tokens=1000)
    assert [c.chunk_id for c in context.chunks_included] == ["a", "b", "c"]
    assert context.context_text.index("first chunk") < context.context_text.index("second chunk")
    assert context.context_text.index("second chunk") < context.context_text.index("third chunk")


def test_assembly_respects_top_k():
    results = [_result(i, f"c{i}", f"doc-{i}", "word " * 5) for i in range(1, 11)]
    context = assemble_context(results, _word_count, top_k=3, max_context_tokens=10000)
    assert len(context.chunks_included) == 3
    assert [c.chunk_id for c in context.chunks_included] == ["c1", "c2", "c3"]


# --- context budget enforcement -----------------------------------------------------------


def test_budget_never_silently_exceeded():
    # 3 chunks of 10 words each; budget of 15 tokens should fit only 1.
    results = [_result(i, f"c{i}", f"doc-{i}", " ".join(["word"] * 10)) for i in range(1, 4)]
    context = assemble_context(results, _word_count, top_k=3, max_context_tokens=15)
    assert context.total_token_count <= 15
    assert len(context.chunks_included) == 1
    assert len(context.chunks_dropped) == 2


def test_chunks_after_first_overflow_are_also_dropped_not_backfilled():
    """A later, smaller chunk that would numerically fit must still be
    dropped once an earlier chunk overflows — rank order beats squeezing
    in extra content."""
    results = [
        _result(1, "big", "doc-1", " ".join(["word"] * 20)),  # 20 tokens, overflows a 15 budget
        _result(2, "small", "doc-2", "tiny"),  # 1 token, would fit alone
    ]
    context = assemble_context(results, _word_count, top_k=2, max_context_tokens=15)
    assert [c.chunk_id for c in context.chunks_included] == []
    assert [c.chunk_id for c in context.chunks_dropped] == ["big", "small"]


def test_exact_budget_boundary_is_inclusive():
    results = [_result(1, "a", "doc-a", " ".join(["word"] * 10))]  # exactly 10 tokens
    context = assemble_context(results, _word_count, top_k=1, max_context_tokens=10)
    assert len(context.chunks_included) == 1
    assert context.total_token_count == 10


def test_budget_of_zero_drops_everything():
    results = [_result(1, "a", "doc-a", "some text")]
    context = assemble_context(results, _word_count, top_k=1, max_context_tokens=0)
    assert context.chunks_included == ()
    assert len(context.chunks_dropped) == 1


def test_empty_results_yields_empty_context():
    context = assemble_context([], _word_count, top_k=5, max_context_tokens=1000)
    assert context.context_text == ""
    assert context.chunks_included == ()
    assert context.chunks_dropped == ()
    assert context.total_token_count == 0
    assert context.source_doc_ids == ()


# --- metadata / chunk id preservation ------------------------------------------------------


def test_chunk_id_and_doc_id_preserved_in_included_chunks():
    results = [_result(1, "chunk-xyz", "doc-abc", "text")]
    context = assemble_context(results, _word_count, top_k=1, max_context_tokens=1000)
    assert context.chunks_included[0].chunk_id == "chunk-xyz"
    assert context.chunks_included[0].doc_id == "doc-abc"


def test_chunk_id_and_doc_id_preserved_in_dropped_chunks():
    results = [_result(1, "chunk-xyz", "doc-abc", " ".join(["word"] * 50))]
    context = assemble_context(results, _word_count, top_k=1, max_context_tokens=1)
    assert context.chunks_dropped[0].chunk_id == "chunk-xyz"
    assert context.chunks_dropped[0].doc_id == "doc-abc"


def test_source_doc_ids_are_unique_and_ordered_by_first_appearance():
    results = [
        _result(1, "c1", "doc-a", "text one"),
        _result(2, "c2", "doc-b", "text two"),
        _result(3, "c3", "doc-a", "text three"),  # doc-a again, different chunk
    ]
    context = assemble_context(results, _word_count, top_k=3, max_context_tokens=1000)
    assert context.source_doc_ids == ("doc-a", "doc-b")  # doc-a once, in first-seen order


def test_approximate_token_count_is_positive_and_deterministic():
    assert approximate_token_count("") >= 1
    assert approximate_token_count("hello world") == approximate_token_count("hello world")
    assert approximate_token_count("a much longer piece of text than before") > approximate_token_count("short")
