"""Deterministic context assembly: ranked retrieval results -> a single
context string bounded by a configurable token budget, for the answer
generator.

Never includes ground-truth `answer`/`evidence_list`/`question_type` —
`assemble_context` only ever reads `.chunk_id`/`.doc_id`/`.text` off
already-retrieved `RetrievalResult` objects (which structurally cannot
carry ground truth — see `mhrag.retrieval.schema`). Source chunk/doc ids
are retained on the returned `AssembledContext` for evaluation/debugging;
they are never part of `context_text` itself and never shown to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mhrag.retrieval.schema import RetrievalResult

TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class ContextChunk:
    chunk_id: str
    doc_id: str
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class AssembledContext:
    context_text: str  # what actually gets sent to the model
    chunks_included: tuple[ContextChunk, ...]
    chunks_dropped: tuple[ContextChunk, ...]  # dropped for budget, original rank order
    total_token_count: int
    source_doc_ids: tuple[str, ...]  # unique, in order of first appearance


def assemble_context(
    results: list[RetrievalResult],
    count_tokens: TokenCounter,
    top_k: int,
    max_context_tokens: int,
) -> AssembledContext:
    """Deterministic context assembly.

    Takes the top-K retrieval results (already rank-ordered) and packs them
    into the context in that order, stopping as soon as the running token
    total would exceed `max_context_tokens` — the budget is NEVER silently
    exceeded. A chunk that would overflow the budget is dropped, and so is
    every chunk after it in rank order (results stay strictly best-first;
    this never reshuffles to backfill with a smaller, lower-ranked chunk
    that would numerically still fit — rank order is treated as more
    important than squeezing in one more chunk).
    """
    candidates = results[:top_k]

    included: list[ContextChunk] = []
    dropped: list[ContextChunk] = []
    running_total = 0
    budget_exceeded = False

    for r in candidates:
        token_count = count_tokens(r.text)
        chunk = ContextChunk(
            chunk_id=r.chunk_id, doc_id=r.doc_id, text=r.text, token_count=token_count
        )
        if budget_exceeded or running_total + token_count > max_context_tokens:
            dropped.append(chunk)
            budget_exceeded = True
            continue
        included.append(chunk)
        running_total += token_count

    context_text = "\n\n".join(f"[Source {i + 1}]\n{c.text}" for i, c in enumerate(included))

    source_doc_ids: list[str] = []
    seen: set[str] = set()
    for c in included:
        if c.doc_id not in seen:
            seen.add(c.doc_id)
            source_doc_ids.append(c.doc_id)

    return AssembledContext(
        context_text=context_text,
        chunks_included=tuple(included),
        chunks_dropped=tuple(dropped),
        total_token_count=running_total,
        source_doc_ids=tuple(source_doc_ids),
    )


def approximate_token_count(text: str) -> int:
    """Rough approximation (~4 chars/token — the common heuristic for
    English text), NOT tied to any specific tokenizer. Sufficient for
    enforcing a soft context budget deterministically; NOT used for exact
    cost accounting — exact token counts for cost come from Mantle's own
    `usage` field on the response (see `mhrag.generation.mantle_client`),
    captured separately after the call actually happens."""
    return max(1, len(text) // 4)
