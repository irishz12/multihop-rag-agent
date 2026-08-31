"""Deterministic, paragraph-aware chunking.

Chunks are built by greedily packing whole paragraphs (split on the corpus's
"\\n\\n" paragraph boundary — verified against the real corpus.json body
text) up to `target_tokens`, never mixing text from two different source
documents (chunking runs per-document; there is no cross-document packing
step). A paragraph that alone exceeds `max_tokens` is sentence-split as a
fallback (and, in the pathological case of one over-long "sentence",
whitespace-split) so no chunk ever exceeds the hard ceiling.

Token counting is injected (`TokenCounter`) rather than hard-coded to a
specific tokenizer, so:
  - production code counts tokens with the actual embedding model's
    tokenizer (see `mhrag.ingestion.embedding.build_token_counter`), keeping
    chunk sizes meaningful relative to what will actually be embedded;
  - unit tests can use a cheap, deterministic stand-in and stay fast/offline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

from mhrag.data.schema import CorpusDocument

TokenCounter = Callable[[str], int]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_FIELD_SEP = "\x1f"


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 300
    max_tokens: int = 400
    overlap_tokens: int = 40


@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexable unit: text plus everything needed to trace it back to
    its source document for later evidence-based evaluation."""

    chunk_id: str
    doc_id: str
    title: str
    url: str
    source: str
    category: str
    published_at: str
    text: str
    position: int  # 0-indexed position of this chunk within its source document
    token_count: int


def _split_paragraphs(body: str) -> list[str]:
    return [p.strip() for p in body.split("\n\n") if p.strip()]


def _split_long_paragraph(paragraph: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Fallback for a single paragraph that alone exceeds max_tokens.

    Guarantees every returned piece is <= max_tokens: sentence-splits first,
    then hard-splits (on whitespace) any individual sentence that is *still*
    too long on its own. A naive version of this that returned raw sentences
    unchecked let a single very long, low-punctuation sentence (long lists,
    "Table of Contents"-style runs — common in the real corpus) through as
    an oversized chunk; this is why every sentence is re-checked, not just
    paragraphs with no sentence boundaries at all.
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph) if s.strip()]
    if not sentences:
        return []

    pieces: list[str] = []
    for sentence in sentences:
        if count_tokens(sentence) <= max_tokens:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split_words(sentence, max_tokens, count_tokens))
    return pieces


def _hard_split_words(text: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Whitespace-window split, respecting max_tokens per piece — last
    resort for text with no usable sentence boundaries."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = count_tokens(word)
        if current and current_tokens + word_tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _take_trailing_overlap(
    units: list[str], overlap_tokens: int, count_tokens: TokenCounter
) -> list[str]:
    overlap: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = count_tokens(unit)
        if overlap and total + unit_tokens > overlap_tokens:
            break
        overlap.insert(0, unit)
        total += unit_tokens
    return overlap


def _pack_units(
    units: list[str], config: ChunkingConfig, count_tokens: TokenCounter
) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        would_exceed_max = bool(current) and (current_tokens + unit_tokens > config.max_tokens)
        reached_target = bool(current) and current_tokens >= config.target_tokens
        if current and (reached_target or would_exceed_max):
            groups.append(current)
            overlap = _take_trailing_overlap(current, config.overlap_tokens, count_tokens)
            overlap_tokens = sum(count_tokens(u) for u in overlap)
            # The hard ceiling always wins over the overlap nicety: if
            # carrying the overlap forward would itself push the next unit
            # (which is individually <= max_tokens, but may be large) over
            # max_tokens, drop the overlap for this boundary instead of
            # violating it. Every `unit` is guaranteed <= max_tokens on its
            # own (see chunk_document/_split_long_paragraph), so starting a
            # fresh chunk with just this unit is always safe.
            if overlap_tokens + unit_tokens > config.max_tokens:
                current, current_tokens = [], 0
            else:
                current, current_tokens = overlap, overlap_tokens
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        groups.append(current)
    return groups


def _chunk_id(doc_id: str, position: int, text: str) -> str:
    basis = _FIELD_SEP.join((doc_id, str(position), text))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def chunk_document(
    doc: CorpusDocument,
    count_tokens: TokenCounter,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[Chunk]:
    """Chunk one document. Never reads or references any other document —
    this is what makes cross-document contamination structurally impossible,
    not just avoided by convention."""
    paragraphs = _split_paragraphs(doc.body)

    units: list[str] = []
    for paragraph in paragraphs:
        if count_tokens(paragraph) <= config.max_tokens:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, config.max_tokens, count_tokens))

    if not units:
        return []

    groups = _pack_units(units, config, count_tokens)

    chunks: list[Chunk] = []
    for position, group in enumerate(groups):
        text = "\n\n".join(group)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc.doc_id, position, text),
                doc_id=doc.doc_id,
                title=doc.title,
                url=doc.url,
                source=doc.source,
                category=doc.category,
                published_at=doc.published_at,
                text=text,
                position=position,
                token_count=count_tokens(text),
            )
        )
    return chunks


def chunk_corpus(
    documents: list[CorpusDocument],
    count_tokens: TokenCounter,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[Chunk]:
    """Chunk every document in the corpus. Chunking is per-document (see
    `chunk_document`), so the flat output list is simply a concatenation —
    no chunk can span or blend two documents."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, count_tokens, config))
    return chunks
