"""Chunking tests: deterministic ids, document mapping, no cross-document
contamination, and packing behavior.

Uses a cheap deterministic word-count token counter rather than the real
embedding model's tokenizer, so these tests stay fast and offline — the
chunking algorithm itself is what's under test, independent of tokenizer
choice. The real tokenizer is exercised by scripts/build_index.py and by
tests/test_embedding.py.
"""

from __future__ import annotations

from mhrag.data.schema import CorpusDocument
from mhrag.ingestion.chunking import ChunkingConfig, chunk_corpus, chunk_document


def _word_count(text: str) -> int:
    return len(text.split())


def _make_doc(title: str, body: str, url: str | None = None) -> CorpusDocument:
    return CorpusDocument(
        title=title,
        author="Author",
        source="Test Source",
        published_at="2024-01-01T00:00:00+00:00",
        category="technology",
        url=url or f"https://example.com/{title.replace(' ', '-').lower()}",
        body=body,
    )


def _paragraph(n_words: int, marker: str) -> str:
    return " ".join(f"{marker}{i}" for i in range(n_words))


# A document with several ~50-word paragraphs, well under max_tokens (400)
# individually, so packing must combine multiple paragraphs per chunk.
MULTI_PARAGRAPH_BODY = "\n\n".join(_paragraph(50, f"p{n}w") for n in range(10))
DOC_A = _make_doc("Document A", MULTI_PARAGRAPH_BODY)
DOC_B = _make_doc("Document B", "\n\n".join(_paragraph(50, f"q{n}w") for n in range(10)))

CONFIG = ChunkingConfig(target_tokens=150, max_tokens=200, overlap_tokens=20)


# --- deterministic chunk ids -----------------------------------------------------


def test_chunk_ids_are_deterministic_across_runs():
    a = chunk_document(DOC_A, _word_count, CONFIG)
    b = chunk_document(DOC_A, _word_count, CONFIG)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_chunk_ids_are_unique_within_a_document():
    chunks = chunk_document(DOC_A, _word_count, CONFIG)
    assert len(chunks) > 1, "test doc should produce multiple chunks"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_differ_across_documents_even_at_same_position():
    a = chunk_document(DOC_A, _word_count, CONFIG)
    b = chunk_document(DOC_B, _word_count, CONFIG)
    # position 0 of each document must not collide
    assert a[0].chunk_id != b[0].chunk_id


def test_identical_body_in_different_documents_yields_different_chunk_ids():
    """Two distinct documents that happen to share body text must still get
    distinct chunk ids (id is derived from doc_id + position + text, not
    text alone)."""
    doc_x = _make_doc("Doc X", MULTI_PARAGRAPH_BODY, url="https://example.com/x")
    doc_y = _make_doc("Doc Y", MULTI_PARAGRAPH_BODY, url="https://example.com/y")
    chunks_x = chunk_document(doc_x, _word_count, CONFIG)
    chunks_y = chunk_document(doc_y, _word_count, CONFIG)
    assert doc_x.doc_id != doc_y.doc_id  # sanity: different urls -> different doc_id
    assert chunks_x[0].chunk_id != chunks_y[0].chunk_id


# --- chunk/document mapping -------------------------------------------------------


def test_chunk_preserves_source_document_metadata():
    chunks = chunk_document(DOC_A, _word_count, CONFIG)
    for c in chunks:
        assert c.doc_id == DOC_A.doc_id
        assert c.title == DOC_A.title
        assert c.url == DOC_A.url
        assert c.source == DOC_A.source
        assert c.category == DOC_A.category
        assert c.published_at == DOC_A.published_at


def test_chunk_positions_are_sequential_from_zero():
    chunks = chunk_document(DOC_A, _word_count, CONFIG)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunk_corpus_maps_every_chunk_back_to_its_document():
    corpus = [DOC_A, DOC_B]
    chunks = chunk_corpus(corpus, _word_count, CONFIG)
    doc_ids_by_id = {d.doc_id: d for d in corpus}
    for c in chunks:
        assert c.doc_id in doc_ids_by_id
        assert c.title == doc_ids_by_id[c.doc_id].title


# --- no cross-document contamination ----------------------------------------------


def test_no_cross_document_chunk_contamination():
    """No chunk produced for DOC_A may contain any word marker unique to
    DOC_B ("q*w*"), and vice versa ("p*w*")."""
    chunks_a = chunk_document(DOC_A, _word_count, CONFIG)
    chunks_b = chunk_document(DOC_B, _word_count, CONFIG)
    for c in chunks_a:
        assert "q0w0" not in c.text
        assert all(f"q{n}w0" not in c.text for n in range(10))
    for c in chunks_b:
        assert "p0w0" not in c.text
        assert all(f"p{n}w0" not in c.text for n in range(10))


def test_chunk_corpus_chunks_never_mix_two_documents_text():
    chunks = chunk_corpus([DOC_A, DOC_B], _word_count, CONFIG)
    for c in chunks:
        if c.doc_id == DOC_A.doc_id:
            assert "q0w" not in c.text
        else:
            assert "p0w" not in c.text


def test_each_chunk_text_is_built_only_from_its_own_document_paragraphs():
    chunks = chunk_document(DOC_A, _word_count, CONFIG)
    source_paragraphs = set(DOC_A.body.split("\n\n"))
    for c in chunks:
        # every paragraph fragment in the chunk must trace back to DOC_A's body
        for fragment in c.text.split("\n\n"):
            assert fragment in source_paragraphs or fragment in DOC_A.body


# --- packing behavior --------------------------------------------------------------


def test_chunks_respect_max_tokens_ceiling():
    chunks = chunk_document(DOC_A, _word_count, CONFIG)
    for c in chunks:
        assert c.token_count <= CONFIG.max_tokens


def test_overlap_produces_shared_content_between_adjacent_chunks():
    chunks = chunk_document(DOC_A, _word_count, CONFIG)
    assert len(chunks) >= 2
    # last paragraph-unit of chunk i should reappear at the start of chunk i+1
    first_chunk_last_para = chunks[0].text.split("\n\n")[-1]
    assert first_chunk_last_para in chunks[1].text


def test_long_single_paragraph_is_sentence_split():
    long_paragraph = " ".join(f"This is sentence number {i}." for i in range(200))
    doc = _make_doc("Long Doc", long_paragraph)
    chunks = chunk_document(doc, _word_count, CONFIG)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= CONFIG.max_tokens


def test_empty_body_produces_no_chunks():
    doc = _make_doc("Empty Doc", "")
    assert chunk_document(doc, _word_count, CONFIG) == []


def test_oversized_sentence_within_a_multi_sentence_paragraph_is_still_split():
    """Regression test for a real bug found indexing the actual corpus:
    a paragraph with >1 sentence, where one individual sentence (a long,
    low-punctuation run — e.g. a "Table of Contents"-style list with no
    periods) is itself longer than max_tokens, must still be split. An
    earlier version returned oversized sentences unchecked whenever the
    paragraph had more than one sentence, producing a 717-token chunk
    against a configured 400-token ceiling."""
    short_sentence = "This is a short intro sentence."
    long_comma_list = " ".join(f"item{i}," for i in range(300))  # one giant "sentence", no periods
    paragraph = f"{short_sentence} {long_comma_list}"
    doc = _make_doc("Doc With Long List", paragraph)

    chunks = chunk_document(doc, _word_count, CONFIG)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= CONFIG.max_tokens, (
            f"chunk at position {c.position} has {c.token_count} tokens, "
            f"exceeds max_tokens={CONFIG.max_tokens}"
        )


def test_no_chunk_in_full_corpus_style_document_exceeds_max_tokens():
    """Broader sweep: many paragraphs of varying, unpredictable structure
    (mixing short sentences, long comma-lists, and normal prose) — no
    resulting chunk may exceed max_tokens, regardless of paragraph shape."""
    paragraphs = [
        "Short paragraph one. Two sentences here.",
        " ".join(f"deal{i}," for i in range(250)),  # long list, no periods
        "A normal paragraph with normal punctuation. It has a few sentences. Nothing unusual.",
        " ".join(f"Sentence {i} is fairly short." for i in range(80)),
    ]
    doc = _make_doc("Mixed Structure Doc", "\n\n".join(paragraphs))
    chunks = chunk_document(doc, _word_count, CONFIG)
    assert all(c.token_count <= CONFIG.max_tokens for c in chunks)
