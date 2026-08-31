"""Ground-truth extraction tests: evidence-to-document matching, dedup, and
hop counting."""

from __future__ import annotations

from mhrag.data.schema import CorpusDocument, Evidence, QARecord, doc_id_from_url
from mhrag.eval.ground_truth import gold_doc_ids, hop_count


def _evidence(url: str, fact: str = "some fact") -> Evidence:
    return Evidence(
        title="A Title",
        author="An Author",
        url=url,
        source="A Source",
        category="technology",
        published_at="2024-01-01T00:00:00+00:00",
        fact=fact,
    )


def _qa(evidence_urls: list[str], question_type: str = "inference_query") -> QARecord:
    return QARecord(
        query="a question",
        answer="an answer",
        question_type=question_type,
        evidence_list=tuple(_evidence(u) for u in evidence_urls),
    )


def _corpus_doc(url: str) -> CorpusDocument:
    return CorpusDocument(
        title="A Title",
        author="An Author",
        source="A Source",
        published_at="2024-01-01T00:00:00+00:00",
        category="technology",
        url=url,
        body="body text",
    )


# --- correct evidence-document matching -------------------------------------------


def test_gold_doc_ids_use_the_same_hash_as_corpus_doc_id():
    """The whole ground-truth-to-index mapping rests on this: an evidence
    url and an indexed document's url, if equal, MUST produce the same
    doc_id."""
    url = "https://example.com/article-1"
    record = _qa([url])
    doc = _corpus_doc(url)
    assert gold_doc_ids(record) == frozenset({doc.doc_id})
    assert doc.doc_id == doc_id_from_url(url)


def test_gold_doc_ids_distinguishes_different_urls():
    record = _qa(["https://example.com/a", "https://example.com/b"])
    ids = gold_doc_ids(record)
    assert len(ids) == 2


def test_gold_doc_ids_matches_only_the_correct_document_not_others():
    record = _qa(["https://example.com/a"])
    doc_a = _corpus_doc("https://example.com/a")
    doc_b = _corpus_doc("https://example.com/b")
    gold = gold_doc_ids(record)
    assert doc_a.doc_id in gold
    assert doc_b.doc_id not in gold


# --- chunk/evidence dedup: repeated citations of the same document ----------------


def test_gold_doc_ids_deduplicates_repeated_document_citations():
    """Real dataset fact: 7.4% of non-null questions cite the same document
    for more than one evidence fact — must collapse to one gold doc id."""
    url = "https://example.com/cited-twice"
    record = _qa([url, url])
    assert len(record.evidence_list) == 2
    assert gold_doc_ids(record) == frozenset({doc_id_from_url(url)})


def test_hop_count_counts_unique_documents_not_evidence_items():
    url = "https://example.com/cited-twice"
    record = _qa([url, url, "https://example.com/other"])
    assert len(record.evidence_list) == 3
    assert hop_count(record) == 2  # 2 unique docs, not 3 evidence items


def test_hop_count_matches_len_gold_doc_ids():
    record = _qa(["https://example.com/a", "https://example.com/b", "https://example.com/c"])
    assert hop_count(record) == len(gold_doc_ids(record)) == 3


# --- null_query handling -----------------------------------------------------------


def test_gold_doc_ids_empty_for_null_query():
    record = _qa([], question_type="null_query")
    assert gold_doc_ids(record) == frozenset()


def test_hop_count_zero_for_null_query():
    record = _qa([], question_type="null_query")
    assert hop_count(record) == 0
