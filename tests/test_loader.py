"""Loader tests, run against real (fixture) MultiHop-RAG records."""

from __future__ import annotations

from mhrag.data.loader import load_corpus, load_qa_records
from mhrag.data.schema import CorpusDocument, QARecord


def test_load_qa_records_returns_typed_records(sample_qa_path):
    records = load_qa_records(sample_qa_path)
    assert len(records) == 8
    assert all(isinstance(r, QARecord) for r in records)
    assert {r.question_type for r in records} == {
        "inference_query",
        "comparison_query",
        "temporal_query",
        "null_query",
    }


def test_load_qa_records_preserves_evidence_content(sample_qa_path):
    records = load_qa_records(sample_qa_path)
    with_evidence = [r for r in records if r.question_type != "null_query"]
    assert with_evidence, "fixture should contain records with evidence"
    for r in with_evidence:
        assert len(r.evidence_list) > 0
        for ev in r.evidence_list:
            assert ev.fact  # non-empty ground-truth fact string
            assert ev.url.startswith("http")


def test_load_corpus_returns_typed_documents(sample_corpus_path):
    docs = load_corpus(sample_corpus_path)
    assert len(docs) == 5
    assert all(isinstance(d, CorpusDocument) for d in docs)
    assert all(d.body for d in docs)


def test_corpus_doc_id_is_stable_and_unique(sample_corpus_path):
    docs = load_corpus(sample_corpus_path)
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids)), "doc_id must be unique per document"
    # same URL -> same id, deterministically, across calls
    again = load_corpus(sample_corpus_path)
    assert [d.doc_id for d in docs] == [d.doc_id for d in again]


def test_evidence_is_not_used_to_construct_query_input(sample_qa_path):
    """Ground-truth evidence must remain a separate field from the query the
    retrieval pipeline will see — this test just pins that QARecord keeps
    them as distinct attributes, not merged into `query`."""
    records = load_qa_records(sample_qa_path)
    for r in records:
        for ev in r.evidence_list:
            assert ev.fact not in r.query
