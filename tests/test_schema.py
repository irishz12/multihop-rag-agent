"""Schema validation tests, run against real (fixture) MultiHop-RAG records."""

from __future__ import annotations

import copy
import json

import pytest

from mhrag.data.schema import (
    SchemaValidationError,
    validate_corpus_records,
    validate_qa_records,
)


def _load(path):
    return json.loads(path.read_text())


def test_valid_qa_fixture_passes(sample_qa_path):
    validate_qa_records(_load(sample_qa_path))


def test_valid_corpus_fixture_passes(sample_corpus_path):
    validate_corpus_records(_load(sample_corpus_path))


def test_qa_missing_key_rejected(sample_qa_path):
    records = _load(sample_qa_path)
    del records[0]["evidence_list"]
    with pytest.raises(SchemaValidationError, match="missing keys"):
        validate_qa_records(records)


def test_qa_unknown_question_type_rejected(sample_qa_path):
    records = _load(sample_qa_path)
    records[0]["question_type"] = "not_a_real_type"
    with pytest.raises(SchemaValidationError, match="question_type"):
        validate_qa_records(records)


def test_qa_evidence_missing_key_rejected(sample_qa_path):
    records = _load(sample_qa_path)
    # find a record with at least one evidence item
    target = next(r for r in records if r["evidence_list"])
    del target["evidence_list"][0]["fact"]
    with pytest.raises(SchemaValidationError, match="evidence"):
        validate_qa_records(records)


def test_corpus_missing_key_rejected(sample_corpus_path):
    records = _load(sample_corpus_path)
    del records[0]["body"]
    with pytest.raises(SchemaValidationError, match="missing keys"):
        validate_corpus_records(records)


def test_empty_qa_rejected():
    with pytest.raises(SchemaValidationError, match="empty"):
        validate_qa_records([])


def test_empty_corpus_rejected():
    with pytest.raises(SchemaValidationError, match="empty"):
        validate_corpus_records([])


def test_null_query_may_have_no_evidence(sample_qa_path):
    """null_query records are expected to have an empty evidence_list — this
    must not be mistaken for a schema violation."""
    records = _load(sample_qa_path)
    null_queries = [r for r in records if r["question_type"] == "null_query"]
    assert null_queries, "fixture should contain at least one null_query record"
    validate_qa_records(copy.deepcopy(null_queries))
