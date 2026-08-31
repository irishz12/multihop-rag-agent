"""Gate mistake analysis tests — offline, synthetic QARecord + retrieved
chunks. Proves the false-sufficiency / false-insufficiency classification
matches ground truth Complete-Evidence@5 exactly."""

from __future__ import annotations

from mhrag.data.schema import Evidence, QARecord
from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.gate_analysis import analyze_gate_verdict


def _record(urls: list[str]) -> QARecord:
    evidence = tuple(
        Evidence(title="t", author=None, url=u, source="s", category="c",
                 published_at="2024-01-01T00:00:00+00:00", fact="f")
        for u in urls
    )
    return QARecord(query="q", answer="a", question_type="inference_query", evidence_list=evidence)


def _chunk(doc_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=1.0, method="hybrid", chunk_id=f"chunk-{doc_id}-{rank}", doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text="text", position=0,
    )


def test_correct_sufficient_when_all_gold_present_and_gate_says_sufficient():
    record = _record(["https://example.com/doc-1", "https://example.com/doc-2"])
    from mhrag.data.schema import doc_id_from_url

    d1, d2 = doc_id_from_url("https://example.com/doc-1"), doc_id_from_url("https://example.com/doc-2")
    chunks = [_chunk(d1, 1), _chunk(d2, 2), _chunk("other", 3)]
    result = analyze_gate_verdict(record, chunks, gate_sufficient=True)
    assert result.outcome == "correct_sufficient"
    assert result.ground_truth_sufficient is True
    assert result.num_gold_docs == 2
    assert result.num_gold_docs_present == 2
    assert result.num_gold_docs_missing == 0


def test_correct_insufficient_when_gold_missing_and_gate_says_insufficient():
    record = _record(["https://example.com/doc-1", "https://example.com/doc-2"])
    from mhrag.data.schema import doc_id_from_url

    d1 = doc_id_from_url("https://example.com/doc-1")
    chunks = [_chunk(d1, 1), _chunk("other-a", 2), _chunk("other-b", 3)]  # doc-2 missing
    result = analyze_gate_verdict(record, chunks, gate_sufficient=False)
    assert result.outcome == "correct_insufficient"
    assert result.ground_truth_sufficient is False
    assert result.num_gold_docs_missing == 1


def test_false_sufficiency_when_gold_missing_but_gate_says_sufficient():
    """The dangerous case — evidence is genuinely incomplete but the gate
    wrongly declared it sufficient, causing harmful under-routing."""
    record = _record(["https://example.com/doc-1", "https://example.com/doc-2"])
    from mhrag.data.schema import doc_id_from_url

    d1 = doc_id_from_url("https://example.com/doc-1")
    chunks = [_chunk(d1, 1), _chunk("other-a", 2)]  # doc-2 missing
    result = analyze_gate_verdict(record, chunks, gate_sufficient=True)
    assert result.outcome == "false_sufficiency"
    assert result.ground_truth_sufficient is False
    assert result.gate_sufficient is True
    assert result.num_gold_docs_missing == 1


def test_false_insufficiency_when_gold_present_but_gate_says_insufficient():
    """Wasteful but not harmful — evidence was actually complete, gate was
    overly cautious, causing unnecessary over-routing."""
    record = _record(["https://example.com/doc-1"])
    from mhrag.data.schema import doc_id_from_url

    d1 = doc_id_from_url("https://example.com/doc-1")
    chunks = [_chunk(d1, 1), _chunk("other", 2)]
    result = analyze_gate_verdict(record, chunks, gate_sufficient=False)
    assert result.outcome == "false_insufficiency"
    assert result.ground_truth_sufficient is True
    assert result.num_gold_docs_missing == 0


def test_gold_doc_beyond_top5_counts_as_missing():
    record = _record([f"https://example.com/doc-{i}" for i in range(2)])
    from mhrag.data.schema import doc_id_from_url

    d0 = doc_id_from_url("https://example.com/doc-0")
    d1 = doc_id_from_url("https://example.com/doc-1")
    # 6 chunks: d0 at rank1, d1 at rank6 (outside CE_K=5 window)
    chunks = [_chunk(d0, 1)] + [_chunk(f"filler-{i}", i + 2) for i in range(4)] + [_chunk(d1, 6)]
    result = analyze_gate_verdict(record, chunks, gate_sufficient=False)
    assert result.ground_truth_sufficient is False
    assert result.num_gold_docs_present == 1
    assert result.num_gold_docs_missing == 1
