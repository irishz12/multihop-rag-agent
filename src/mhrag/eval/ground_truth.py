"""Ground-truth extraction for retrieval evaluation.

`evidence_list` is used ONLY here — never passed to a retriever, embedding
model, or index (see scripts/run_retrieval_eval.py, which loads QA records
purely for their `query` text on the retrieval side, and calls these
functions on a separate, later pass to score results already returned).
"""

from __future__ import annotations

from mhrag.data.schema import QARecord, doc_id_from_url


def gold_doc_ids(record: QARecord) -> frozenset[str]:
    """Unique set of gold document ids required to answer `record`.

    Multiple evidence facts can cite the SAME document — verified
    empirically: 168/2,255 (7.4%) of real non-null questions have at least
    one repeated evidence url. So this is deliberately a set of unique
    doc_ids derived from evidence urls, not one entry per evidence item.
    Empty for null_query (MultiHop-RAG gives null_query an empty
    evidence_list by construction).
    """
    return frozenset(doc_id_from_url(e.url) for e in record.evidence_list)


def hop_count(record: QARecord) -> int:
    """Number of distinct gold documents required — the "N-hop" grouping
    key used for the 2/3/4-hop breakdown. NOT `len(record.evidence_list)`,
    which overcounts whenever a document is cited by more than one evidence
    fact (see `gold_doc_ids`)."""
    return len(gold_doc_ids(record))
