"""Typed schema for the MultiHop-RAG dataset.

Field names below were verified directly against the published dataset
(https://huggingface.co/datasets/yixuantt/MultiHopRAG, files `corpus.json`
and `MultiHopRAG.json`) by inspecting real records — not inferred from the
paper or README. See `validate_qa_records` / `validate_corpus_records`,
which re-check this assumption against every download so a future upstream
schema change fails loudly instead of silently.

Known-good example (verbatim from MultiHopRAG.json):

    {
        "query": "Who is the individual associated with the cryptocurrency
                   industry facing a criminal trial on fraud and conspiracy
                   charges, as reported by both The Verge and TechCrunch...",
        "answer": "Sam Bankman-Fried",
        "question_type": "inference_query",
        "evidence_list": [
            {
                "title": "The FTX trial is bigger than Sam Bankman-Fried",
                "author": "Elizabeth Lopatto",
                "url": "https://www.theverge.com/...",
                "source": "The Verge",
                "category": "technology",
                "published_at": "2023-09-28T12:00:00+00:00",
                "fact": "Before his fall, Bankman-Fried made himself out..."
            }
        ]
    }

Known-good example (verbatim from corpus.json):

    {
        "title": "200+ of the best deals from Amazon's Cyber Monday sale",
        "author": null,
        "source": "Mashable",
        "published_at": "2023-11-27T08:45:59+00:00",
        "category": "entertainment",
        "url": "https://mashable.com/article/cyber-monday-deals-amazon-2023",
        "body": "Table of Contents ..."
    }

Note the corpus has no explicit document id — `CorpusDocument.doc_id` derives
a stable one from the URL (documents are uniquely identified by `url` in
both the corpus and evidence_list entries).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

# question_type values observed in the dataset (Tang & Yang, 2024).
QUESTION_TYPES = frozenset(
    {
        "inference_query",
        "comparison_query",
        "temporal_query",
        "null_query",
    }
)

EXPECTED_EVIDENCE_KEYS = frozenset(
    {"title", "author", "url", "source", "category", "published_at", "fact"}
)
EXPECTED_QA_KEYS = frozenset({"query", "answer", "question_type", "evidence_list"})
EXPECTED_CORPUS_KEYS = frozenset(
    {"title", "author", "source", "published_at", "category", "url", "body"}
)


def doc_id_from_url(url: str) -> str:
    """Stable document id derived from a URL — the same hash used for every
    indexed `CorpusDocument.doc_id` (see property below). Exposed as a
    module-level function, not just the property, because Phase 4's
    evaluation ground truth needs to derive the identical id from
    `Evidence.url` (a gold citation) to match it against indexed documents —
    verified empirically that every evidence url in the real dataset's
    non-null questions equals some corpus document's url exactly (0
    mismatches across 6,084 evidence items / 2,255 non-null questions), so
    this one hash function is the whole "stable source document identity"
    mapping between ground truth and the index.
    """
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


class SchemaValidationError(ValueError):
    """Raised when downloaded data does not match the expected MultiHop-RAG schema."""


def validate_qa_records(raw: list[dict[str, Any]], sample_size: int = 100) -> None:
    """Validate the shape of raw QA records before parsing.

    Checks a leading sample (not the whole file, for speed) so a schema
    change upstream fails fast and clearly instead of surfacing as a
    confusing KeyError deep in `QARecord.from_dict`.
    """
    if not raw:
        raise SchemaValidationError("QA dataset is empty")
    for i, rec in enumerate(raw[:sample_size]):
        missing = EXPECTED_QA_KEYS - rec.keys()
        if missing:
            raise SchemaValidationError(f"QA record {i} missing keys: {sorted(missing)}")
        if rec["question_type"] not in QUESTION_TYPES:
            raise SchemaValidationError(
                f"QA record {i} has unrecognized question_type: {rec['question_type']!r} "
                f"(expected one of {sorted(QUESTION_TYPES)})"
            )
        for j, ev in enumerate(rec["evidence_list"]):
            missing_ev = EXPECTED_EVIDENCE_KEYS - ev.keys()
            if missing_ev:
                raise SchemaValidationError(
                    f"QA record {i} evidence[{j}] missing keys: {sorted(missing_ev)}"
                )


def validate_corpus_records(raw: list[dict[str, Any]], sample_size: int = 100) -> None:
    if not raw:
        raise SchemaValidationError("Corpus dataset is empty")
    for i, rec in enumerate(raw[:sample_size]):
        missing = EXPECTED_CORPUS_KEYS - rec.keys()
        if missing:
            raise SchemaValidationError(f"Corpus record {i} missing keys: {sorted(missing)}")


@dataclass(frozen=True, slots=True)
class Evidence:
    """One supporting-fact citation for a QA record.

    Ground truth only — never fed to the retrieval pipeline as input, only
    used later to score retrieval/evidence recall.
    """

    title: str
    author: str | None
    url: str
    source: str
    category: str
    published_at: str
    fact: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evidence":
        return cls(
            title=d["title"],
            author=d.get("author"),
            url=d["url"],
            source=d["source"],
            category=d["category"],
            published_at=d["published_at"],
            fact=d["fact"],
        )


@dataclass(frozen=True, slots=True)
class QARecord:
    """One question from MultiHopRAG.json, with its ground-truth answer and evidence."""

    query: str
    answer: str
    question_type: str
    evidence_list: tuple[Evidence, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QARecord":
        return cls(
            query=d["query"],
            answer=d["answer"],
            question_type=d["question_type"],
            evidence_list=tuple(Evidence.from_dict(e) for e in d["evidence_list"]),
        )


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One news article from corpus.json."""

    title: str
    author: str | None
    source: str
    published_at: str
    category: str
    url: str
    body: str

    @property
    def doc_id(self) -> str:
        """Stable id derived from the document's URL (the corpus has no native id field)."""
        return doc_id_from_url(self.url)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusDocument":
        return cls(
            title=d["title"],
            author=d.get("author"),
            source=d["source"],
            published_at=d["published_at"],
            category=d["category"],
            url=d["url"],
            body=d["body"],
        )
