"""Load raw MultiHop-RAG JSON files into validated, typed records.

Every load goes through schema validation first (see `mhrag.data.schema`) —
callers never get half-parsed data from a silently-changed upstream file.
"""

from __future__ import annotations

import json
from pathlib import Path

from mhrag.data.schema import (
    CorpusDocument,
    QARecord,
    validate_corpus_records,
    validate_qa_records,
)


def load_qa_records(path: Path) -> list[QARecord]:
    raw = json.loads(Path(path).read_text())
    validate_qa_records(raw)
    return [QARecord.from_dict(r) for r in raw]


def load_corpus(path: Path) -> list[CorpusDocument]:
    raw = json.loads(Path(path).read_text())
    validate_corpus_records(raw)
    return [CorpusDocument.from_dict(r) for r in raw]
