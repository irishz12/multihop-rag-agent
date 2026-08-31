from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_corpus_path() -> Path:
    return FIXTURES_DIR / "sample_corpus.json"


@pytest.fixture
def sample_qa_path() -> Path:
    return FIXTURES_DIR / "sample_qa.json"
