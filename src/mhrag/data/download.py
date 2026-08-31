"""Deterministic downloader for the MultiHop-RAG dataset.

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG
Downloads the two published JSON files verbatim (no transformation) into a
local raw/ directory. Re-running is a no-op unless `force=True` — the raw
files are the reproducibility anchor, so they are content-hashed on every
run (including cache hits) rather than trusted on file presence alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
CORPUS_FILENAME = "corpus.json"
QA_FILENAME = "MultiHopRAG.json"
USER_AGENT = "agentic-multi-hop-rag/0.1"


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    num_bytes: int
    from_cache: bool


def _fetch(url: str, dest: Path, timeout: int, headers: dict[str, str]) -> DownloadResult:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    dest.write_bytes(response.content)
    digest = hashlib.sha256(response.content).hexdigest()
    return DownloadResult(
        path=dest, sha256=digest, num_bytes=len(response.content), from_cache=False
    )


def download_dataset(
    raw_dir: Path,
    base_url: str = DEFAULT_BASE_URL,
    force: bool = False,
    hf_token: str | None = None,
    timeout: int = 120,
) -> dict[str, DownloadResult]:
    """Download corpus.json and MultiHopRAG.json into `raw_dir`.

    Returns a dict keyed by {"corpus", "qa"} -> DownloadResult. Skips the
    network request (but still hashes the file on disk) when the target
    already exists and `force` is False.
    """
    headers = {"User-Agent": USER_AGENT}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    targets = {
        "corpus": (f"{base_url}/{CORPUS_FILENAME}", raw_dir / CORPUS_FILENAME),
        "qa": (f"{base_url}/{QA_FILENAME}", raw_dir / QA_FILENAME),
    }

    results: dict[str, DownloadResult] = {}
    for name, (url, dest) in targets.items():
        if dest.exists() and not force:
            data = dest.read_bytes()
            results[name] = DownloadResult(
                path=dest,
                sha256=hashlib.sha256(data).hexdigest(),
                num_bytes=len(data),
                from_cache=True,
            )
            continue
        results[name] = _fetch(url, dest, timeout=timeout, headers=headers)
    return results
