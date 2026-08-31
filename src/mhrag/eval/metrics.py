"""Document-level retrieval metrics.

All metrics operate on a *unique-document ranking* — see
`collapse_to_unique_documents` — never on a raw, possibly-duplicate-per-doc
chunk ranking. Chunk-to-document dedup always happens BEFORE any metric is
computed, per the Phase 4 requirement that multiple retrieved chunks from
the same document collapse to a single document rank (the rank of that
document's best/first-appearing chunk).

Metric definitions (K = a cutoff on the unique-document ranking, 1-indexed
document positions 1..K):

- **Recall@K** = |top-K docs ∩ gold docs| / |gold docs|
  Fraction of the required evidence documents found within the top-K
  unique retrieved documents. Range [0, 1]; 1.0 only when every gold
  document is found within K.

- **Hit@K** = 1 if |top-K docs ∩ gold docs| > 0 else 0
  Whether AT LEAST ONE gold document appears within the top-K unique
  retrieved documents. Averaged over queries, this is a hit *rate*.

- **MRR@10** = 1 / rank(first gold doc in top-10 unique docs), else 0
  Reciprocal rank of the first-appearing gold document among the top-10
  unique retrieved documents. 0 if no gold document appears in the top 10.

- **NDCG@10** = DCG@10 / IDCG@10, binary relevance
  DCG@10 = sum_{i=1..10} rel_i / log2(i + 1), where rel_i = 1 if the
  document at unique-document rank i (1-indexed) is a gold document, else 0.
  IDCG@10 = the best possible DCG@10 for this question — all gold documents
  placed first: sum_{i=1..min(|gold|,10)} 1 / log2(i + 1).
  Only computed for questions with >=1 gold document (null_query is
  excluded from evaluation entirely, so IDCG is always > 0 here).

- **Complete-Evidence@K** = 1 if gold docs ⊆ top-K docs else 0
  Whether ALL required evidence documents (not just one) appear within the
  top-K unique retrieved documents — a strict bar meaningful specifically
  for multi-hop questions, where a partial evidence set cannot fully
  support a correct multi-hop answer. This is OUR OWN metric.

Alignment with the official MultiHop-RAG repo
(github.com/yixuantt/MultiHop-RAG, retrieval_evaluate.py — fetched and
inspected for this phase): the official script computes Hits@10, Hits@4,
MAP@10, and MRR@10 by substring-matching each gold evidence *fact* string
against retrieved *chunk text* (whitespace-stripped), over the top-10
retrieved chunks, skipping null_query. We align Hit@K and MRR@10's
semantics — presence-within-top-K / reciprocal-rank-of-first-hit — but
deliberately evaluate at DOCUMENT granularity, after explicit
chunk-to-document collapse matched via URL-derived doc_id, rather than
fact-substring-in-raw-chunk-text. That's what the Phase 4 spec asks for,
and it also sidesteps a real weakness of substring matching: an evidence
fact sentence can end up split across a chunk boundary or lightly reworded
during paragraph packing, silently under-counting hits at the fact level
even when the correct document was retrieved. We do not implement MAP@10
(not requested this phase). Recall@K, NDCG@10, and Complete-Evidence@K have
no equivalent in the official script — they are ours, kept clearly
distinguished from Hit@K/MRR@10 in the naming and in this docstring.
"""

from __future__ import annotations

import math

from mhrag.retrieval.schema import RetrievalResult

RECALL_KS = (4, 5, 10)
HIT_KS = (4, 10)
COMPLETE_EVIDENCE_KS = (4, 10)
MRR_K = 10
NDCG_K = 10


def collapse_to_unique_documents(results: list[RetrievalResult]) -> list[str]:
    """Collapse a rank-ordered chunk list to an ordered, deduplicated
    document-id ranking: a document's rank is the rank of its
    best (first-appearing) chunk. Trusts `results` to already be
    rank-ordered (rank 1 first) — every retrieval function in this project
    (`dense_search`, `bm25_search`, `hybrid_search`) returns results in
    that order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for r in results:
        if r.doc_id not in seen:
            seen.add(r.doc_id)
            ordered.append(r.doc_id)
    return ordered


def _require_nonempty_gold(gold: frozenset[str], metric_name: str) -> None:
    if not gold:
        raise ValueError(
            f"{metric_name} requires a non-empty gold document set "
            "(null_query has none and must be excluded before scoring)"
        )


def recall_at_k(unique_doc_ids: list[str], gold: frozenset[str], k: int) -> float:
    _require_nonempty_gold(gold, "recall_at_k")
    top_k = set(unique_doc_ids[:k])
    return len(top_k & gold) / len(gold)


def hit_at_k(unique_doc_ids: list[str], gold: frozenset[str], k: int) -> int:
    _require_nonempty_gold(gold, "hit_at_k")
    top_k = set(unique_doc_ids[:k])
    return 1 if top_k & gold else 0


def mrr_at_k(unique_doc_ids: list[str], gold: frozenset[str], k: int = MRR_K) -> float:
    _require_nonempty_gold(gold, "mrr_at_k")
    for rank, doc_id in enumerate(unique_doc_ids[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(unique_doc_ids: list[str], gold: frozenset[str], k: int = NDCG_K) -> float:
    _require_nonempty_gold(gold, "ndcg_at_k")
    dcg = 0.0
    for rank, doc_id in enumerate(unique_doc_ids[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def complete_evidence_at_k(unique_doc_ids: list[str], gold: frozenset[str], k: int) -> int:
    _require_nonempty_gold(gold, "complete_evidence_at_k")
    top_k = set(unique_doc_ids[:k])
    return 1 if gold <= top_k else 0


def compute_all_metrics(unique_doc_ids: list[str], gold: frozenset[str]) -> dict[str, float]:
    """Compute the full Phase 4 metric set for one query's unique-document
    ranking against its gold document set. Raises ValueError if `gold` is
    empty (callers must exclude null_query before calling this)."""
    metrics: dict[str, float] = {}
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = recall_at_k(unique_doc_ids, gold, k)
    for k in HIT_KS:
        metrics[f"hit@{k}"] = float(hit_at_k(unique_doc_ids, gold, k))
    for k in COMPLETE_EVIDENCE_KS:
        metrics[f"complete_evidence@{k}"] = float(complete_evidence_at_k(unique_doc_ids, gold, k))
    metrics[f"mrr@{MRR_K}"] = mrr_at_k(unique_doc_ids, gold, MRR_K)
    metrics[f"ndcg@{NDCG_K}"] = ndcg_at_k(unique_doc_ids, gold, NDCG_K)
    return metrics


def mean_metrics(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate (mean) a list of per-query metric dicts into one dict —
    used for overall, question-type, and hop-count breakdowns alike."""
    if not per_query_metrics:
        raise ValueError("mean_metrics requires at least one per-query metrics dict")
    keys = per_query_metrics[0].keys()
    n = len(per_query_metrics)
    return {key: sum(m[key] for m in per_query_metrics) / n for key in keys}
