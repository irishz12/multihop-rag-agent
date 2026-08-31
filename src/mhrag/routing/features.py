"""RUNTIME router feature extraction — no gold fields, ever.

Two feature families, matching the Phase 8A spec exactly:

  - `QueryFeatures` (`extract_query_features`): deterministic, regex/word-
    list based signals from the raw question TEXT alone. No retrieval, no
    model call.
  - `RetrievalSignals` (`extract_retrieval_signals`): deterministic signals
    derived from the frozen Hybrid RRF baseline's OWN output (dense-only,
    BM25-only, and fused results) — no reranker score, no agentic trace, no
    gold document id. `compute_router_features` is the live wrapper that
    actually calls the frozen, unmodified `dense_search`/`bm25_search`/
    `rrf_fuse` (Phase 2-4.1) to get those three lists; `extract_retrieval_
    signals` itself is a pure function of already-computed `RetrievalResult`
    lists, injectable for offline testing (see tests/test_routing_features.py)
    exactly like `mhrag.agent.loop`'s `hop_runner` pattern.

Structural guarantee: `extract_query_features` takes only `question: str`;
`extract_retrieval_signals` takes only `RetrievalResult` lists;
`compute_router_features` takes only `question` plus retrieval
infrastructure (client/models) — none of the three has a parameter for
gold answer, evidence_list, question_type, or an oracle route label, so
there is no channel through which evaluator-only data could reach the
runtime router even by mistake. See tests/test_routing_no_gold_leakage.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from qdrant_client import QdrantClient

from mhrag.eval.metrics import collapse_to_unique_documents
from mhrag.ingestion.bm25 import Bm25Model
from mhrag.ingestion.embedding import EmbeddingModel
from mhrag.retrieval.bm25 import bm25_search
from mhrag.retrieval.dense import dense_search
from mhrag.retrieval.rrf import RRF_K, rrf_fuse
from mhrag.retrieval.schema import RetrievalResult

# --- query text features ----------------------------------------------------------------

# Hand-picked, documented word lists — deliberately simple/transparent
# (the spec asks for interpretable features, not a learned text classifier).
# Matched case-insensitively as whole words.
COMPARISON_MARKERS = frozenset(
    {
        "compare", "compared", "comparing", "comparison", "versus", "vs",
        "both", "while", "whereas", "unlike", "similarly", "similar",
        "differ", "differs", "differing", "difference", "consistent",
        "consistency", "inconsistent", "inconsistency", "contrast",
    }
)
TEMPORAL_MARKERS = frozenset(
    {
        "before", "after", "between", "since", "until", "during",
        "earlier", "later", "previously", "subsequently", "then",
        "when", "timeline", "over time", "change", "changed", "shift",
        "shifted", "evolve", "evolved", "evolution",
    }
)
CONJUNCTION_MARKERS = frozenset({"and", "or", "as well as", "along with"})

_WORD_RE = re.compile(r"[a-zA-Z']+")
_QUOTED_SPAN_RE = re.compile(r"'[^']{2,80}'|\"[^\"]{2,80}\"")
# Explicit dates ("October 7, 2023"), bare 4-digit years, and month names —
# all deterministic, no NER/model involved.
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTH_NAMES) + r")\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\b\d+(st|nd|rd|th)?\b")


def _count_markers(words: list[str], markers: frozenset[str]) -> int:
    return sum(1 for w in words if w in markers)


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    query_length_words: int
    query_length_chars: int
    comparison_marker_count: int
    has_comparison_marker: bool
    temporal_marker_count: int
    has_temporal_marker: bool
    conjunction_count: int
    has_conjunction_marker: bool
    quoted_span_count: int
    numeric_date_indicator_count: int


def extract_query_features(question: str) -> QueryFeatures:
    """Pure function of the raw question text — no retrieval, no model
    call, no gold field of any kind."""
    lowered = question.lower()
    words = _WORD_RE.findall(lowered)

    comparison_count = _count_markers(words, COMPARISON_MARKERS)
    temporal_count = _count_markers(words, TEMPORAL_MARKERS) + lowered.count("over time")
    conjunction_count = sum(lowered.count(m) for m in CONJUNCTION_MARKERS if " " in m) + _count_markers(
        words, frozenset({"and", "or"})
    )
    quoted_span_count = len(_QUOTED_SPAN_RE.findall(question))
    numeric_date_count = len(_YEAR_RE.findall(question)) + len(_MONTH_RE.findall(question)) + len(
        _NUMERIC_RE.findall(question)
    )

    return QueryFeatures(
        query_length_words=len(words),
        query_length_chars=len(question),
        comparison_marker_count=comparison_count,
        has_comparison_marker=comparison_count > 0,
        temporal_marker_count=temporal_count,
        has_temporal_marker=temporal_count > 0,
        conjunction_count=conjunction_count,
        has_conjunction_marker=conjunction_count > 0,
        quoted_span_count=quoted_span_count,
        numeric_date_indicator_count=numeric_date_count,
    )


# --- cheap Hybrid retrieval signals ------------------------------------------------------

# How many fused hybrid positions we look at for feature engineering —
# deeper than the production final_top_k=5 (which is what a SIMPLE-routed
# question actually keeps), matching the same pattern already used
# elsewhere in this project for evaluation-only pool depth (e.g.
# scripts/run_retrieval_eval.py's RAW_CANDIDATE_POOL_SIZE=50): a bigger
# `final_top_k` passed to the SAME unmodified `rrf_fuse` is not a change to
# RRF, just how much of its output we choose to look at.
FEATURE_HYBRID_DEPTH = 10
FEATURE_TOP_K_FOR_AGREEMENT = 10


@dataclass(frozen=True, slots=True)
class RetrievalSignals:
    hybrid_top1_score: float
    hybrid_top5_mean_score: float
    score_gap_top1_top2: float
    score_gap_top1_top5: float
    dense_bm25_jaccard_top10: float
    consensus_fraction_top5: float
    num_unique_docs_top5: int
    num_unique_docs_top10: int
    mean_abs_rank_diff_common_docs: float


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def extract_retrieval_signals(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    hybrid_results: list[RetrievalResult],
) -> RetrievalSignals:
    """Pure function of three already-computed `RetrievalResult` lists —
    dense-only, BM25-only, and fused Hybrid RRF (all from the frozen,
    unmodified pipeline). No gold document id, no reranker score, no
    live call happens here — inject fixtures/fakes to test this function
    offline (see tests/test_routing_features.py)."""
    dense_docs_top10 = set(collapse_to_unique_documents(dense_results[:FEATURE_TOP_K_FOR_AGREEMENT]))
    bm25_docs_top10 = set(collapse_to_unique_documents(bm25_results[:FEATURE_TOP_K_FOR_AGREEMENT]))

    hybrid_unique_docs = collapse_to_unique_documents(hybrid_results)
    top5_unique_docs = hybrid_unique_docs[:5]
    top10_unique_docs = hybrid_unique_docs[:10]

    top5_scores = [r.score for r in hybrid_results[:5]]
    hybrid_top1_score = hybrid_results[0].score if hybrid_results else 0.0
    hybrid_top5_mean_score = sum(top5_scores) / len(top5_scores) if top5_scores else 0.0
    score_gap_top1_top2 = (
        hybrid_results[0].score - hybrid_results[1].score if len(hybrid_results) >= 2 else 0.0
    )
    score_gap_top1_top5 = (
        hybrid_results[0].score - top5_scores[-1] if len(top5_scores) >= 2 else 0.0
    )

    dense_bm25_jaccard = _jaccard(dense_docs_top10, bm25_docs_top10)

    consensus_docs = set(top5_unique_docs) & dense_docs_top10 & bm25_docs_top10
    consensus_fraction_top5 = len(consensus_docs) / len(top5_unique_docs) if top5_unique_docs else 0.0

    dense_rank = {doc_id: i + 1 for i, doc_id in enumerate(collapse_to_unique_documents(dense_results))}
    bm25_rank = {doc_id: i + 1 for i, doc_id in enumerate(collapse_to_unique_documents(bm25_results))}
    common_docs = set(dense_rank) & set(bm25_rank)
    mean_abs_rank_diff = (
        sum(abs(dense_rank[d] - bm25_rank[d]) for d in common_docs) / len(common_docs)
        if common_docs
        else 0.0
    )

    return RetrievalSignals(
        hybrid_top1_score=hybrid_top1_score,
        hybrid_top5_mean_score=hybrid_top5_mean_score,
        score_gap_top1_top2=score_gap_top1_top2,
        score_gap_top1_top5=score_gap_top1_top5,
        dense_bm25_jaccard_top10=dense_bm25_jaccard,
        consensus_fraction_top5=consensus_fraction_top5,
        num_unique_docs_top5=len(top5_unique_docs),
        num_unique_docs_top10=len(top10_unique_docs),
        mean_abs_rank_diff_common_docs=mean_abs_rank_diff,
    )


@dataclass(frozen=True, slots=True)
class RouterFeatures:
    query: QueryFeatures
    retrieval: RetrievalSignals


def compute_router_features(
    question: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: EmbeddingModel,
    bm25_model: Bm25Model,
    dense_top_k: int = 20,
    bm25_top_k: int = 20,
) -> RouterFeatures:
    """Live wrapper: calls the frozen, unmodified `dense_search`/
    `bm25_search`/`rrf_fuse` (production `dense_top_k`/`bm25_top_k`/
    `k=RRF_K` from configs/retrieval.yaml) to get the three result lists,
    then derives `RouterFeatures` from them. `question` is the ONLY
    ground-truth-adjacent input — no gold field is used or accepted."""
    dense_results = dense_search(question, qdrant_client, collection_name, embedding_model, top_k=dense_top_k)
    bm25_results = bm25_search(question, qdrant_client, collection_name, bm25_model, top_k=bm25_top_k)
    hybrid_results = rrf_fuse(dense_results, bm25_results, k=RRF_K, final_top_k=FEATURE_HYBRID_DEPTH)

    return RouterFeatures(
        query=extract_query_features(question),
        retrieval=extract_retrieval_signals(dense_results, bm25_results, hybrid_results),
    )
