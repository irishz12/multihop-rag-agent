"""RUNTIME feature-vector assembly for the learned router (Phase 8A.2).

Stage 1 uses exactly the Phase 8A `RouterFeatures` (query features + cheap
Hybrid retrieval signals — `mhrag.routing.features`, unmodified). Stage 2
uses everything Stage 1 uses PLUS `mhrag.routing.rerank_features.
RerankSignals` (reranker scores/gaps + ranking-change signals). Every
field mapped here already has NO gold field anywhere upstream (see
`mhrag.routing.features`/`mhrag.routing.rerank_features` docstrings) — this
module just flattens dataclasses into ordered `list[float]` vectors for
`mhrag.routing.learned_router`'s linear-model arithmetic.

Field order is a FROZEN CONTRACT: `STAGE1_FEATURE_NAMES`/
`STAGE2_FEATURE_NAMES` must match the order `stage1_feature_vector`/
`stage2_feature_vector` produce, 1:1 — this is what lets a persisted
`LinearModel`'s `feature_names`/`coef` stay meaningfully paired with a
freshly-computed feature vector at inference time (see
tests/test_routing_learned_features.py::
test_feature_vector_order_matches_declared_names).
"""

from __future__ import annotations

from mhrag.routing.features import RouterFeatures
from mhrag.routing.rerank_features import RerankSignals

STAGE1_FEATURE_NAMES: tuple[str, ...] = (
    # query features
    "query_length_words",
    "query_length_chars",
    "comparison_marker_count",
    "has_comparison_marker",
    "temporal_marker_count",
    "has_temporal_marker",
    "conjunction_count",
    "has_conjunction_marker",
    "quoted_span_count",
    "numeric_date_indicator_count",
    # Hybrid scores/gaps
    "hybrid_top1_score",
    "hybrid_top5_mean_score",
    "score_gap_top1_top2",
    "score_gap_top1_top5",
    # dense/BM25 overlap/agreement
    "dense_bm25_jaccard_top10",
    "consensus_fraction_top5",
    "mean_abs_rank_diff_common_docs",
    # document diversity
    "num_unique_docs_top5",
    "num_unique_docs_top10",
)

STAGE2_ADDITIONAL_FEATURE_NAMES: tuple[str, ...] = (
    "rerank_top1_score",
    "rerank_top5_mean_score",
    "rerank_score_gap_top1_top2",
    "rerank_score_gap_top1_top5",
    "rank_change_mean_abs",
    "top5_overlap_with_hybrid",
    "num_docs_new_in_rerank_top5",
)

STAGE2_FEATURE_NAMES: tuple[str, ...] = STAGE1_FEATURE_NAMES + STAGE2_ADDITIONAL_FEATURE_NAMES


def stage1_feature_vector(features: RouterFeatures) -> list[float]:
    q, r = features.query, features.retrieval
    return [
        float(q.query_length_words),
        float(q.query_length_chars),
        float(q.comparison_marker_count),
        float(q.has_comparison_marker),
        float(q.temporal_marker_count),
        float(q.has_temporal_marker),
        float(q.conjunction_count),
        float(q.has_conjunction_marker),
        float(q.quoted_span_count),
        float(q.numeric_date_indicator_count),
        float(r.hybrid_top1_score),
        float(r.hybrid_top5_mean_score),
        float(r.score_gap_top1_top2),
        float(r.score_gap_top1_top5),
        float(r.dense_bm25_jaccard_top10),
        float(r.consensus_fraction_top5),
        float(r.mean_abs_rank_diff_common_docs),
        float(r.num_unique_docs_top5),
        float(r.num_unique_docs_top10),
    ]


def stage2_feature_vector(features: RouterFeatures, rerank: RerankSignals) -> list[float]:
    return stage1_feature_vector(features) + [
        float(rerank.rerank_top1_score),
        float(rerank.rerank_top5_mean_score),
        float(rerank.rerank_score_gap_top1_top2),
        float(rerank.rerank_score_gap_top1_top5),
        float(rerank.rank_change_mean_abs),
        float(rerank.top5_overlap_with_hybrid),
        float(rerank.num_docs_new_in_rerank_top5),
    ]
