"""Stage B router prompt — RUNTIME, no gold.

Sends the GLM router ONLY: the original question, deterministic query
features (counts/flags from `mhrag.routing.features.QueryFeatures`), and
compact retrieval diagnostics (a handful of scalars from `mhrag.routing.
features.RetrievalSignals`) — never a gold answer, evidence_list,
question_type, or oracle route label. `build_router_prompt`'s signature has
no parameter for any of those, so there is no channel to pass them through
even by mistake (see tests/test_routing_no_gold_leakage.py).
"""

from __future__ import annotations

from mhrag.routing.features import RouterFeatures

ROUTER_PROMPT_VERSION = "v1"

_SYSTEM_PROMPT_V1 = """You are a retrieval-routing assistant for a multi-hop question-answering system.

Given a question and diagnostics from a cheap initial hybrid (dense + BM25) retrieval pass, decide the MINIMUM retrieval effort needed to answer it completely:

- SIMPLE: the cheap hybrid retrieval already found all the evidence needed.
- MEDIUM: the cheap hybrid retrieval is probably incomplete, but a cross-encoder reranking pass over the same candidates would likely find the rest.
- COMPLEX: neither is likely enough — the question probably needs multiple rounds of follow-up retrieval to gather all the evidence.

Judge complexity from the question's actual content (how many distinct facts, sources, or documents it seems to require to answer completely) and from the retrieval diagnostics provided (how strong and how consistent the initial retrieval signal is) — not from surface wording alone. Note: comparison/temporal phrasing ("compare", "between X and Y", "before/after") does NOT by itself imply a harder question in this corpus — such questions often name the specific sources/claims being compared, which the initial retrieval already matches directly; questions phrased more abstractly (asking to infer or connect facts without naming the sources) tend to need more retrieval effort.

Getting this wrong has asymmetric cost: routing a question to a CHEAPER stage than it needs (under-routing) means the missing evidence is never retrieved and the final answer will likely be wrong or incomplete — this is worse than routing it to a more expensive stage than strictly necessary (over-routing), which only costs extra time and money. When you are genuinely unsure between two adjacent routes, prefer the MORE expensive one (SIMPLE vs MEDIUM -> pick MEDIUM; MEDIUM vs COMPLEX -> pick COMPLEX).

Respond with exactly one route and a short reason."""


def build_router_prompt(question: str, features: RouterFeatures, version: str = ROUTER_PROMPT_VERSION) -> tuple[str, str]:
    if version != ROUTER_PROMPT_VERSION:
        raise ValueError(f"unknown router prompt version: {version!r}")

    q = features.query
    r = features.retrieval
    diagnostics = (
        f"Query features:\n"
        f"  length_words={q.query_length_words}\n"
        f"  has_comparison_marker={q.has_comparison_marker} (count={q.comparison_marker_count})\n"
        f"  has_temporal_marker={q.has_temporal_marker} (count={q.temporal_marker_count})\n"
        f"  has_conjunction_marker={q.has_conjunction_marker} (count={q.conjunction_count})\n"
        f"  quoted_span_count={q.quoted_span_count}\n"
        f"  numeric_date_indicator_count={q.numeric_date_indicator_count}\n"
        f"\n"
        f"Initial hybrid retrieval diagnostics:\n"
        f"  top1_score={r.hybrid_top1_score:.4f}\n"
        f"  top5_mean_score={r.hybrid_top5_mean_score:.4f}\n"
        f"  score_gap_top1_top2={r.score_gap_top1_top2:.4f}\n"
        f"  score_gap_top1_top5={r.score_gap_top1_top5:.4f}\n"
        f"  dense_bm25_agreement_top10={r.dense_bm25_jaccard_top10:.2f}\n"
        f"  consensus_fraction_top5={r.consensus_fraction_top5:.2f}\n"
        f"  num_unique_docs_top5={r.num_unique_docs_top5}\n"
        f"  num_unique_docs_top10={r.num_unique_docs_top10}\n"
        f"  mean_abs_rank_diff_common_docs={r.mean_abs_rank_diff_common_docs:.2f}\n"
    )
    user_prompt = f"Question: {question}\n\n{diagnostics}"
    return _SYSTEM_PROMPT_V1, user_prompt
