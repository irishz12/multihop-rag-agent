"""The ONE place this project translates between its current canonical
pipeline names and the legacy names baked into already-frozen Phase 9
result artifacts under `results/` (raw traces, judge scores, and the
aggregate reports built from them, e.g. `phase9_holdout_report.json`).

Those artifacts are immutable — renaming their filenames or internal JSON
keys would invalidate their provenance — so anything that still needs to
open them by their original on-disk names, or read their original JSON
keys, does it through this module and nowhere else. Every other module in
this codebase (scripts, tests, the frontend, the backend) refers to these
two pipelines only by their canonical name:

    agentic_multi_hop -> always_agentic   (the bounded agent loop, always run)
    adaptive_rag       -> adaptive         (the learned-router cost-aware baseline)

`dense`, `hybrid`, and `hybrid_reranker` were never renamed and pass
through this module unchanged.
"""

from __future__ import annotations

CANONICAL_TO_LEGACY: dict[str, str] = {
    "agentic_multi_hop": "always_agentic",
    "adaptive_rag": "adaptive",
}
LEGACY_TO_CANONICAL: dict[str, str] = {legacy: canonical for canonical, legacy in CANONICAL_TO_LEGACY.items()}


def to_legacy_name(canonical_name: str) -> str:
    """Canonical pipeline name -> the name already baked into frozen
    `results/*.json` filenames and JSON keys (returned unchanged if
    `canonical_name` isn't one of the two renamed pipelines)."""
    return CANONICAL_TO_LEGACY.get(canonical_name, canonical_name)


def to_canonical_name(name: str) -> str:
    """A name read from a frozen `results/*.json` filename or JSON key ->
    this project's current canonical pipeline name (returned unchanged if
    `name` is already canonical, or isn't one of the two renamed
    pipelines)."""
    return LEGACY_TO_CANONICAL.get(name, name)


def rekey_legacy_report(report: dict) -> dict:
    """Rekey a dict directly keyed by pipeline name at its top level (e.g.
    `phase9_holdout_report.json["combined_quality_mean"]`) so `always_agentic`
    and `adaptive` become their canonical names. Every other key, and every
    value, passes through completely unchanged — this never touches a
    measured number, only how this codebase refers to the pipeline that
    produced it."""
    return {to_canonical_name(key): value for key, value in report.items()}


def rekey_legacy_prefixed_keys(d: dict) -> dict:
    """Rekey a flat dict whose keys are PREFIXED with a pipeline name (e.g.
    `phase9_holdout_report.json["cost_latency"]`, whose keys look like
    `always_agentic_mean_cost_usd`, or one hop/query-type breakdown group,
    whose keys look like `adaptive_mean_quality`) so any key beginning with
    `always_agentic_` or `adaptive_` is rewritten to begin with the
    canonical name instead. Keys with no legacy prefix pass through
    unchanged."""
    rekeyed = {}
    for key, value in d.items():
        for legacy, canonical in LEGACY_TO_CANONICAL.items():
            prefix = f"{legacy}_"
            if key.startswith(prefix):
                key = f"{canonical}_{key[len(prefix):]}"
                break
        rekeyed[key] = value
    return rekeyed


# `phase9_sample_report.json` / `phase9_holdout_report.json`'s one compound
# key naming the Adaptive-vs-Agentic quality-retention ratio — a one-off
# that doesn't fit either rekeying helper above, so it gets its own named
# accessor rather than a third generic pattern.
_RETENTION_KEY = "adaptive_quality_retention_pct_vs_always_agentic"


def get_quality_retention_pct(report: dict) -> float:
    """Adaptive RAG's quality retention (0-1) relative to Agentic Multi-Hop
    RAG, read from the frozen report's one compound-named legacy key."""
    return report[_RETENTION_KEY]
