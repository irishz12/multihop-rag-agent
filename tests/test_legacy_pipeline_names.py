"""mhrag.eval.legacy_pipeline_names is the ONLY place this project
translates between canonical pipeline names and the legacy names baked
into already-frozen results/*.json artifacts. These tests pin that
mapping and confirm rekeying never touches anything but the two renamed
top-level keys.
"""

from __future__ import annotations

from mhrag.eval.legacy_pipeline_names import (
    CANONICAL_TO_LEGACY,
    LEGACY_TO_CANONICAL,
    get_quality_retention_pct,
    rekey_legacy_prefixed_keys,
    rekey_legacy_report,
    to_canonical_name,
    to_legacy_name,
)


def test_canonical_to_legacy_mapping_is_exactly_the_two_renamed_pipelines():
    assert CANONICAL_TO_LEGACY == {"agentic_multi_hop": "always_agentic", "adaptive_rag": "adaptive"}
    assert LEGACY_TO_CANONICAL == {"always_agentic": "agentic_multi_hop", "adaptive": "adaptive_rag"}


def test_to_legacy_name_translates_the_two_renamed_pipelines():
    assert to_legacy_name("agentic_multi_hop") == "always_agentic"
    assert to_legacy_name("adaptive_rag") == "adaptive"


def test_to_canonical_name_translates_the_two_legacy_names():
    assert to_canonical_name("always_agentic") == "agentic_multi_hop"
    assert to_canonical_name("adaptive") == "adaptive_rag"


def test_unrenamed_pipeline_names_pass_through_unchanged():
    for name in ("dense", "hybrid", "hybrid_reranker"):
        assert to_legacy_name(name) == name
        assert to_canonical_name(name) == name


def test_round_trip_is_identity():
    for canonical in CANONICAL_TO_LEGACY:
        assert to_canonical_name(to_legacy_name(canonical)) == canonical


def test_rekey_legacy_report_only_touches_the_two_legacy_keys():
    report = {
        "dense": {"quality": 0.5},
        "hybrid": {"quality": 0.55},
        "always_agentic": {"quality": 0.7},
        "adaptive": {"quality": 0.56},
        "generated_at": "2026-08-29T09:03:37.867435+00:00",
    }
    rekeyed = rekey_legacy_report(report)
    assert rekeyed == {
        "dense": {"quality": 0.5},
        "hybrid": {"quality": 0.55},
        "agentic_multi_hop": {"quality": 0.7},
        "adaptive_rag": {"quality": 0.56},
        "generated_at": "2026-08-29T09:03:37.867435+00:00",
    }
    # every value object is passed through by identity, never copied/mutated
    assert rekeyed["agentic_multi_hop"] is report["always_agentic"]
    assert rekeyed["adaptive_rag"] is report["adaptive"]


def test_rekey_legacy_report_is_a_no_op_on_an_already_canonical_report():
    report = {"dense": {"quality": 0.5}, "agentic_multi_hop": {"quality": 0.7}}
    assert rekey_legacy_report(report) == report


def test_rekey_legacy_prefixed_keys_rewrites_only_the_legacy_prefixes():
    cost_latency = {
        "always_agentic_mean_cost_usd": 0.00116,
        "adaptive_mean_cost_usd": 0.00091,
        "cost_reduction_pct": 0.2176,
    }
    assert rekey_legacy_prefixed_keys(cost_latency) == {
        "agentic_multi_hop_mean_cost_usd": 0.00116,
        "adaptive_rag_mean_cost_usd": 0.00091,
        "cost_reduction_pct": 0.2176,
    }


def test_rekey_legacy_prefixed_keys_does_not_touch_dense_hybrid_keys():
    breakdown_group = {
        "always_agentic_mean_quality": 0.7,
        "adaptive_mean_quality": 0.56,
        "n": 16,
    }
    rekeyed = rekey_legacy_prefixed_keys(breakdown_group)
    assert rekeyed["n"] == 16
    assert set(rekeyed) == {"agentic_multi_hop_mean_quality", "adaptive_rag_mean_quality", "n"}


def test_get_quality_retention_pct_reads_the_compound_legacy_key():
    report = {"adaptive_quality_retention_pct_vs_always_agentic": 0.8}
    assert get_quality_retention_pct(report) == 0.8
