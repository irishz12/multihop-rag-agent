"""Cost estimation tests: pure calculation, no client/network involved."""

from __future__ import annotations

import pytest

from mhrag.generation.cost import estimate_cost_usd
from mhrag.generation.mantle_client import MantleUsage

# Mumbai standard-tier pricing from configs/mantle.yaml, hardcoded here so
# these tests don't silently change meaning if the config file is edited.
INPUT_PRICE = 0.18
OUTPUT_PRICE = 1.41


def test_cost_calculation_with_full_usage():
    usage = MantleUsage(input_tokens=1_000_000, output_tokens=1_000_000, total_tokens=2_000_000)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd == pytest.approx(0.18)
    assert cost.output_cost_usd == pytest.approx(1.41)
    assert cost.total_cost_usd == pytest.approx(1.59)


def test_cost_calculation_scales_linearly_with_tokens():
    usage = MantleUsage(input_tokens=500_000, output_tokens=250_000, total_tokens=750_000)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd == pytest.approx(0.09)
    assert cost.output_cost_usd == pytest.approx(0.3525)
    assert cost.total_cost_usd == pytest.approx(0.4425)


def test_cost_calculation_small_realistic_token_counts():
    usage = MantleUsage(input_tokens=800, output_tokens=150, total_tokens=950)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd == pytest.approx(800 / 1_000_000 * 0.18)
    assert cost.output_cost_usd == pytest.approx(150 / 1_000_000 * 1.41)


# --- missing usage fields ---------------------------------------------------------------


def test_missing_input_tokens_yields_none_input_and_total_cost():
    usage = MantleUsage(input_tokens=None, output_tokens=100, total_tokens=None)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd is None
    assert cost.output_cost_usd is not None
    assert cost.total_cost_usd is None  # never guessed when one component is unknown


def test_missing_output_tokens_yields_none_output_and_total_cost():
    usage = MantleUsage(input_tokens=100, output_tokens=None, total_tokens=None)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd is not None
    assert cost.output_cost_usd is None
    assert cost.total_cost_usd is None


def test_entirely_missing_usage_yields_all_none():
    usage = MantleUsage(input_tokens=None, output_tokens=None, total_tokens=None)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd is None
    assert cost.output_cost_usd is None
    assert cost.total_cost_usd is None


def test_zero_tokens_yields_zero_cost_not_none():
    """Zero is a valid known value, distinct from missing (None) — a
    zero-token completion (e.g. an immediate refusal) has a real $0 cost,
    not an unknown one."""
    usage = MantleUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    cost = estimate_cost_usd(usage, INPUT_PRICE, OUTPUT_PRICE)
    assert cost.input_cost_usd == 0.0
    assert cost.output_cost_usd == 0.0
    assert cost.total_cost_usd == 0.0
