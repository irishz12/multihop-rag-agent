"""USD cost estimation from Mantle token usage.

Pure calculation, no I/O — pricing values are passed in (read from
`configs/mantle.yaml: pricing`, kept in config with source/date metadata so
they can be updated without a code change; see that file). Deliberately
does not include any offline LLM-judge cost — this module only ever prices
the actual generation call it's given usage for.
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.generation.mantle_client import MantleUsage


@dataclass(frozen=True, slots=True)
class CostEstimate:
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None


def estimate_cost_usd(
    usage: MantleUsage,
    input_price_per_million: float,
    output_price_per_million: float,
) -> CostEstimate:
    """Estimate USD cost from token usage and per-million-token pricing.

    Returns `None` for any component whose required token count is missing
    (e.g. the backend omitted `usage` entirely) — cost is never silently
    guessed or defaulted to zero when the real figure is unknown.
    """
    input_cost = (
        (usage.input_tokens / 1_000_000) * input_price_per_million
        if usage.input_tokens is not None
        else None
    )
    output_cost = (
        (usage.output_tokens / 1_000_000) * output_price_per_million
        if usage.output_tokens is not None
        else None
    )
    total_cost = (
        input_cost + output_cost if input_cost is not None and output_cost is not None else None
    )
    return CostEstimate(
        input_cost_usd=input_cost, output_cost_usd=output_cost, total_cost_usd=total_cost
    )
