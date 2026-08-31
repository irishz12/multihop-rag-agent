"""Evidence-grounded answer generation.

    question + retrieved context -> Mantle -> answer

Never sends ground-truth `answer`, `evidence_list`, or `question_type` to
the model — `generate_answer`'s signature only accepts a `question: str`
and `retrieved: list[RetrievalResult]` (which structurally cannot carry
ground truth, see `mhrag.retrieval.schema`), so there is no parameter a
caller could even attempt to pass gold data through. Source chunk/doc ids
used to build the context are retained on `GenerationResult.context` for
evaluation/debugging — never shown to the model, never added as citations
to the answer text itself (see `mhrag.generation.prompts`, which instructs
the model not to include citations).
"""

from __future__ import annotations

from dataclasses import dataclass

from mhrag.generation.context import AssembledContext, TokenCounter, assemble_context
from mhrag.generation.cost import CostEstimate, estimate_cost_usd
from mhrag.generation.mantle_client import MantleClient, MantleResponse
from mhrag.generation.prompts import PROMPT_VERSION, build_prompt
from mhrag.retrieval.schema import RetrievalResult


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    context: AssembledContext
    mantle_response: MantleResponse
    cost: CostEstimate
    prompt_version: str


def generate_answer(
    question: str,
    retrieved: list[RetrievalResult],
    client: MantleClient,
    count_tokens: TokenCounter,
    top_k: int,
    max_context_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
    prompt_version: str = PROMPT_VERSION,
) -> GenerationResult:
    context = assemble_context(
        retrieved, count_tokens, top_k=top_k, max_context_tokens=max_context_tokens
    )
    system_prompt, user_prompt = build_prompt(question, context.context_text, version=prompt_version)
    response = client.complete(system_prompt, user_prompt)
    cost = estimate_cost_usd(response.usage, input_price_per_million, output_price_per_million)
    return GenerationResult(
        answer=response.text,
        context=context,
        mantle_response=response,
        cost=cost,
        prompt_version=prompt_version,
    )
