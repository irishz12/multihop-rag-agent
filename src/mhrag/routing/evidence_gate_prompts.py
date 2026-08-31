"""Evidence Sufficiency Gate prompt — RUNTIME, no gold.

Sends the gate ONLY: the original question, and the actual retrieved
chunks (id, source title, rank, retrieval score, chunk text) — the same
fields already public on any `RetrievalResult`. Never a gold answer,
evidence_list, expected documents, oracle route, question_type, or
Complete-Evidence result. `build_gate_prompt`'s signature has no parameter
for any of those, so there is no channel to pass them through even by
mistake (see tests/test_routing_no_gold_leakage.py).

The gate is explicitly NOT asked to propose a follow-up search query —
that responsibility stays with the Agentic controller (`mhrag.agent.
controller`, Phase 7); this prompt only ever asks a yes/no sufficiency
question.
"""

from __future__ import annotations

from dataclasses import dataclass

EVIDENCE_GATE_PROMPT_VERSION = "v1"

_SYSTEM_PROMPT_V1 = """You are an evidence-sufficiency judge for a multi-hop question-answering system.

You will be given a question and a small set of retrieved text chunks (each with a chunk id, source title, retrieval rank, and retrieval score). Decide whether these chunks, TAKEN TOGETHER, contain ALL the information needed to answer the question completely and correctly — not just information that is topically related or partially relevant.

Be conservative: if you are unsure whether something important is still missing, judge the evidence as NOT sufficient. It is much better to wrongly say "not sufficient" when it actually is, than to wrongly say "sufficient" when something important is missing — a wrong "sufficient" verdict cannot be corrected later, while a wrong "not sufficient" verdict only costs a bit of extra retrieval effort.

Do NOT propose a follow-up search query or suggest what to search for next — only judge whether the evidence given to you is enough, as-is.

For "supporting_chunk_ids", list only the ids of chunks (from the ones given to you) that directly support the answer — never invent an id that was not given to you. For "missing_information", if the evidence is not fully sufficient, briefly name what specific information is still needed; leave it empty only if you are confident nothing is missing."""


@dataclass(frozen=True, slots=True)
class GateChunkInput:
    chunk_id: str
    title: str
    text: str
    rank: int
    score: float


def build_gate_prompt(
    question: str,
    chunks: list[GateChunkInput],
    version: str = EVIDENCE_GATE_PROMPT_VERSION,
) -> tuple[str, str]:
    if version != EVIDENCE_GATE_PROMPT_VERSION:
        raise ValueError(f"unknown evidence gate prompt version: {version!r}")

    chunks_text = "\n\n".join(
        f"[chunk_id={c.chunk_id}] (source: {c.title!r}, rank={c.rank}, score={c.score:.4f})\n{c.text}"
        for c in chunks
    )
    user_prompt = f"Question: {question}\n\nRetrieved chunks:\n\n{chunks_text}"
    return _SYSTEM_PROMPT_V1, user_prompt
