"""Sequential evidence-aware router tests — entirely offline: injected
`fused_candidates=` bypasses live Qdrant/embedding/BM25, a fake reranker
(same `.score()` duck-type as tests/test_rerank.py's `_FakeReranker`)
stands in for the real cross-encoder, and a fake Mantle client (same
pattern as tests/test_routing_evidence_gate.py) stands in for GLM.

Proves: Gate 2 runs only after Gate 1 fails, Agentic (COMPLEX) is selected
only after both gates fail, routing is deterministic with fixed responses,
and GLM token/cost tracking is correct.
"""

from __future__ import annotations

import inspect
import json

import numpy as np

from mhrag.generation.mantle_client import MantleClient
from mhrag.retrieval.schema import RetrievalResult
from mhrag.routing.sequential_router import route_question_sequential


class _FakeReranker:
    """Same duck-type as tests/test_rerank.py's _FakeReranker — `rerank_
    results` only ever calls `.score(query, texts)`."""

    def __init__(self, score_by_text: dict[str, float]):
        self._score_by_text = score_by_text
        self.score_calls: list[list[str]] = []

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        self.score_calls.append(list(texts))
        return np.array([self._score_by_text.get(t, 0.0) for t in texts])


class _FakeUsage:
    def __init__(self, prompt_tokens=30, completion_tokens=15, total_tokens=45):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeChatCompletion:
    def __init__(self, content, usage=None):
        self.choices = [_FakeChoice(content)]
        self.usage = usage or _FakeUsage()


class _ScriptedCompletions:
    def __init__(self, actions):
        self._actions = list(actions)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._actions.pop(0)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _mantle_client(actions):
    completions = _ScriptedCompletions(actions)
    client = MantleClient(
        model_id="zai.glm-4.7-flash", client=_FakeOpenAIClient(completions), max_retries=1,
        retry_base_delay_seconds=0.0,
    )
    return client, completions


def _gate_json(sufficient, supporting=None, missing=None, reason="r"):
    return json.dumps({
        "sufficient": sufficient,
        "supporting_chunk_ids": supporting or [],
        "missing_information": missing or [],
        "reason": reason,
    })


def _result(chunk_id, rank, score, text="text") -> RetrievalResult:
    return RetrievalResult(
        rank=rank, score=score, method="hybrid", chunk_id=chunk_id, doc_id=f"doc-{chunk_id}",
        title=f"Source {chunk_id}", url=f"https://example.com/{chunk_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text=text, position=0,
    )


def _fused_pool(n=20) -> list[RetrievalResult]:
    return [_result(f"c{i}", i + 1, 1.0 - i * 0.01, text=f"chunk text {i}") for i in range(n)]


# --- Gate 2 runs only after Gate 1 fails ---------------------------------------------------


def test_gate1_sufficient_routes_simple_without_calling_gate2_or_reranker():
    client, completions = _mantle_client([_FakeChatCompletion(_gate_json(True, supporting=["c0"]))])
    reranker = _FakeReranker({})
    result = route_question_sequential(
        "q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool(),
    )
    assert result.route == "SIMPLE"
    assert len(completions.calls) == 1  # only Gate 1 called
    assert reranker.score_calls == []  # reranker never invoked
    assert result.gate2_result is None
    assert result.reranked_top5 is None
    assert result.num_glm_calls == 1


def test_gate1_insufficient_triggers_reranker_and_gate2():
    client, completions = _mantle_client(
        [_FakeChatCompletion(_gate_json(False, missing=["m"])), _FakeChatCompletion(_gate_json(True, supporting=["c0"]))]
    )
    reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
    result = route_question_sequential(
        "q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool(),
    )
    assert len(completions.calls) == 2  # Gate 1 AND Gate 2 called
    assert len(reranker.score_calls) == 1
    assert result.gate2_result is not None
    assert result.reranked_top5 is not None
    assert len(result.reranked_top5) == 5


# --- Agentic (COMPLEX) selected only after both gates fail ---------------------------------


def test_both_gates_insufficient_routes_complex():
    client, _ = _mantle_client(
        [_FakeChatCompletion(_gate_json(False, missing=["a"])), _FakeChatCompletion(_gate_json(False, missing=["b"]))]
    )
    reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
    result = route_question_sequential(
        "q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool(),
    )
    assert result.route == "COMPLEX"


def test_gate1_insufficient_gate2_sufficient_routes_medium():
    client, _ = _mantle_client(
        [_FakeChatCompletion(_gate_json(False, missing=["a"])), _FakeChatCompletion(_gate_json(True, supporting=["c0"]))]
    )
    reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
    result = route_question_sequential(
        "q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool(),
    )
    assert result.route == "MEDIUM"


# --- GLM cost/token tracking ----------------------------------------------------------------


def test_single_gate_call_glm_tracking():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True), usage=_FakeUsage(100, 20, 120))])
    reranker = _FakeReranker({})
    result = route_question_sequential(
        "q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool(),
        glm_input_price_per_million=0.08, glm_output_price_per_million=0.48,
    )
    assert result.num_glm_calls == 1
    assert result.glm_input_tokens == 100
    assert result.glm_output_tokens == 20
    assert result.glm_cost is not None
    assert result.glm_cost.total_cost_usd == (100 / 1_000_000) * 0.08 + (20 / 1_000_000) * 0.48


def test_two_gate_calls_glm_tokens_summed():
    client, _ = _mantle_client([
        _FakeChatCompletion(_gate_json(False, missing=["m"]), usage=_FakeUsage(100, 20, 120)),
        _FakeChatCompletion(_gate_json(True, supporting=["c0"]), usage=_FakeUsage(150, 25, 175)),
    ])
    reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
    result = route_question_sequential("q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool())
    assert result.num_glm_calls == 2
    assert result.glm_input_tokens == 250
    assert result.glm_output_tokens == 45


def test_no_cost_computed_when_pricing_not_provided():
    client, _ = _mantle_client([_FakeChatCompletion(_gate_json(True))])
    reranker = _FakeReranker({})
    result = route_question_sequential("q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool())
    assert result.glm_cost is None


# --- determinism ----------------------------------------------------------------------------


def test_repeated_routing_deterministic_with_fixed_responses():
    outcomes = []
    for _ in range(3):
        client, _ = _mantle_client(
            [_FakeChatCompletion(_gate_json(False, missing=["m"])), _FakeChatCompletion(_gate_json(True, supporting=["c0"]))]
        )
        reranker = _FakeReranker({f"chunk text {i}": float(20 - i) for i in range(20)})
        result = route_question_sequential("q", None, "c", None, None, reranker, client, fused_candidates=_fused_pool())
        outcomes.append((result.route, result.num_glm_calls, result.gate1_result.decision, result.gate2_result.decision))
    assert all(o == outcomes[0] for o in outcomes)


# --- structural: no gold, no heuristic/oracle/gate_analysis imports ------------------------


def test_route_question_sequential_signature_has_no_gold_parameter():
    params = list(inspect.signature(route_question_sequential).parameters)
    forbidden = {"answer", "evidence_list", "question_type", "gold", "oracle_route", "oracle_label", "complete_evidence"}
    assert not (forbidden & set(params))


def test_sequential_router_never_imports_oracle_gate_analysis_tune_thresholds_or_heuristic():
    import re
    from pathlib import Path

    path = Path(__file__).parent.parent / "src" / "mhrag" / "routing" / "sequential_router.py"
    source = path.read_text()
    assert not re.search(
        r"^\s*(import|from)\s+.*\b(oracle|gate_analysis|tune_thresholds|heuristic)\b", source, re.MULTILINE
    )
