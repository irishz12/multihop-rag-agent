"""Calibration sweep tests — offline, using Phase 7's fake hop_runner /
fake Mantle client pattern (tests/test_agent_loop.py). Proves: swept
configs differ ONLY in max_context_tokens, gold evidence stays
evaluator-only (never touches the agent), and evidence evaluation/new-doc
tracking is computed correctly.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.calibration.sweep import (
    build_swept_configs,
    evaluate_against_gold,
    new_unique_docs_per_hop,
    run_calibration_query,
)
from mhrag.data.schema import Evidence, QARecord, doc_id_from_url
from mhrag.generation.mantle_client import MantleClient
from mhrag.retrieval.schema import RetrievalResult

# --- fakes (same pattern as tests/test_agent_loop.py) -----------------------------------


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, total_tokens=15):
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


def _mantle_client(actions) -> tuple[MantleClient, _ScriptedCompletions]:
    completions = _ScriptedCompletions(actions)
    client = MantleClient(
        model_id="test-model", client=_FakeOpenAIClient(completions), max_retries=1, retry_base_delay_seconds=0.0
    )
    return client, completions


def _controller_json(sufficient, next_query, reason="r"):
    return json.dumps({"sufficient": sufficient, "next_query": next_query, "reason": reason})


def _result(chunk_id: str, doc_id: str, text: str = "chunk text") -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=1.0, method="hybrid_reranked", chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text=text, position=0,
    )


class _ScriptedHopRunner:
    def __init__(self, results_sequence):
        self._sequence = list(results_sequence)
        self.calls: list[str] = []

    def __call__(self, query):
        self.calls.append(query)
        return self._sequence[len(self.calls) - 1], 10.0, 5.0


# --- swept configs differ only in max_context_tokens -------------------------------------


def test_swept_configs_differ_only_in_max_context_tokens():
    base = AgenticConfig()
    swept = build_swept_configs(base, [3000, 4500, 6000])
    assert set(swept) == {3000, 4500, 6000}
    for budget, cfg in swept.items():
        assert cfg.max_context_tokens == budget
        replaced_back = dataclasses.replace(cfg, max_context_tokens=base.max_context_tokens)
        assert replaced_back == base


def test_swept_configs_keep_frozen_fields_from_base():
    base = AgenticConfig(max_hops=3, hop_top_k=5, timeout_seconds=30.0)
    swept = build_swept_configs(base, [3000, 6000])
    for cfg in swept.values():
        assert cfg.max_hops == 3
        assert cfg.hop_top_k == 5
        assert cfg.timeout_seconds == 30.0
        assert cfg.controller_prompt_version == base.controller_prompt_version
        assert cfg.generation_prompt_version == base.generation_prompt_version


# --- new_unique_docs_per_hop ---------------------------------------------------------------


def test_new_unique_docs_per_hop_counts_documents_not_chunks():
    hop_results = [
        [_result("a", "doc-1"), _result("b", "doc-1")],  # 2 chunks, SAME doc -> 1 new doc at hop1
        [_result("c", "doc-2"), _result("d", "doc-1")],  # "d" is a NEW chunk but doc-1 already seen
    ]
    controller_client, _ = _mantle_client(
        [_FakeChatCompletion(_controller_json(False, "follow up")), _FakeChatCompletion(_controller_json(True, None))]
    )
    generation_client, _ = _mantle_client([_FakeChatCompletion("answer")])
    trace = run_agentic_retrieval(
        "q", None, "collection", None, None, None, controller_client, generation_client,
        hop_runner=_ScriptedHopRunner(hop_results),
    )
    gains = new_unique_docs_per_hop(trace)
    assert gains == (1, 1)  # hop1: doc-1 (1 new doc despite 2 chunks); hop2: doc-2 only (doc-1 already seen)


def test_new_unique_docs_per_hop_is_zero_when_all_new_chunks_are_from_seen_docs():
    hop_results = [[_result("a", "doc-1")], [_result("b", "doc-1")]]  # hop2's new chunk is from doc-1 again
    controller_client, _ = _mantle_client(
        [_FakeChatCompletion(_controller_json(False, "follow up")), _FakeChatCompletion(_controller_json(True, None))]
    )
    generation_client, _ = _mantle_client([_FakeChatCompletion("answer")])
    trace = run_agentic_retrieval(
        "q", None, "collection", None, None, None, controller_client, generation_client,
        hop_runner=_ScriptedHopRunner(hop_results),
    )
    assert new_unique_docs_per_hop(trace) == (1, 0)


# --- gold evidence stays evaluator-only ---------------------------------------------------


def test_evaluate_against_gold_signature_only_takes_record_and_trace():
    """Structural: evaluate_against_gold is the only place gold touches
    anything, and it only accepts a completed trace — there's no parameter
    through which gold could reach the agent itself."""
    params = list(inspect.signature(evaluate_against_gold).parameters)
    assert params == ["record", "trace"]


def test_evaluate_against_gold_computes_recall_and_complete_evidence():
    record = QARecord(
        query="q", answer="a", question_type="inference_query",
        evidence_list=(
            Evidence(title="t", author=None, url="https://example.com/doc-1", source="s", category="c",
                     published_at="2024-01-01T00:00:00+00:00", fact="f1"),
            Evidence(title="t", author=None, url="https://example.com/doc-2", source="s", category="c",
                     published_at="2024-01-01T00:00:00+00:00", fact="f2"),
        ),
    )
    controller_client, _ = _mantle_client([_FakeChatCompletion(_controller_json(True, None))])
    generation_client, _ = _mantle_client([_FakeChatCompletion("answer")])
    hop_results = [[
        _result("a", doc_id_from_url("https://example.com/doc-1")),
        _result("b", doc_id_from_url("https://example.com/doc-2")),
    ]]
    trace = run_agentic_retrieval(
        record.query, None, "collection", None, None, None, controller_client, generation_client,
        hop_runner=_ScriptedHopRunner(hop_results),
    )
    evaluation = evaluate_against_gold(record, trace)
    assert evaluation.recall == 1.0
    assert evaluation.complete_evidence is True


def test_evaluate_against_gold_partial_recall_when_one_doc_missing():
    record = QARecord(
        query="q", answer="a", question_type="inference_query",
        evidence_list=(
            Evidence(title="t", author=None, url="https://example.com/doc-1", source="s", category="c",
                     published_at="2024-01-01T00:00:00+00:00", fact="f1"),
            Evidence(title="t", author=None, url="https://example.com/doc-2", source="s", category="c",
                     published_at="2024-01-01T00:00:00+00:00", fact="f2"),
        ),
    )
    controller_client, _ = _mantle_client([_FakeChatCompletion(_controller_json(True, None))])
    generation_client, _ = _mantle_client([_FakeChatCompletion("answer")])
    hop_results = [[_result("a", doc_id_from_url("https://example.com/doc-1"))]]  # only 1 of 2 gold docs
    trace = run_agentic_retrieval(
        record.query, None, "collection", None, None, None, controller_client, generation_client,
        hop_runner=_ScriptedHopRunner(hop_results),
    )
    evaluation = evaluate_against_gold(record, trace)
    assert evaluation.recall == 0.5
    assert evaluation.complete_evidence is False


def test_run_calibration_query_never_sends_gold_fields_to_the_controller():
    """End-to-end: run_calibration_query takes a full QARecord (with gold
    fields) but the underlying controller only ever sees `record.query` —
    verified by inspecting exactly what was sent to the fake completions."""
    record = QARecord(
        query="What year was it founded?",
        answer="GOLD_ANSWER_MARKER_11223",
        question_type="inference_query",
        evidence_list=(
            Evidence(title="t", author=None, url="https://example.com/doc-1", source="s", category="c",
                     published_at="2024-01-01T00:00:00+00:00", fact="GOLD_FACT_MARKER_44556"),
        ),
    )
    controller_client, controller_completions = _mantle_client(
        [_FakeChatCompletion(_controller_json(True, None))]
    )
    generation_client, generation_completions = _mantle_client([_FakeChatCompletion("final answer")])

    result = run_calibration_query(
        record, qa_id="id1", hop_count=1,
        qdrant_client=None, collection_name="c", embedding_model=None, bm25_model=None, reranker=None,
        controller_client=controller_client, generation_client=generation_client,
        config=AgenticConfig(),
        hop_runner=_ScriptedHopRunner([[_result("a", doc_id_from_url("https://example.com/doc-1"))]]),
    )

    for call in controller_completions.calls + generation_completions.calls:
        sent_text = str(call["messages"])
        assert record.answer not in sent_text
        assert record.evidence_list[0].fact not in sent_text
        assert record.question_type not in sent_text

    assert result.qa_id == "id1"
    assert result.question_type == "inference_query"
    assert result.evaluation.recall == 1.0
