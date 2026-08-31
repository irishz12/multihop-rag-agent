"""Agentic loop control-flow tests — entirely offline: a scripted fake
`hop_runner` stands in for the real retrieval pipeline, and fake Mantle
clients (same injection pattern as tests/test_mantle_client.py) stand in
for the controller (GLM) and final-generation (Qwen) calls. No live
Qdrant/embedding/reranker/Mantle involved.
"""

from __future__ import annotations

import time

from mhrag.agent.loop import AgenticConfig, run_agentic_retrieval
from mhrag.generation.mantle_client import MantleClient
from mhrag.retrieval.schema import RetrievalResult

# --- fakes -----------------------------------------------------------------------------


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
        action = self._actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _mantle_client(actions) -> MantleClient:
    fake = _FakeOpenAIClient(_ScriptedCompletions(actions))
    return MantleClient(model_id="test-model", client=fake, max_retries=1, retry_base_delay_seconds=0.0)


def _controller_json(sufficient: bool, next_query: str | None, reason: str = "r") -> str:
    import json

    return json.dumps({"sufficient": sufficient, "next_query": next_query, "reason": reason})


def _result(chunk_id: str, doc_id: str, text: str = "chunk text") -> RetrievalResult:
    return RetrievalResult(
        rank=1, score=1.0, method="hybrid_reranked", chunk_id=chunk_id, doc_id=doc_id,
        title="t", url=f"https://example.com/{doc_id}", source="s", category="c",
        published_at="2024-01-01T00:00:00+00:00", text=text, position=0,
    )


class _ScriptedHopRunner:
    """Fake retrieval pipeline: returns one scripted result list per call,
    in order. Records every query it was called with."""

    def __init__(self, results_sequence: list[list[RetrievalResult]], latency_ms: float = 10.0):
        self._sequence = list(results_sequence)
        self._latency_ms = latency_ms
        self.calls: list[str] = []

    def __call__(self, query: str):
        self.calls.append(query)
        results = self._sequence[len(self.calls) - 1]
        return results, self._latency_ms, self._latency_ms / 2


def _run(
    question="What year and who?",
    hop_results_sequence=None,
    controller_actions=None,
    generation_action=None,
    config=None,
    hop_runner=None,
):
    if hop_runner is None:
        hop_runner = _ScriptedHopRunner(hop_results_sequence or [[_result("a", "doc-a")]])
    controller_client = _mantle_client(controller_actions or [_FakeChatCompletion(_controller_json(True, None))])
    generation_client = _mantle_client([generation_action or _FakeChatCompletion("final answer text")])
    trace = run_agentic_retrieval(
        question, None, "collection", None, None, None,
        controller_client, generation_client,
        config=config or AgenticConfig(),
        hop_runner=hop_runner,
    )
    return trace, hop_runner


# --- max 3 retrieval calls enforced ------------------------------------------------------


def test_max_three_retrieval_calls_hard_enforced():
    hop_results = [[_result("a", "doc-a")], [_result("b", "doc-b")], [_result("c", "doc-c")]]
    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "follow up 1")),
        _FakeChatCompletion(_controller_json(False, "follow up 2")),
        _FakeChatCompletion(_controller_json(False, "follow up 3")),  # would ask for a 4th hop if allowed
    ]
    trace, runner = _run(hop_results_sequence=hop_results, controller_actions=controller_actions)
    assert trace.num_retrieval_calls == 3
    assert len(runner.calls) == 3
    assert trace.stop_reason == "max_hops"


def test_default_config_max_hops_is_three():
    assert AgenticConfig().max_hops == 3


# --- sufficient evidence stops early -----------------------------------------------------


def test_sufficient_on_first_hop_stops_immediately():
    trace, runner = _run(
        hop_results_sequence=[[_result("a", "doc-a")]],
        controller_actions=[_FakeChatCompletion(_controller_json(True, None, "fully sufficient"))],
    )
    assert trace.num_retrieval_calls == 1
    assert len(runner.calls) == 1
    assert trace.stop_reason == "evidence_sufficient"
    assert trace.num_controller_calls == 1


def test_sufficient_on_second_hop_stops_after_two():
    hop_results = [[_result("a", "doc-a")], [_result("b", "doc-b")]]
    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "more specific query")),
        _FakeChatCompletion(_controller_json(True, None, "now sufficient")),
    ]
    trace, runner = _run(hop_results_sequence=hop_results, controller_actions=controller_actions)
    assert trace.num_retrieval_calls == 2
    assert trace.stop_reason == "evidence_sufficient"


# --- follow-up query generation ----------------------------------------------------------


def test_follow_up_query_is_passed_to_next_hop():
    hop_results = [[_result("a", "doc-a")], [_result("b", "doc-b")]]
    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "who is the CEO specifically")),
        _FakeChatCompletion(_controller_json(True, None)),
    ]
    trace, runner = _run(hop_results_sequence=hop_results, controller_actions=controller_actions)
    assert runner.calls[0] == "What year and who?"
    assert runner.calls[1] == "who is the CEO specifically"


def test_empty_follow_up_query_triggers_controller_failure():
    trace, runner = _run(
        hop_results_sequence=[[_result("a", "doc-a")]],
        controller_actions=[_FakeChatCompletion(_controller_json(False, "   "))],  # whitespace-only
    )
    assert trace.stop_reason == "controller_failure"
    assert len(runner.calls) == 1


# --- duplicate-query prevention -----------------------------------------------------------


def test_duplicate_query_stops_the_loop_before_retrieving_again():
    question = "What year and who?"
    controller_actions = [_FakeChatCompletion(_controller_json(False, question))]  # repeats the original question
    trace, runner = _run(
        question=question,
        hop_results_sequence=[[_result("a", "doc-a")]],
        controller_actions=controller_actions,
    )
    assert trace.stop_reason == "duplicate_query"
    assert len(runner.calls) == 1  # hop 2 never actually ran


def test_duplicate_query_check_is_case_and_whitespace_insensitive():
    question = "What year?"
    controller_actions = [_FakeChatCompletion(_controller_json(False, "  WHAT YEAR?  "))]
    trace, runner = _run(
        question=question,
        hop_results_sequence=[[_result("a", "doc-a")]],
        controller_actions=controller_actions,
    )
    assert trace.stop_reason == "duplicate_query"
    assert len(runner.calls) == 1


# --- chunk/doc dedup at loop level ---------------------------------------------------------


def test_evidence_deduplicated_across_hops_at_loop_level():
    hop1 = [_result("a", "doc-a"), _result("b", "doc-b")]
    hop2 = [_result("b", "doc-b"), _result("c", "doc-c")]  # "b" duplicate
    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "follow up")),
        _FakeChatCompletion(_controller_json(True, None)),
    ]
    trace, _ = _run(hop_results_sequence=[hop1, hop2], controller_actions=controller_actions)
    assert [c.chunk_id for c in trace.evidence_pool] == ["a", "b", "c"]
    assert trace.unique_chunks_retrieved == 3
    assert trace.duplicate_chunks_removed == 1
    assert trace.unique_documents_retrieved == 3


# --- token-budget enforcement ---------------------------------------------------------------


def test_token_budget_stops_loop_and_controller_is_not_called_that_hop():
    long_text = " ".join(["word"] * 5000)  # comfortably over any small budget
    trace, runner = _run(
        hop_results_sequence=[[_result("a", "doc-a", long_text)]],
        config=AgenticConfig(max_context_tokens=10),
        controller_actions=[_FakeChatCompletion(_controller_json(True, None))],  # should never be consumed
    )
    assert trace.stop_reason == "token_budget"
    assert trace.num_controller_calls == 0
    assert len(runner.calls) == 1


def test_token_budget_not_exceeded_proceeds_normally():
    trace, _ = _run(
        hop_results_sequence=[[_result("a", "doc-a", "short text")]],
        config=AgenticConfig(max_context_tokens=3000),
        controller_actions=[_FakeChatCompletion(_controller_json(True, None))],
    )
    assert trace.stop_reason == "evidence_sufficient"


# --- timeout behavior ------------------------------------------------------------------------


def test_timeout_stops_before_a_later_hop():
    class _SlowHopRunner:
        def __init__(self):
            self.calls: list[str] = []

        def __call__(self, query):
            self.calls.append(query)
            time.sleep(0.05)
            return [_result(f"c{len(self.calls)}", f"doc-{len(self.calls)}")], 10.0, 5.0

    runner = _SlowHopRunner()
    controller_actions = [_FakeChatCompletion(_controller_json(False, "follow up"))]
    trace, _ = _run(
        hop_runner=runner,
        config=AgenticConfig(timeout_seconds=0.02, max_hops=3),
        controller_actions=controller_actions,
    )
    assert trace.stop_reason == "timeout"
    assert len(runner.calls) == 1  # hop 1 completed, hop 2's pre-check caught the timeout


# --- controller failure fallback + malformed JSON at loop level ------------------------------


def test_controller_failure_stops_loop_and_still_generates_an_answer():
    trace, runner = _run(
        hop_results_sequence=[[_result("a", "doc-a")]],
        controller_actions=[_FakeChatCompletion("not valid json")],
    )
    assert trace.stop_reason == "controller_failure"
    assert len(runner.calls) == 1
    assert trace.final_generation.answer == "final answer text"  # best-effort answer still produced


# --- every LLM call contributes to cost tracking ---------------------------------------------


def test_cost_tracking_includes_glm_and_qwen_from_every_call():
    hop_results = [[_result("a", "doc-a")], [_result("b", "doc-b")]]
    controller_actions = [
        _FakeChatCompletion(_controller_json(False, "follow up"), usage=_FakeUsage(20, 10, 30)),
        _FakeChatCompletion(_controller_json(True, None), usage=_FakeUsage(25, 8, 33)),
    ]
    generation_action = _FakeChatCompletion("answer", usage=_FakeUsage(100, 40, 140))
    trace, _ = _run(
        hop_results_sequence=hop_results, controller_actions=controller_actions, generation_action=generation_action
    )
    assert trace.num_controller_calls == 2
    assert trace.cost.glm_input_tokens == 20 + 25
    assert trace.cost.glm_output_tokens == 10 + 8
    assert trace.cost.qwen_input_tokens == 100
    assert trace.cost.qwen_output_tokens == 40
    assert trace.cost.glm_cost_usd is not None
    assert trace.cost.qwen_cost_usd is not None
    assert trace.cost.total_cost_usd == trace.cost.glm_cost_usd + trace.cost.qwen_cost_usd


# --- repeatable loop behavior with injected fake responses -----------------------------------


# --- gold answer/evidence/question_type never enter any controller prompt in the loop ---


def test_full_loop_never_leaks_gold_fields_into_any_controller_call():
    from mhrag.data.schema import Evidence, QARecord

    record = QARecord(
        query="What year was the company founded and who is the CEO?",
        answer="GOLD_ANSWER_MARKER_98765",
        question_type="inference_query",
        evidence_list=(
            Evidence(
                title="t", author=None, url="https://example.com/doc-a", source="s", category="c",
                published_at="2024-01-01T00:00:00+00:00", fact="GOLD_EVIDENCE_FACT_MARKER_44556",
            ),
        ),
    )
    hop_results = [[_result("a", "doc-a")], [_result("b", "doc-b")]]
    controller_completions = _ScriptedCompletions(
        [
            _FakeChatCompletion(_controller_json(False, "follow up query")),
            _FakeChatCompletion(_controller_json(True, None)),
        ]
    )
    controller_client = MantleClient(
        model_id="glm-test", client=_FakeOpenAIClient(controller_completions), max_retries=1, retry_base_delay_seconds=0.0
    )
    generation_client = _mantle_client([_FakeChatCompletion("final answer")])

    run_agentic_retrieval(
        record.query, None, "collection", None, None, None,
        controller_client, generation_client,
        hop_runner=_ScriptedHopRunner(hop_results),
    )

    for call in controller_completions.calls:
        sent_text = str(call["messages"])
        assert record.answer not in sent_text
        assert record.evidence_list[0].fact not in sent_text
        assert record.question_type not in sent_text


def test_repeated_run_with_same_scripted_inputs_is_deterministic():
    def build_and_run():
        hop_results = [[_result("a", "doc-a")], [_result("b", "doc-b")]]
        controller_actions = [
            _FakeChatCompletion(_controller_json(False, "follow up")),
            _FakeChatCompletion(_controller_json(True, None)),
        ]
        return _run(hop_results_sequence=hop_results, controller_actions=controller_actions)

    trace1, _ = build_and_run()
    trace2, _ = build_and_run()

    assert trace1.stop_reason == trace2.stop_reason
    assert [h.query for h in trace1.hops] == [h.query for h in trace2.hops]
    assert [c.chunk_id for c in trace1.evidence_pool] == [c.chunk_id for c in trace2.evidence_pool]
    assert trace1.final_generation.answer == trace2.final_generation.answer
    assert trace1.num_retrieval_calls == trace2.num_retrieval_calls
    assert trace1.num_controller_calls == trace2.num_controller_calls
