"""LLM-judge tests — entirely offline: a fake Mantle client (same
injection pattern as tests/test_mantle_client.py / tests/test_agent_loop.py)
stands in for GLM. No live call.
"""

from __future__ import annotations

import json

from mhrag.eval.judge import GRADE_TO_SCORE, call_judge
from mhrag.generation.mantle_client import MantleClient

# --- fakes (same pattern as tests/test_agent_loop.py) ----------------------------------------


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
    return MantleClient(model_id="test-judge-model", client=fake, max_retries=1, retry_base_delay_seconds=0.0)


def _judge_json(grade: str, reason: str = "r") -> str:
    return json.dumps({"grade": grade, "reason": reason})


# --- grading -------------------------------------------------------------------------------


def test_correct_grade_maps_to_score_one():
    client = _mantle_client([_FakeChatCompletion(_judge_json("correct"))])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.verdict.grade == "correct"
    assert result.verdict.score == 1.0
    assert not result.fallback_used


def test_partially_correct_grade_maps_to_score_half():
    client = _mantle_client([_FakeChatCompletion(_judge_json("partially_correct"))])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.verdict.score == 0.5


def test_incorrect_grade_maps_to_score_zero():
    client = _mantle_client([_FakeChatCompletion(_judge_json("incorrect"))])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.verdict.score == 0.0


def test_grade_to_score_mapping_is_fixed():
    assert GRADE_TO_SCORE == {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


# --- robustness: never trust a broken call ---------------------------------------------------


def test_call_failure_falls_back_to_conservative_incorrect():
    client = _mantle_client([RuntimeError("boom")])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.fallback_used
    assert result.verdict.grade == "incorrect"
    assert result.verdict.score == 0.0


def test_unparseable_response_falls_back_to_conservative_incorrect():
    client = _mantle_client([_FakeChatCompletion("not json at all")])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.fallback_used
    assert result.verdict.score == 0.0


def test_invalid_grade_value_falls_back():
    client = _mantle_client([_FakeChatCompletion(json.dumps({"grade": "sort_of", "reason": "r"}))])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.fallback_used
    assert result.verdict.grade == "incorrect"


def test_missing_reason_defaults_to_empty_string_not_a_failure():
    client = _mantle_client([_FakeChatCompletion(json.dumps({"grade": "correct"}))])
    result = call_judge(client, "q", "gold", "candidate")
    assert not result.fallback_used
    assert result.verdict.reason == ""


def test_known_openai_gpt_oss_double_open_brace_glitch_is_repaired():
    """Discovered live during Phase 9's mandatory small-sample judge
    validation (results/phase9_judge_validation.json): openai.gpt-oss-120b
    reliably emits a duplicated leading '{' with no matching second '}' —
    this must parse successfully, not fall back."""
    malformed = '{\n{\n  "grade": "correct"\n  \n  ,\n  "reason": "matches"\n}'
    client = _mantle_client([_FakeChatCompletion(malformed)])
    result = call_judge(client, "q", "gold", "candidate")
    assert not result.fallback_used
    assert result.verdict.grade == "correct"
    assert result.verdict.reason == "matches"


def test_double_open_brace_repair_does_not_mask_genuinely_broken_json():
    """The repair is narrow: text that merely STARTS with two braces but
    is otherwise broken must still fall back, not be silently accepted."""
    malformed = '{\n{\n  "grade": "correct" this is not valid json at all'
    client = _mantle_client([_FakeChatCompletion(malformed)])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.fallback_used


# --- judge cost/latency tracked separately, never merged into a pipeline's own cost ------------


def test_judge_result_exposes_its_own_mantle_response_for_separate_cost_tracking():
    usage = _FakeUsage(prompt_tokens=123, completion_tokens=7)
    client = _mantle_client([_FakeChatCompletion(_judge_json("correct"), usage=usage)])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.mantle_response.usage.input_tokens == 123
    assert result.mantle_response.usage.output_tokens == 7


# --- cost tracked separately, never guessed when pricing is unknown ---------------------------


def test_judge_cost_is_none_when_pricing_not_supplied():
    client = _mantle_client([_FakeChatCompletion(_judge_json("correct"))])
    result = call_judge(client, "q", "gold", "candidate")
    assert result.cost is None


def test_judge_cost_computed_when_pricing_supplied():
    usage = _FakeUsage(prompt_tokens=1000, completion_tokens=500)
    client = _mantle_client([_FakeChatCompletion(_judge_json("correct"), usage=usage)])
    result = call_judge(
        client, "q", "gold", "candidate", input_price_per_million=1.0, output_price_per_million=2.0
    )
    assert result.cost is not None
    assert result.cost.total_cost_usd == (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 2.0


# --- structural: gold answer required, no pipeline-identifying parameter, never leaked ----------


def test_call_judge_signature_requires_gold_answer_parameter():
    import inspect

    params = list(inspect.signature(call_judge).parameters)
    assert "gold_answer" in params


def test_call_judge_signature_has_no_pipeline_identifying_parameter():
    """The judge must never be told which pipeline/route/retrieval method/
    generation model produced the candidate answer — grading must be based
    solely on (question, gold_answer, candidate_answer)."""
    import inspect

    params = set(inspect.signature(call_judge).parameters)
    forbidden = {"pipeline", "pipeline_name", "route", "predicted_route", "retrieval_method", "model", "model_id"}
    assert not (forbidden & params)
