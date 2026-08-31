"""End-to-end (offline, fake-client) answer generation tests: wiring between
context assembly, prompting, the Mantle client, and cost estimation — and
the structural guarantee that gold answers/evidence never enter a prompt.
"""

from __future__ import annotations

import inspect

from mhrag.generation.answer import generate_answer
from mhrag.generation.mantle_client import MantleClient
from mhrag.retrieval.schema import RetrievalResult


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens):
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
    def __init__(self, content, usage):
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class _RecordingCompletions:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _word_count(text: str) -> int:
    return len(text.split())


def _result(rank: int, chunk_id: str, doc_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=1.0 / rank,
        method="hybrid_reranked",
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=f"Title {doc_id}",
        url=f"https://example.com/{doc_id}",
        source="Source",
        category="technology",
        published_at="2024-01-01T00:00:00+00:00",
        text=text,
        position=0,
    )


def _make_client(answer_text="the answer", usage=None):
    usage = usage or _FakeUsage(10, 5, 15)
    completions = _RecordingCompletions(_FakeChatCompletion(answer_text, usage))
    fake_openai = _FakeOpenAIClient(completions)
    client = MantleClient(model_id="test-model", client=fake_openai)
    return client, completions


def test_generate_answer_wires_context_prompt_and_cost_together():
    client, completions = _make_client(usage=_FakeUsage(100, 50, 150))
    retrieved = [_result(1, "c1", "doc-1", "The company was founded in 1999.")]

    result = generate_answer(
        "What year was the company founded?",
        retrieved,
        client,
        _word_count,
        top_k=5,
        max_context_tokens=1000,
        input_price_per_million=0.18,
        output_price_per_million=1.41,
    )

    assert result.answer == "the answer"
    assert result.mantle_response.success is True
    assert len(result.context.chunks_included) == 1
    assert result.context.chunks_included[0].chunk_id == "c1"
    assert result.cost.total_cost_usd is not None
    assert result.prompt_version == "v1"

    # the actual text sent to the fake client contains the question and context
    sent_messages = completions.calls[0]["messages"]
    user_content = sent_messages[1]["content"]
    assert "What year was the company founded?" in user_content
    assert "The company was founded in 1999." in user_content


def test_generate_answer_respects_context_budget():
    client, _ = _make_client()
    retrieved = [_result(1, "c1", "doc-1", " ".join(["word"] * 100))]

    result = generate_answer(
        "q", retrieved, client, _word_count, top_k=5, max_context_tokens=10,
        input_price_per_million=0.18, output_price_per_million=1.41,
    )
    assert result.context.total_token_count <= 10
    assert len(result.context.chunks_dropped) == 1


# --- gold answers/evidence never enter prompts ------------------------------------------


def test_generate_answer_signature_has_no_ground_truth_parameter():
    params = set(inspect.signature(generate_answer).parameters)
    for forbidden in ("answer", "gold_answer", "evidence", "evidence_list", "question_type"):
        assert forbidden not in params


def test_retrieval_result_has_no_ground_truth_fields():
    """RetrievalResult (what `retrieved` is built from) structurally cannot
    carry gold answer/evidence/question_type — there is no field for it,
    so generate_answer has no way to forward ground truth even if a caller
    tried to smuggle it into a RetrievalResult's other fields."""
    fields = {f for f in RetrievalResult.__dataclass_fields__}
    for forbidden in ("answer", "gold_answer", "evidence", "evidence_list", "question_type"):
        assert forbidden not in fields


def test_generate_answer_never_transmits_a_qa_records_gold_fields():
    """Concrete behavioral check using a realistic QARecord: only
    `record.query` and the retrieved chunks' `.text` reach the prompt —
    `record.answer` and every evidence fact are distinct, deliberately
    unique strings here, and must never appear in what gets sent."""
    from mhrag.data.schema import Evidence, QARecord

    record = QARecord(
        query="What year was the company founded?",
        answer="GOLD_ANSWER_MARKER_69420",
        question_type="inference_query",
        evidence_list=(
            Evidence(
                title="t",
                author=None,
                url="https://example.com/doc-1",
                source="s",
                category="c",
                published_at="2024-01-01T00:00:00+00:00",
                fact="GOLD_EVIDENCE_FACT_MARKER_13579",
            ),
        ),
    )
    client, completions = _make_client()
    retrieved = [_result(1, "c1", "doc-1", "ordinary retrieved chunk text, unrelated to the gold fields")]

    generate_answer(
        record.query, retrieved, client, _word_count, top_k=5, max_context_tokens=1000,
        input_price_per_million=0.18, output_price_per_million=1.41,
    )

    sent_text = str(completions.calls[0]["messages"])
    assert record.answer not in sent_text
    assert record.evidence_list[0].fact not in sent_text
    assert record.question_type not in sent_text
