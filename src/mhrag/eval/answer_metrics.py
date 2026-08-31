"""EVALUATOR-ONLY: answer-quality string metrics — Phase 9.

Standard SQuAD-style normalized Exact Match and token F1 (Rajpurkar et al.,
2016) — the customary baseline metrics for open-domain QA answer strings,
used here because MultiHop-RAG's own official evaluation script
(github.com/yixuantt/MultiHop-RAG, retrieval_evaluate.py — already fetched
and inspected for `mhrag.eval.metrics`) scores RETRIEVAL only (Hits@K,
MAP@10, MRR@10 against evidence facts); it has no answer-generation
evaluation to align with, since generating an answer is not part of the
original benchmark's task. Normalized EM/F1 are "aligned where appropriate"
by being the standard the wider QA literature uses for exactly this kind of
free-text answer scoring, applied on top of retrieval that IS evaluated the
official way.

Also provides an abstention detector for `null_query` (MultiHop-RAG's
gold answer for null_query is verbatim "Insufficient information."; the
project's own generation prompt, `mhrag.generation.prompts.SYSTEM_PROMPT_
V1`, instructs the model to "say exactly that the available information is
insufficient to answer" rather than parroting the gold string, so detection
is phrase-based, not an exact-match against the literal gold text).

Pure string functions, no I/O, no model call — takes only already-generated
answer text and the gold answer text, both plain strings.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = frozenset({"a", "an", "the"})
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer_text(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, drop
    articles, collapse whitespace. Deterministic, no model/locale
    dependency."""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    words = [w for w in no_punct.split() if w not in _ARTICLES]
    return " ".join(words)


def exact_match(prediction: str, gold: str) -> int:
    """1 if the normalized strings are identical, else 0. Meaningful only
    for short, factoid-style gold answers (many MultiHop-RAG answers are
    short spans/names) — for longer explanatory gold answers this is
    expected to be near-always 0, which is why token F1 and the LLM judge
    exist alongside it rather than in place of it."""
    return int(normalize_answer_text(prediction) == normalize_answer_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Standard bag-of-words token F1 over normalized tokens. 1.0 for a
    perfect token multiset match, 0.0 if there is no overlap at all (or
    either string normalizes to zero tokens)."""
    pred_tokens = normalize_answer_text(prediction).split()
    gold_tokens = normalize_answer_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)  # both empty -> 1.0, only one empty -> 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# Phrases the project's own generation prompt (mhrag.generation.prompts.SYSTEM_PROMPT_V1)
# steers the model toward when it judges the context insufficient — kept as a small,
# explicit, documented list rather than one brittle exact-string check, since the model's
# exact phrasing varies (observed in Phase 8B: "insufficient to answer.", "does not include
# ... insufficient information to answer.", "insufficient information does not support...").
ABSTENTION_PHRASES: tuple[str, ...] = (
    "insufficient",
    "not enough information",
    "no information",
    "cannot be determined",
    "cannot determine",
    "does not contain enough information",
)

_WHITESPACE_RE = re.compile(r"\s+")


def is_abstention(answer_text: str) -> bool:
    """Whether `answer_text` reads as a refusal-to-answer for insufficient
    context, per `ABSTENTION_PHRASES`. Case-insensitive substring match —
    deliberately simple and inspectable (no model call), matching the
    project's "interpretable features over black-box heuristics" pattern
    used throughout the router work."""
    lowered = _WHITESPACE_RE.sub(" ", answer_text.lower())
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)
