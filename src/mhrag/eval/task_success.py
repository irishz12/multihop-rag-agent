"""EVALUATOR-ONLY: Task Success classification — a deterministic-first
correctness/abstention/groundedness layer, independent of (and reported
alongside, never merged silently into) the existing LLM judge.

METHODOLOGY, STATED EXPLICITLY: Task Success is NOT "whatever the judge
says." Four signals are computed and kept SEPARATE:

  1. Deterministic answer checks (`verdict_match` for comparison_query /
     temporal_query, `entity_containment` for inference_query — the latter
     explicitly NOT treated as proof of correctness, only a secondary
     signal; see its own docstring).
  2. The existing LLM judge grade/score (`mhrag.eval.judge`) — passed
     through unchanged, never recomputed here, never allowed to overwrite
     a deterministic verdict.
  3. Evidence coverage (`mhrag.eval.metrics`/ground truth doc-id overlap)
     — passed through, used only for the `unsupported` grounding flag.
  4. Abstention behavior (`mhrag.eval.answer_metrics.is_abstention`) —
     extended here from null_query-only to the full population, producing
     all four cells of the abstention x question-type-is-null 2x2, not
     just the one cell ("correct abstention on null_query") the rest of
     this project already measures.

`classify_task_success` combines these into one `TaskSuccessResult` with
every signal as its OWN field — never one opaque blended number. Where the
deterministic check and the judge disagree, that disagreement is surfaced
(`judge_deterministic_agree`), never silently resolved in the judge's
favor. `task_success_confident` is True only when the deterministic
component itself produced a non-ambiguous verdict (see its own docstring
below) — it does NOT mean "judge and deterministic agree."

GOLD DATA USAGE: this module reads `gold_answer` (already-generated
answer text is compared against it) exactly the same way
`mhrag.eval.judge`/`mhrag.eval.answer_metrics` already do. It does NOT
read `Evidence.fact` (fact-level groundedness is an explicitly deferred,
separate, not-yet-approved phase — see the Task Success design doc). Like
every other module in `mhrag.eval`, this module must NEVER be imported by
`mhrag.agent`, `mhrag.generation`, `mhrag.routing`, or `mhrag.adaptive` —
see tests/test_task_success_no_gold_leakage.py.

DETERMINISTIC VERDICT EXTRACTION — DESIGN NOTES (see `extract_verdict`):
Empirically (this project's own dev-split audit), comparison_query and
temporal_query gold answers are drawn from a small closed vocabulary
(observed: yes/no/similar/true/align/different-aspects for comparison_query;
no/yes/consistent/change for temporal_query) — NOT open-ended explanatory
text as this project's README currently (and incorrectly) characterizes
ALL MultiHop-RAG answers. `extract_verdict` exploits this: it looks for a
committed verdict token via three tiers, from highest to lowest confidence:

  1. LEADING: the answer's very first word is a canonical verdict token
     (optionally negated), e.g. "Yes, the sources agree."
  2. EXPLICIT: an "the answer is X" / "it is X" / "overall, X" marker
     anywhere in the text (optionally negated), e.g. "It is unclear, but
     overall yes."
  3. CLAUSE FALLBACK: a canonical verdict token appearing as a whole word,
     but ONLY when it is CLAUSE-FINAL — i.e. nothing but punctuation
     follows it before the next clause boundary — with a same-clause
     negation check.

If tier-1/2 matches exist, ONLY those are used (tier 3 is ignored) and the
LAST such match wins — this is what correctly resolves the negation trap
"The answer is not no; it is yes." (two tier-2 matches; the second,
unnegated "yes," wins) without needing to trust the negation-flip logic on
the first, ambiguous "not no" fragment. If zero matches exist across all
three tiers, the result is explicitly `is_ambiguous=True` — this function
NEVER guesses a verdict it did not find textual evidence for. An "X or Y"
meta-reference (e.g., "hard to give a simple yes or no") is stripped
before scanning specifically so that hedge language mentioning both
options isn't misread as a commitment to either.

PHASE 3 HARDENING — THE CLAUSE-FINAL RESTRICTION ON TIER 3, WHY IT EXISTS:
the original (Phase 2) tier-3 accepted a canonical word ANYWHERE in its
clause. This produced a real, discovered false positive: "There is no
praise for the Biden administration..." was read as a "no" verdict, when
"no" here is a DETERMINER modifying "praise" ("an absence of praise"), not
a verdict commitment — "no" is the one canonical token in this project's
vocabulary that also functions as an ordinary English determiner/adverb
("no evidence", "no mention", "no indication", "no longer", ...) wherever
it is immediately followed by more clause content. Genuine verdict
assertions, by contrast, overwhelmingly place the verdict word at the END
of its clause ("...so the answer here is no.", "No, they don't agree."
— the latter's "No" is itself a one-word clause, bounded by the comma).
Requiring clause-finality is therefore a general fix (it also protects
"similar"/"consistent"/etc. from an analogous adjective-before-noun
misread, e.g. "a similar situation" — untested here since the observed
vocabulary skews overwhelmingly yes/no, but the same rule applies), not a
"no"-specific patch. The cost is real and documented: a genuine verdict
buried mid-clause with trailing filler ("the answer is no, in my view")
would now be missed by tier 3 and fall through to ambiguous — accepted
deliberately, per the explicit instruction that an uncertain case must
return ambiguous rather than guess.

KNOWN LIMITATIONS (do not silently paper over):
  - The negator list and clause-splitting are regex-based, not a real
    parser — they can be fooled by negation spanning a clause boundary
    ("The sources do not appear to agree, however they say yes.") in ways
    not covered by the test suite; that's why ambiguous cases are surfaced
    rather than resolved by default.
  - The canonical-verdict vocabulary was derived from the DEVELOPMENT
    split only (300 questions) — a gold value outside that vocabulary
    (e.g., the single observed "different aspects") is intentionally
    scored `not_applicable`, never forced into the closest-looking token.
  - The clause-final restriction (above) trades recall for precision on
    tier 3 — some genuine but non-clause-final verdicts now resolve to
    ambiguous instead of being extracted. This is a deliberate, documented
    trade-off, not an oversight.

BACKWARD COMPATIBILITY (Phase 3 vs Phase 2): `verdict_match`/
`extract_verdict`'s PUBLIC BEHAVIOR CHANGES for inputs where tier 3 was
the only tier that matched AND the matched word was not clause-final —
those now resolve to `ambiguous` instead of `correct`/`incorrect`. Tier
1/2 behavior, `entity_containment`, `classify_abstention`, `is_abstention`
usage, and every other function in this module are UNCHANGED. See
scripts/task_success_hardening_error_analysis.py for exactly which
already-computed qa_ids this affects and by how much.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mhrag.eval.answer_metrics import is_abstention, normalize_answer_text

# --- canonical verdict vocabulary --------------------------------------------------------
# Derived empirically from data/processed/dev_subset.json (300 questions): comparison_query
# and temporal_query gold answers are overwhelmingly yes/no, with a handful of other
# single-token verdicts. Kept as one shared set (comparison_query and temporal_query use the
# same vocabulary in practice) rather than two near-duplicate lists.
CANONICAL_VERDICTS: tuple[str, ...] = ("yes", "no", "true", "false", "similar", "consistent", "change", "align")
_FLIP_PAIRS: dict[str, str] = {"yes": "no", "no": "yes", "true": "false", "false": "true"}
_NEGATORS: tuple[str, ...] = ("not", "n't", "never", "cannot", "can't", "doesn't", "does not", "did not", "didn't")
# PHASE 4 addition, found via adversarial construction (not a pre-existing bug report): a
# sentence-final canonical word preceded, in the same sentence, by an uncertainty indicator
# ("uncertain", "unsure", "unclear", "unknown", "whether") is a HEDGE, not a commitment —
# "It's uncertain whether they align." must not extract "align" just because it happens to
# be the last word. Unlike a negator, there is no clean "flip" for this (no canonical
# opposite of a hedged "align"), so a match preceded by one of these is dropped entirely,
# never resolved.
_UNCERTAINTY_BLOCKERS: tuple[str, ...] = ("uncertain", "unsure", "unclear", "unknown", "whether")

_VERDICT_ALTERNATION = "|".join(CANONICAL_VERDICTS)
_LEADING_RE = re.compile(rf"^\s*(not\s+)?\b({_VERDICT_ALTERNATION})\b", re.IGNORECASE)
_EXPLICIT_RE = re.compile(
    rf"\b(?:the answer is|it is|overall,?)\s+(not\s+)?\b({_VERDICT_ALTERNATION})\b", re.IGNORECASE
)
_META_OR_RE = re.compile(rf"\b({_VERDICT_ALTERNATION})\s+or\s+({_VERDICT_ALTERNATION})\b", re.IGNORECASE)
# PHASE 4: sentence-only boundary for tier 3 — deliberately excludes the comma. A comma in
# English marks list items, appositives, and parentheticals far more often than it marks a
# real clause boundary ("climate change, the modern internet, and authoritarianism" is a
# 3-item list, not three clauses) — Phase 3's comma-inclusive splitter was fooled by exactly
# this into reading "change" as clause-final. A period/question mark/exclamation mark is a
# genuine sentence boundary; a semicolon is kept too since it specifically marks a boundary
# between two independent clauses (a much rarer, stronger signal than a comma) and no
# required test case depends on it being excluded. See module docstring's PHASE 4 HARDENING
# section.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?;]")
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True, slots=True)
class VerdictExtraction:
    """Result of scanning free text for a committed canonical verdict.

    `verdict` is `None` whenever `is_ambiguous` is True — the two fields
    are never in conflict (there is no "ambiguous but here's a guess"
    state; see module docstring)."""

    verdict: str | None
    is_ambiguous: bool
    matched_tier: str | None  # "leading" | "explicit" | "clause_fallback" | None


def _strip_meta_or_phrases(text: str) -> str:
    """Remove "X or Y" hedge references (e.g. "a simple yes or no") by
    blanking them out with spaces of equal length, so downstream regex
    positions elsewhere in the string are unaffected. This is what makes
    "It's hard to give a simple yes or no on this one." correctly resolve
    to ambiguous instead of picking up a spurious "no" match."""

    def _blank(match: re.Match[str]) -> str:
        return " " * len(match.group(0))

    return _META_OR_RE.sub(_blank, text)


def _resolve_negation(verdict: str, negated: bool) -> str | None:
    """Apply a detected negation to a matched verdict token. Only the
    yes/no and true/false pairs have a well-defined single opposite in
    the observed vocabulary; negating a non-flippable token (e.g. "not
    similar") has no clean canonical opposite, so that occurrence is
    dropped (returns None) rather than guessed."""
    if not negated:
        return verdict
    return _FLIP_PAIRS.get(verdict)


def _scan_priority_patterns(text: str) -> list[tuple[int, str]]:
    """Tier 1 (leading) + Tier 2 (explicit) matches, in text order, each
    as (position, resolved_verdict) — already negation-resolved, with
    non-resolvable negations dropped."""
    matches: list[tuple[int, str]] = []
    leading = _LEADING_RE.match(text)
    if leading:
        resolved = _resolve_negation(leading.group(2).lower(), leading.group(1) is not None)
        if resolved is not None:
            matches.append((leading.start(), resolved))
    for m in _EXPLICIT_RE.finditer(text):
        resolved = _resolve_negation(m.group(2).lower(), m.group(1) is not None)
        if resolved is not None:
            matches.append((m.start(), resolved))
    matches.sort(key=lambda pair: pair[0])
    return matches


def _scan_clause_fallback(text: str) -> list[tuple[int, str]]:
    """Tier 3, HARDENED TWICE (Phase 3, then Phase 4): a canonical verdict
    word counts only when it is SENTENCE-FINAL — nothing but
    whitespace/punctuation follows it before the next `.`/`!`/`?`/`;` —
    with a same-sentence negation check on what precedes it.

    Phase 3 required "clause-final," splitting on commas too, which is
    what let "climate change, the modern internet, and authoritarianism"
    misread "change" as a verdict — a comma there marks a LIST ITEM
    boundary, not a clause boundary. Phase 4 drops the comma from the
    boundary set entirely: a canonical word followed by ANYTHING else in
    its sentence — a list continuation, an appositive, a quoted aside, an
    explanatory clause joined by a comma — is rejected, no exceptions.
    This is deliberately conservative: it will miss some genuine
    mid-sentence, comma-separated verdicts (e.g. "No, in short, they
    don't agree" — "No" is caught by tier 1 regardless, so this specific
    example is fine, but a verdict appearing only after an internal
    comma with nothing before it would now resolve to ambiguous). That
    trade-off is intentional — see module docstring's "prefer ambiguous
    over guessing" principle."""
    matches: list[tuple[int, str]] = []
    offset = 0
    for clause in _SENTENCE_SPLIT_RE.split(text):
        clause_lower = clause.lower()
        for word_match in re.finditer(rf"\b({_VERDICT_ALTERNATION})\b", clause_lower):
            following = clause_lower[word_match.end() :]
            if following.strip():
                continue  # more sentence content after the word -> not a standalone verdict
            preceding = clause_lower[: word_match.start()]
            if any(re.search(rf"\b{re.escape(blocker)}\b", preceding) for blocker in _UNCERTAINTY_BLOCKERS):
                continue  # "It's uncertain whether they align." -> hedge, not a commitment; drop, don't flip
            negated = any(re.search(rf"\b{re.escape(neg)}\b", preceding) for neg in _NEGATORS)
            resolved = _resolve_negation(word_match.group(1), negated)
            if resolved is not None:
                matches.append((offset + word_match.start(), resolved))
        offset += len(clause) + 1  # +1 for the stripped delimiter character
    matches.sort(key=lambda pair: pair[0])
    return matches


def extract_verdict(text: str) -> VerdictExtraction:
    """Scan `text` for a committed canonical verdict (see module
    docstring for the exact three-tier algorithm). Never guesses: returns
    `is_ambiguous=True, verdict=None` when no tier finds a match."""
    cleaned = _strip_meta_or_phrases(text)

    priority_matches = _scan_priority_patterns(cleaned)
    if priority_matches:
        tier = "leading" if _LEADING_RE.match(cleaned) and priority_matches[0][0] == _LEADING_RE.match(cleaned).start() else "explicit"
        return VerdictExtraction(verdict=priority_matches[-1][1], is_ambiguous=False, matched_tier=tier)

    fallback_matches = _scan_clause_fallback(cleaned)
    if fallback_matches:
        return VerdictExtraction(verdict=fallback_matches[-1][1], is_ambiguous=False, matched_tier="clause_fallback")

    return VerdictExtraction(verdict=None, is_ambiguous=True, matched_tier=None)


def normalize_gold_verdict(gold_answer: str) -> str | None:
    """Map a gold answer string to a canonical verdict token, or `None`
    if it isn't one of the recognized canonical values (e.g. the observed
    "different aspects" — a real gold value in this dataset that this
    deterministic checker intentionally does not attempt to match; see
    module docstring's KNOWN LIMITATIONS)."""
    normalized = gold_answer.strip().lower()
    return normalized if normalized in CANONICAL_VERDICTS else None


@dataclass(frozen=True, slots=True)
class VerdictMatchResult:
    """Deterministic comparison_query / temporal_query scoring result."""

    correctness: str  # "correct" | "incorrect" | "ambiguous" | "not_applicable"
    gold_verdict: str | None
    extracted_verdict: str | None
    is_ambiguous: bool


def verdict_match(gold_answer: str, generated_answer: str) -> VerdictMatchResult:
    """Deterministically score a comparison_query/temporal_query answer
    against its gold verdict. Returns `not_applicable` when the GOLD
    answer itself isn't a recognized canonical verdict (a vocabulary gap,
    not a matching failure); returns `ambiguous` when the gold IS
    recognized but no committed verdict could be extracted from the
    PREDICTION (a genuine matching failure, never forced to correct/
    incorrect)."""
    gold_verdict = normalize_gold_verdict(gold_answer)
    if gold_verdict is None:
        return VerdictMatchResult(correctness="not_applicable", gold_verdict=None, extracted_verdict=None, is_ambiguous=False)

    extraction = extract_verdict(generated_answer)
    if extraction.is_ambiguous:
        return VerdictMatchResult(
            correctness="ambiguous", gold_verdict=gold_verdict, extracted_verdict=None, is_ambiguous=True
        )

    correctness = "correct" if extraction.verdict == gold_verdict else "incorrect"
    return VerdictMatchResult(
        correctness=correctness, gold_verdict=gold_verdict, extracted_verdict=extraction.verdict, is_ambiguous=False
    )


def entity_containment(gold_answer: str, generated_answer: str) -> bool | None:
    """SECONDARY, LOWER-CONFIDENCE deterministic signal for
    inference_query only: does the normalized gold entity's token
    sequence appear as a contiguous subsequence within the normalized
    prediction's tokens? Token-level (not raw-substring) containment, so
    a gold value like "meta" cannot spuriously match inside an unrelated
    word like "climate."

    THIS DOES NOT PROVE SEMANTIC CORRECTNESS. It is a text-containment
    check only: a correct paraphrase that never states the gold entity
    verbatim (e.g. gold "Google" answered as "the tech giant") produces a
    FALSE NEGATIVE here, not evidence the answer is wrong — see
    tests/test_task_success.py's paraphrase case. It is reported
    alongside, never in place of, the judge grade.

    Returns `None` if the gold answer normalizes to zero tokens (should
    not occur in practice, but never silently treated as a match)."""
    gold_tokens = tuple(normalize_answer_text(gold_answer).split())
    if not gold_tokens:
        return None
    pred_tokens = normalize_answer_text(generated_answer).split()
    n = len(gold_tokens)
    return any(tuple(pred_tokens[i : i + n]) == gold_tokens for i in range(len(pred_tokens) - n + 1))


# --- abstention: the full 2x2, not just "correct abstention on null_query" --------------

ABSTENTION_STATUSES: tuple[str, ...] = (
    "correct_abstention",  # null_query, abstained — the one cell the rest of this project already measures
    "incorrect_abstention",  # non-null, abstained — wrongly declined an answerable question
    "hallucinated_non_abstention",  # null_query, did NOT abstain — stated a specific answer to an unanswerable question
    "normal_non_abstention",  # non-null, did NOT abstain — the expected case; content correctness is evaluated separately
)


@dataclass(frozen=True, slots=True)
class AbstentionClassification:
    is_abstention: bool
    status: str  # one of ABSTENTION_STATUSES


def classify_abstention(question_type: str, generated_answer: str) -> AbstentionClassification:
    """Extends `mhrag.eval.answer_metrics.is_abstention` — previously
    applied only to null_query records throughout this project — to the
    full population, producing all four abstention x question-nullity
    cells as separate, never-collapsed states (see module docstring).

    UNCHANGED SINCE PHASE 2 — see `classify_response_structure` below for
    the Phase 3 hardening. `is_abstention()` itself is not modified here
    (nor could it be — it lives in `mhrag.eval.answer_metrics` and is used
    project-wide for every existing null_query metric); this function's
    output, `ABSTENTION_STATUSES`, and every `abstention_status` value
    already persisted in `results/task_success_report.json` from Phase 2
    are BYTE-IDENTICAL under Phase 3. The richer signal lives entirely in
    the new, separate `response_structure` field."""
    abstained = is_abstention(generated_answer)
    if question_type == "null_query":
        status = "correct_abstention" if abstained else "hallucinated_non_abstention"
    else:
        status = "incorrect_abstention" if abstained else "normal_non_abstention"
    return AbstentionClassification(is_abstention=abstained, status=status)


# --- response structure (Phase 3): a richer, SEPARATE signal from abstention_status ------
#
# WHY THIS EXISTS: `is_abstention()` (unchanged, above) is a single-phrase substring check —
# it cannot distinguish a clean refusal from a hedge that still substantively answers, e.g.
# "Insufficient information to answer the question. While Source 1 identifies Travis Kelce
# as..." (a real Phase 2 validation-sample case) was flagged `incorrect_abstention` even
# though the model went on to correctly name the entity. This module does NOT change
# `abstention_status` to fix that (see docstring above) — it adds `response_structure`, a
# genuinely different, structural signal, computed from the FULL response rather than one
# phrase's presence.
#
# ALGORITHM: strip every known decline/hedge PHRASE (not single trigger words — full
# templated spans, e.g. "insufficient information to answer", so a clean abstention's
# boilerplate leaves ~0 residue) from the text, from wherever it occurs (start, middle, or
# end — a real answer can precede its hedge, e.g. "Google, although the evidence is
# limited.", or follow it, e.g. "Insufficient information; however, the answer is Google.").
# What remains, after also discarding a small set of generic connective/filler words, is
# checked for at least one substantive token. This handles either sentence order uniformly,
# unlike a directional "look only after the marker" design would.
#
# DECLINE_PHRASES is intentionally a RICHER set than `ABSTENTION_PHRASES`
# (mhrag.eval.answer_metrics) — including real paraphrases the original list misses (e.g.
# "don't have enough information") — but this richer set is used ONLY here, never by
# `is_abstention()`/`classify_abstention()` above, so it can never change
# `abstention_status`'s meaning. This is "design a richer classification", not "expand the
# allowlist" the existing metric already uses.
#
# PHASE 4 REDESIGN, WHY IT WAS NECESSARY: Phase 3's algorithm (stripping decline/hedge
# phrases, then checking whether ANY non-generic word survived) was validated against 6
# random real dev-split records that had been flagged by abstention_status. 5/6 were false
# positives: clean refusals like "The available information does not include any article
# from 'The Verge'... insufficient to answer" were misread as `answer_with_uncertainty`
# purely because named sources/topics mentioned AS PART OF EXPLAINING THEIR ABSENCE (The
# Verge, Polygon, Fortune, ...) survived the generic-word filter. A residue-word-count
# heuristic cannot distinguish "names a source that IS the answer" from "names a source
# cited as absent" — that is a negation-scope problem, not a vocabulary problem, and no
# amount of allowlist/blocklist tuning fixes it (per the explicit instruction: do not
# attempt another generic residue-word heuristic).
#
# THE PHASE 4 APPROACH instead requires a genuinely STRUCTURAL signal: an explicit
# assertion marker ("the answer is", "indicates", "confirms", ...) immediately followed, in
# the ORIGINAL case-preserved text, by a CAPITALIZED token — a cheap, orthographic proxy for
# "this names an actual entity being asserted as the answer," not "some word happens to
# survive." Re-validated against the same 6-record sample this design was built to fix (see
# scripts/task_success_hardening_error_analysis.py's Phase 4 section): 6/6 correct.
#
# THIS IS STILL NOT VALIDATED AT SCALE. Six records is not proof of general reliability —
# see RESPONSE_STRUCTURE_RELIABILITY below. It is a documented, deliberately conservative
# heuristic with a KNOWN false-negative mode (a bare entity name stated BEFORE its hedge,
# e.g. "Google, although the evidence is limited." — no assertion marker precedes "Google,"
# so this resolves to clean_abstention/ambiguous, incorrectly). Accepted rather than patched
# with a fragile sentence-initial-capitalization rule, which would reintroduce a different
# false-positive risk (ordinary sentence-initial capitalization on "The"/"It"/"This" is not
# a reliable entity signal at all, unlike genuine mid-sentence capitalization).

RESPONSE_STRUCTURE_RELIABILITY = "EXPERIMENTAL"
"""Explicit reliability status for `classify_response_structure`'s output. NOT "TRUSTED" —
see TRUSTED_SIGNALS / EXPERIMENTAL_SIGNALS below. `response_structure` is reported as an
INFORMATIONAL signal only: it must never be read into `deterministic_correctness`,
`task_success_confident`, or any Task Success score (and structurally cannot be — see
`classify_task_success`, which never passes `response_structure` into any of those
computations)."""

RESPONSE_STRUCTURE_STATES: tuple[str, ...] = (
    "clean_abstention",  # a decline phrase is present and no explicit assertion of a named answer was found
    "answer_with_uncertainty",  # a decline or hedge phrase is present, AND an explicit assertion names an answer
    "substantive_answer",  # no decline/hedge language detected at all
    "ambiguous",  # hedge language present, no decline phrase, and no explicit assertion found
)

DECLINE_PHRASES: tuple[str, ...] = (
    "the available information is insufficient to answer",
    "insufficient information to answer",
    "insufficient to answer",
    "insufficient",
    "not enough information to answer",
    "not enough information",
    "does not contain enough information to answer",
    "does not contain enough information",
    "no information is available to answer",
    "no information",
    "cannot be determined from the available information",
    "cannot be determined",
    "cannot determine",
    "don't have enough information",
    "do not have enough information",
    "unable to determine",
    "unable to answer",
)
HEDGE_PHRASES: tuple[str, ...] = (
    "cannot be certain",
    "can't be certain",
    "is limited",
    "are limited",
    "not entirely clear",
    "is unclear",
    "are unclear",
)
_WHITESPACE_RE_LOCAL = re.compile(r"\s+")

# STRUCTURAL signal (Phase 4), replacing Phase 3's word-bag residue count: an explicit
# assertion marker, curated from this project's own generation style (plain prose, per
# mhrag.generation.prompts.SYSTEM_PROMPT_V1) plus the one real hedge-then-answer pattern
# observed during validation ("is described as"). Matched case-insensitively; what matters
# is what's found in a short window AFTER the marker, in the ORIGINAL case.
_ASSERTION_MARKERS: tuple[str, ...] = (
    "the answer is", "indicates", "indicate", "suggests", "suggest", "confirms", "confirm",
    "reveals", "reveal", "shows that", "show that", "states that", "state that",
    "is described as", "are described as", "identifies", "identify", "attributes", "attribute",
)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z'\-]*\b")
_ASSERTION_LOOKAHEAD_WINDOW = 60  # characters — bounded so a marker far from any real subject doesn't match


def _has_explicit_named_assertion(original_text: str) -> bool:
    """True only when an assertion marker is immediately followed (within
    `_ASSERTION_LOOKAHEAD_WINDOW` characters) by a capitalized token in
    the ORIGINAL, case-preserved text — a proxy for "this names an actual
    entity as the answer," not "some word happens to survive stripping."
    See the module comment above `RESPONSE_STRUCTURE_RELIABILITY` for why
    this replaced Phase 3's approach and what it still misses."""
    lowered = original_text.lower()
    for marker in _ASSERTION_MARKERS:
        for m in re.finditer(re.escape(marker), lowered):
            window = original_text[m.end() : m.end() + _ASSERTION_LOOKAHEAD_WINDOW]
            if _PROPER_NOUN_RE.search(window):
                return True
    return False


def classify_response_structure(generated_answer: str) -> str:
    """One of `RESPONSE_STRUCTURE_STATES`. EXPERIMENTAL — see
    `RESPONSE_STRUCTURE_RELIABILITY` and the module comment above it.
    Pure text analysis — no judge, no evidence coverage, no
    question_type — so it can never be "repaired" by another signal, per
    the explicit methodological requirement that the deterministic
    evaluator stay independent. Its output is informational only and is
    never read by `classify_task_success` into `deterministic_correctness`
    or `task_success_confident`."""
    text = generated_answer.strip()
    if not text:
        return "ambiguous"

    lowered = _WHITESPACE_RE_LOCAL.sub(" ", text.lower())
    has_decline = any(phrase in lowered for phrase in DECLINE_PHRASES)
    has_hedge = any(phrase in lowered for phrase in HEDGE_PHRASES)

    if not has_decline and not has_hedge:
        return "substantive_answer"

    if _has_explicit_named_assertion(text):
        return "answer_with_uncertainty"
    if has_decline:
        return "clean_abstention"
    return "ambiguous"


# --- unsupported: judge grade x evidence coverage ONLY, independent of the deterministic checks above ---

_JUDGE_GRADES_COUNTING_AS_APPARENTLY_CORRECT = frozenset({"correct", "partially_correct"})


def compute_unsupported(judge_grade: str | None, evidence_coverage: float | None) -> bool | None:
    """Grounding flag, built ONLY from the existing judge grade and the
    existing evidence-coverage metric — deliberately independent of
    `verdict_match`/`entity_containment` above, per the explicit
    methodology requirement that these four signals stay separate.

    True: the judge graded the answer correct/partially_correct, but
      evidence_coverage is exactly 0.0 — none of the required gold
      documents were even retrieved, so a "correct-looking" answer is
      most likely explained by the generation model's own prior/
      parametric knowledge rather than the provided context, directly
      contradicting the generation prompt's "use ONLY the provided
      context" instruction.
    False: either the judge graded it incorrect (an ungrounded WRONG
      answer isn't a new, separate concern — it's just wrong), or
      evidence_coverage > 0 (at least some required evidence was present).
    None: judge_grade or evidence_coverage is unavailable (e.g.
      null_query, which the judge never grades and which has no gold
      documents to compute coverage from) — "not applicable," never
      silently treated as False.
    """
    if judge_grade is None or evidence_coverage is None:
        return None
    if judge_grade not in _JUDGE_GRADES_COUNTING_AS_APPARENTLY_CORRECT:
        return False
    return evidence_coverage == 0.0


# --- top-level assembly ------------------------------------------------------------------

# PHASE 4: explicit TRUSTED vs EXPERIMENTAL signal separation, per the audit requirement
# that an experimental heuristic must never silently become a correctness label. This is a
# DOCUMENTATION/CONTRACT declaration — `classify_task_success` already never reads
# `response_structure` into `deterministic_correctness`/`task_success_confident`/`unsupported`
# (see its body), so nothing downstream needs to change to honor this; these two tuples make
# the boundary explicit and inspectable rather than implicit in the code's control flow alone.
TRUSTED_SIGNALS: tuple[str, ...] = (
    "abstention_status",  # is_abstention() unchanged since Phase 2; the 2x2 classification itself is deterministic and exhaustive
    "deterministic_correctness",  # hardened (Phase 3 + Phase 4) verdict_match; explicit_ambiguous rather than guessed
    "judge_grade",  # existing LLM judge, passed through unchanged — trusted as "a second, independent opinion," not ground truth
    "judge_score",
    "evidence_coverage",  # existing doc-id-overlap computation, unchanged
    "unsupported",  # deterministic cross-tab of judge_grade x evidence_coverage only
)
EXPERIMENTAL_SIGNALS: tuple[str, ...] = (
    "response_structure",  # see RESPONSE_STRUCTURE_RELIABILITY — validated on n=6 real records only; informational, not a score input
    "entity_containment_match",  # always was documented as "not proof of correctness" (Phase 2); grouped here for the same reason
)
"""Every field in `TaskSuccessResult` not covered by both tuples above is either a
pass-through of an already-trusted input (question_type, gold_verdict, extracted_verdict —
diagnostic, not scores) or a meta field (judge_deterministic_agree, task_success_confident)
that is itself computed ONLY from TRUSTED_SIGNALS."""

DETERMINISTIC_MATCH_TYPES: tuple[str, ...] = (
    "abstention_only",  # abstention_status != normal_non_abstention: content correctness not separately evaluated
    "verdict_match",  # comparison_query / temporal_query, normal_non_abstention
    "entity_containment_secondary",  # inference_query, normal_non_abstention — NEVER feeds deterministic_correctness
    "not_available",  # a question_type this module doesn't have a deterministic check for
)


@dataclass(frozen=True, slots=True)
class TaskSuccessResult:
    """Every signal as its own field — see module docstring. No field
    here is derived by letting one signal silently overwrite another."""

    question_type: str

    abstention_status: str  # one of ABSTENTION_STATUSES — UNCHANGED since Phase 2, see classify_abstention's docstring
    is_abstention: bool

    response_structure: str  # one of RESPONSE_STRUCTURE_STATES — NEW in Phase 3, additive, never overwrites abstention_status

    deterministic_match_type: str  # one of DETERMINISTIC_MATCH_TYPES
    deterministic_correctness: str  # "correct" | "incorrect" | "ambiguous" | "not_applicable"
    extracted_verdict: str | None
    gold_verdict: str | None

    entity_containment_match: bool | None  # inference_query only; see entity_containment's docstring — NOT a correctness proof

    judge_grade: str | None
    judge_score: float | None

    evidence_coverage: float | None
    unsupported: bool | None

    judge_deterministic_agree: bool | None  # None when not comparable (see classify_task_success)
    task_success_confident: bool


def classify_task_success(
    question_type: str,
    gold_answer: str,
    generated_answer: str,
    judge_grade: str | None = None,
    judge_score: float | None = None,
    evidence_coverage: float | None = None,
) -> TaskSuccessResult:
    """Assemble one `TaskSuccessResult`. Pure function: no I/O, no model
    call, no randomness — the same inputs always produce the same output
    (see tests/test_task_success.py::test_deterministic_repeatability).

    `judge_grade`/`judge_score`/`evidence_coverage` are OPTIONAL,
    ALREADY-COMPUTED inputs from elsewhere (the existing judge pipeline,
    the existing evidence-coverage computation) — this function never
    calls a judge or computes coverage itself, and never lets a supplied
    judge_grade change `deterministic_correctness` (see
    `judge_deterministic_agree` for how disagreement is surfaced instead
    of silently resolved).
    """
    abstention = classify_abstention(question_type, generated_answer)
    response_structure = classify_response_structure(generated_answer)

    deterministic_match_type = "abstention_only"
    deterministic_correctness = "not_applicable"
    extracted_verdict: str | None = None
    gold_verdict: str | None = None
    entity_match: bool | None = None

    if abstention.status == "normal_non_abstention":
        if question_type in ("comparison_query", "temporal_query"):
            deterministic_match_type = "verdict_match"
            result = verdict_match(gold_answer, generated_answer)
            deterministic_correctness = result.correctness
            extracted_verdict = result.extracted_verdict
            gold_verdict = result.gold_verdict
        elif question_type == "inference_query":
            deterministic_match_type = "entity_containment_secondary"
            entity_match = entity_containment(gold_answer, generated_answer)
            deterministic_correctness = "not_applicable"  # explicitly never claims correctness from containment alone
        else:
            deterministic_match_type = "not_available"
            deterministic_correctness = "not_applicable"

    unsupported = compute_unsupported(judge_grade, evidence_coverage)

    judge_deterministic_agree: bool | None = None
    if judge_grade is not None and deterministic_correctness in ("correct", "incorrect"):
        judge_deterministic_agree = (deterministic_correctness == "correct" and judge_grade == "correct") or (
            deterministic_correctness == "incorrect" and judge_grade == "incorrect"
        )

    task_success_confident = abstention.status != "normal_non_abstention" or deterministic_correctness in (
        "correct",
        "incorrect",
    )

    return TaskSuccessResult(
        question_type=question_type,
        abstention_status=abstention.status,
        is_abstention=abstention.is_abstention,
        response_structure=response_structure,
        deterministic_match_type=deterministic_match_type,
        deterministic_correctness=deterministic_correctness,
        extracted_verdict=extracted_verdict,
        gold_verdict=gold_verdict,
        entity_containment_match=entity_match,
        judge_grade=judge_grade,
        judge_score=judge_score,
        evidence_coverage=evidence_coverage,
        unsupported=unsupported,
        judge_deterministic_agree=judge_deterministic_agree,
        task_success_confident=task_success_confident,
    )
