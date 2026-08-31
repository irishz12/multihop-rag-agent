"""EVALUATOR-ONLY: reusable statistical utilities for Task Success
reporting — proportions, Wilson confidence intervals, paired bootstrap
CIs/deltas/effect sizes.

`mhrag.eval` had NO statistical tooling before this file. Every other
statistical computation in this project (the router's threshold
selection, this session's context-matched-ablation analysis scripts) was
written ad hoc, once, inline in a `scripts/*.py` file. This module
generalizes that same ad hoc bootstrap/Cohen's-d code (originally in
`scripts/analyze_context_matched_ablation.py`) into one tested,
reusable implementation — it does not change or reinterpret any existing
experiment's already-reported numbers; it only gives Task Success
reporting (and any future caller) a shared implementation instead of
another one-off copy.

NO ARBITRARY SIGNIFICANCE THRESHOLD is introduced here. This module
returns intervals and deltas only — never a boolean "is_significant"
verdict. Whether an interval excludes zero is a fact a caller can compute
by inspecting the returned bounds; this module does not editorialize.

Multiple-comparison correction (`bonferroni_alpha`) requires the caller
to state the comparison family size explicitly — this module never
infers or assumes how many comparisons are being made.
"""

from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass

# Same seed this project's own evaluation sampling already uses
# (mhrag.data.sampling / configs/dataset.yaml's stratified-sample seed,
# and this session's own ablation analysis scripts) — reused here for
# consistency across the project's statistical work, not a new arbitrary
# choice invented for this module.
DEFAULT_BOOTSTRAP_SEED = 2029
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

# Two-sided normal-distribution critical values for the confidence levels
# this project actually uses — an explicit lookup, not a scipy dependency
# (this project has never depended on scipy; see mhrag.routing's own
# scikit-learn-only-at-fit-time pattern for the same "no extra runtime
# dependency" discipline).
_Z_FOR_CONFIDENCE: dict[float, float] = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.975: 2.2414027276848015,  # standard two-sided critical value for alpha=0.025 — needed for a
    # Bonferroni-corrected pair of primary comparisons (n=2, nominal alpha 0.05 -> 0.025 each), NOT
    # an arbitrary addition: this is the same well-known tabulated constant as the other three entries.
    0.99: 2.5758293035489004,
}


def proportion(successes: int, n: int) -> float:
    """`successes / n`. Raises ValueError on `n == 0` rather than
    returning a silently meaningless 0.0/NaN."""
    if n == 0:
        raise ValueError("proportion() requires n > 0")
    return successes / n


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    lower: float
    upper: float
    point_estimate: float
    n: int
    confidence: float


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> WilsonInterval:
    """Wilson score interval for a binomial proportion — better-behaved
    than a naive normal-approximation interval at small n or when the
    proportion is near 0 or 1 (both routine here: Task Success
    per-question-type cells are small, and abstention-correctness rates
    are frequently exactly 1.0 in this project's existing results).

    `confidence` must be one of the explicit values in `_Z_FOR_CONFIDENCE`
    — an unsupported value raises rather than silently approximating a
    z-score.
    """
    if n == 0:
        raise ValueError("wilson_ci() requires n > 0")
    if confidence not in _Z_FOR_CONFIDENCE:
        raise ValueError(f"unsupported confidence level {confidence!r} — supported: {sorted(_Z_FOR_CONFIDENCE)}")
    z = _Z_FOR_CONFIDENCE[confidence]
    phat = successes / n

    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    adjustment = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lower = (center - adjustment) / denom
    upper = (center + adjustment) / denom
    return WilsonInterval(
        lower=max(0.0, lower), upper=min(1.0, upper), point_estimate=phat, n=n, confidence=confidence
    )


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    seed: int


def paired_bootstrap_ci(
    deltas: list[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI on the mean of `deltas` (one paired
    per-question difference per element). Same method this session's
    `scripts/analyze_context_matched_ablation*.py` already used —
    generalized here rather than re-derived. Deterministic given the same
    (deltas, seed): uses `random.Random(seed)`, never the shared global
    RNG, so repeated calls in the same process never interfere with each
    other's sequence."""
    if len(deltas) < 2:
        raise ValueError("paired_bootstrap_ci() requires at least 2 paired observations")
    if confidence not in _Z_FOR_CONFIDENCE:
        raise ValueError(f"unsupported confidence level {confidence!r} — supported: {sorted(_Z_FOR_CONFIDENCE)}")

    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()

    alpha = 1 - confidence
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return BootstrapCI(
        lower=means[lo_idx], upper=means[hi_idx], confidence=confidence, n_resamples=n_resamples, seed=seed
    )


@dataclass(frozen=True, slots=True)
class PairedDeltaSummary:
    n: int
    mean_delta: float
    median_delta: float
    stdev_delta: float | None  # None when n < 2 (stdev undefined)
    ci: BootstrapCI | None  # None when n < 2
    cohens_d: float | None  # None when n < 2 or stdev == 0


def cohens_d_paired(deltas: list[float]) -> float | None:
    """Paired Cohen's d: mean(deltas) / stdev(deltas). `None` when
    undefined (n < 2, or zero variance — an exact tie across every paired
    observation, which this project's own ablation work has actually
    observed for at least one comparison; see
    scripts/analyze_context_matched_ablation.py's matched_vs_baseline5
    result)."""
    if len(deltas) < 2:
        return None
    sd = st.stdev(deltas)
    if sd == 0:
        return None
    return st.mean(deltas) / sd


def paired_delta_summary(
    a_values: list[float],
    b_values: list[float],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedDeltaSummary:
    """Full paired-comparison summary for `a - b`, one delta per matched
    (qa_id-aligned) pair — the caller is responsible for ensuring
    `a_values[i]`/`b_values[i]` are the same question, same order (this
    function has no notion of qa_id; see scripts/compute_task_success.py
    for how alignment is enforced upstream)."""
    if len(a_values) != len(b_values):
        raise ValueError(f"paired_delta_summary() requires equal-length inputs, got {len(a_values)} vs {len(b_values)}")
    deltas = [a - b for a, b in zip(a_values, b_values)]
    n = len(deltas)
    if n < 2:
        return PairedDeltaSummary(n=n, mean_delta=(deltas[0] if deltas else 0.0), median_delta=(deltas[0] if deltas else 0.0), stdev_delta=None, ci=None, cohens_d=None)

    return PairedDeltaSummary(
        n=n,
        mean_delta=st.mean(deltas),
        median_delta=st.median(deltas),
        stdev_delta=st.stdev(deltas),
        ci=paired_bootstrap_ci(deltas, n_resamples=n_resamples, seed=seed, confidence=confidence),
        cohens_d=cohens_d_paired(deltas),
    )


def bonferroni_alpha(nominal_alpha: float, n_comparisons: int) -> float:
    """`nominal_alpha / n_comparisons`. The caller MUST state
    `n_comparisons` explicitly (the size of the pre-specified comparison
    family) — this function never infers it, per the "explicitly define
    the comparison family before calculating corrected intervals"
    requirement."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    return nominal_alpha / n_comparisons


def excludes_zero(lower: float, upper: float) -> bool:
    """Plain factual check on an already-computed interval — NOT a
    significance verdict, just whether 0.0 falls inside [lower, upper].
    Named as a descriptive predicate, deliberately not `is_significant`,
    to avoid implying a p-value-style judgment this module does not
    make."""
    return not (lower <= 0.0 <= upper)
