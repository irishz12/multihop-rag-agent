// Every number rendered on the page passes through one of these — never a
// hand-typed "80.0%" or "$0.0009" in a component. Formatting only; no
// derivation happens here (see lib/data.ts for that).

export function formatPercent(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function formatScore(value: number, digits = 3): string {
  return value.toFixed(digits);
}

export function formatUsd(value: number): string {
  return `$${value.toFixed(6)}`;
}

export function formatMs(value: number): string {
  return `${Math.round(value).toLocaleString("en-US")} ms`;
}

export function formatSeconds(msValue: number): string {
  return `${(msValue / 1000).toFixed(1)}s`;
}

export function formatInt(value: number): string {
  return value.toLocaleString("en-US");
}

/**
 * Round, evenly-spaced axis ticks from 0 up to (at least) `maxValue` — the
 * classic 1/2/5-per-decade "nice number" rule, so a chart's gridlines read
 * as 2,500 / 5,000 / 7,500 / 10,000 rather than the arithmetic-but-ugly
 * 2,778 / 5,556 / 8,333 a plain even split of the real maximum would give.
 */
export function niceTicks(maxValue: number, targetCount = 4): number[] {
  if (maxValue <= 0) return [0];
  const roughStep = maxValue / targetCount;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const niceStep = (normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10) * magnitude;

  // Always push at least one more tick than the raw max needs, so a bar
  // sitting exactly at the real maximum never touches (let alone exceeds)
  // the top of the chart.
  const ticks: number[] = [0];
  while (ticks[ticks.length - 1] < maxValue) {
    ticks.push(Number((ticks[ticks.length - 1] + niceStep).toFixed(10)));
  }
  return ticks;
}

export function titleCaseQuestionType(key: string): string {
  const map: Record<string, string> = {
    inference_query: "Inference",
    comparison_query: "Comparison",
    temporal_query: "Temporal",
    null_query: "Null",
  };
  return map[key] ?? key;
}

export function titleCaseHopBucket(key: string): string {
  const map: Record<string, string> = {
    hop2: "2-hop",
    hop3: "3-hop",
    hop4: "4-hop",
    null: "Null",
  };
  return map[key] ?? key;
}

export function titleCaseStopReason(key: string): string {
  const map: Record<string, string> = {
    evidence_sufficient: "Evidence sufficient",
    max_hops: "Max hops reached",
    token_budget: "Token budget reached",
    duplicate_query: "Duplicate query",
    timeout: "Timeout",
  };
  return map[key] ?? key;
}

export const PIPELINE_LABEL: Record<string, string> = {
  dense: "Dense RAG",
  hybrid: "Hybrid RAG",
  hybrid_reranker: "Hybrid + Reranker",
  agentic_multi_hop: "Agentic Multi-Hop RAG",
  adaptive_rag: "Adaptive RAG",
};

// Fixed, never re-cycled per chart — the same identity color for a given
// pipeline everywhere on the page, matching the README's charts exactly
// (Okabe-Ito colorblind-safe blue/orange for the two systems under primary
// comparison; a light-to-dark gray ramp for the three single-pass baselines).
export const PIPELINE_COLOR: Record<string, string> = {
  dense: "#B0B0B0",
  hybrid: "#808080",
  hybrid_reranker: "#4D4D4D",
  agentic_multi_hop: "#0F5AA8",
  adaptive_rag: "#B8620A",
};
