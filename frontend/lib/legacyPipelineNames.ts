// The ONE place this frontend translates between its current canonical
// pipeline names and the legacy names baked into already-frozen Phase 9
// result artifacts under ../results/ (phase9_sample_report.json,
// phase9_holdout_report.json). Those artifacts are immutable — see the
// project README's Reproducibility section — so lib/data.ts reads their
// raw legacy-keyed shape and rekeys it through this module, once, right
// after loading. Every component downstream of lib/data.ts sees only
// canonical pipeline names.
//
//     agentic_multi_hop -> always_agentic   (the bounded agent loop, always run)
//     adaptive_rag       -> adaptive         (the learned-router cost-aware baseline)
//
// This mirrors mhrag.eval.legacy_pipeline_names on the Python side exactly.

export const CANONICAL_TO_LEGACY = {
  agentic_multi_hop: "always_agentic",
  adaptive_rag: "adaptive",
} as const;

export type CanonicalName = keyof typeof CANONICAL_TO_LEGACY;
export type LegacyName = (typeof CANONICAL_TO_LEGACY)[CanonicalName];

const LEGACY_TO_CANONICAL: Record<string, string> = Object.fromEntries(
  Object.entries(CANONICAL_TO_LEGACY).map(([canonical, legacy]) => [legacy, canonical]),
);

/** A name read from a frozen results/*.json key -> this project's current
 * canonical pipeline name (returned unchanged if not one of the two
 * renamed pipelines). */
export function toCanonicalName(name: string): string {
  return LEGACY_TO_CANONICAL[name] ?? name;
}

/** Rekey a dict directly keyed by pipeline name at its top level (e.g.
 * `combined_quality_mean`, `evidence_coverage_mean`) so `always_agentic`
 * and `adaptive` become their canonical names. Every other key, and every
 * value, passes through unchanged. */
export function rekeyLegacyReport<T>(report: Record<string, T>): Record<string, T> {
  const rekeyed: Record<string, T> = {};
  for (const [key, value] of Object.entries(report)) {
    rekeyed[toCanonicalName(key)] = value;
  }
  return rekeyed;
}

/** Rekey a flat dict whose keys are PREFIXED with a pipeline name (e.g.
 * `cost_latency`'s `always_agentic_mean_cost_usd`, or one breakdown
 * group's `adaptive_mean_quality`) so any key beginning with a legacy
 * pipeline name is rewritten to begin with the canonical name instead. */
export function rekeyLegacyPrefixedKeys<T>(d: Record<string, T>): Record<string, T> {
  const rekeyed: Record<string, T> = {};
  for (const [key, value] of Object.entries(d)) {
    let newKey = key;
    for (const [canonical, legacy] of Object.entries(CANONICAL_TO_LEGACY)) {
      const prefix = `${legacy}_`;
      if (key.startsWith(prefix)) {
        newKey = `${canonical}_${key.slice(prefix.length)}`;
        break;
      }
    }
    rekeyed[newKey] = value;
  }
  return rekeyed;
}
