import "server-only";

import fs from "node:fs";
import path from "node:path";

import { rekeyLegacyPrefixedKeys, rekeyLegacyReport } from "./legacyPipelineNames";
import type {
  BaselineCostLatency,
  BreakdownRow,
  CostLatency,
  HoldoutConsumed,
  HoldoutReport,
  RetrievalEval,
  RouterModel,
  SampleReport,
  UnderRoutedFailure,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "..", "data", "processed");

/**
 * READ-ONLY access to this project's real, already-measured result
 * artifacts (`../results/*.json`, produced by the Python evaluation
 * pipeline — never by this frontend). Nothing in this module writes to
 * disk, calls a network API, or constructs an LLM client — it only parses
 * JSON that already exists on disk at build time. There is no code path
 * here that can reach `data/processed/final_holdout.json` itself (that
 * file was consumed once, by the evaluation scripts, months before this
 * frontend existed) — only the aggregated *reports* derived from it.
 *
 * `RESULTS_DIR` points one level up from the Next.js app root, at the
 * same `results/` directory the rest of this repository already uses as
 * its single source of truth — intentionally not copied or duplicated
 * into `frontend/`, so there is exactly one place these numbers can live.
 *
 * The Phase 9 reports below still use this project's two legacy pipeline
 * names (`always_agentic`, `adaptive`) internally — those files are
 * immutable, so `getSampleReport`/`getHoldoutReport` are the one place
 * this frontend translates them to canonical names
 * (`lib/legacyPipelineNames.ts`), immediately after parsing. Every
 * component that imports from this module sees only canonical names.
 */
const RESULTS_DIR = path.join(process.cwd(), "..", "results");

function readJSON<T>(filename: string): T {
  const filePath = path.join(RESULTS_DIR, filename);
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

/** The on-disk shape of phase9_sample_report.json / phase9_holdout_report.json —
 * legacy pipeline names, exactly as the frozen file has them. Not exported;
 * getSampleReport/getHoldoutReport are the only callers, and they rekey
 * this to the canonical SampleReport/HoldoutReport shape before returning. */
interface RawPhase9Report {
  [key: string]: unknown;
  deterministic_metrics: Record<string, unknown>;
  judge_scores: Record<string, unknown>;
  combined_quality_mean: Record<string, number>;
  evidence_coverage_mean: Record<string, number>;
  cost_latency: Record<string, number>;
  breakdown_by_question_type: Record<string, Record<string, unknown>>;
  breakdown_by_hop_count: Record<string, Record<string, unknown>>;
  under_routed_failures: Array<Record<string, unknown>>;
}

function rekeyBreakdown(breakdown: Record<string, Record<string, unknown>>): Record<string, BreakdownRow> {
  const rekeyed: Record<string, BreakdownRow> = {};
  for (const [group, row] of Object.entries(breakdown)) {
    rekeyed[group] = rekeyLegacyPrefixedKeys(row) as unknown as BreakdownRow;
  }
  return rekeyed;
}

/** Rekey every legacy-named field a raw Phase 9 report shares between the
 * development-sample and holdout reports. Retention's one compound key
 * (`adaptive_quality_retention_pct_vs_always_agentic`) is renamed by its
 * caller, since its canonical name differs slightly between call sites. */
function rekeyPhase9Report(raw: RawPhase9Report) {
  return {
    ...raw,
    deterministic_metrics: rekeyLegacyReport(raw.deterministic_metrics),
    judge_scores: rekeyLegacyReport(raw.judge_scores),
    combined_quality_mean: rekeyLegacyReport(raw.combined_quality_mean),
    evidence_coverage_mean: rekeyLegacyReport(raw.evidence_coverage_mean),
    cost_latency: rekeyLegacyPrefixedKeys(raw.cost_latency) as unknown as CostLatency,
    breakdown_by_question_type: rekeyBreakdown(raw.breakdown_by_question_type),
    breakdown_by_hop_count: rekeyBreakdown(raw.breakdown_by_hop_count),
    under_routed_failures: raw.under_routed_failures.map(
      (failure) => rekeyLegacyPrefixedKeys(failure) as unknown as UnderRoutedFailure,
    ),
  };
}

export function getSampleReport(): SampleReport {
  const raw = readJSON<RawPhase9Report & { adaptive_quality_retention_pct_vs_always_agentic: number }>(
    "phase9_sample_report.json",
  );
  const { adaptive_quality_retention_pct_vs_always_agentic, ...rest } = raw;
  return {
    ...rekeyPhase9Report(rest),
    adaptive_quality_retention_pct_vs_agentic_multi_hop: adaptive_quality_retention_pct_vs_always_agentic,
  } as unknown as SampleReport;
}

export function getHoldoutReport(): HoldoutReport {
  const raw = readJSON<
    RawPhase9Report & {
      adaptive_quality_retention_pct_vs_always_agentic: number;
      development_vs_holdout: {
        development: Record<string, unknown> & { combined_quality_mean: Record<string, number>; evidence_coverage_mean: Record<string, number> };
        holdout: Record<string, unknown> & { combined_quality_mean: Record<string, number>; evidence_coverage_mean: Record<string, number> };
      };
    }
  >("phase9_holdout_report.json");
  const { adaptive_quality_retention_pct_vs_always_agentic, development_vs_holdout, ...rest } = raw;
  return {
    ...rekeyPhase9Report(rest),
    adaptive_quality_retention_pct_vs_agentic_multi_hop: adaptive_quality_retention_pct_vs_always_agentic,
    development_vs_holdout: {
      development: {
        ...development_vs_holdout.development,
        combined_quality_mean: rekeyLegacyReport(development_vs_holdout.development.combined_quality_mean),
        evidence_coverage_mean: rekeyLegacyReport(development_vs_holdout.development.evidence_coverage_mean),
      },
      holdout: {
        ...development_vs_holdout.holdout,
        combined_quality_mean: rekeyLegacyReport(development_vs_holdout.holdout.combined_quality_mean),
        evidence_coverage_mean: rekeyLegacyReport(development_vs_holdout.holdout.evidence_coverage_mean),
      },
    },
  } as unknown as HoldoutReport;
}

export function getRetrievalEval(): RetrievalEval {
  return readJSON<RetrievalEval>("retrieval_eval_development.json");
}

export function getRouterModel(): RouterModel {
  return readJSON<RouterModel>("learned_router_model.json");
}

export function getHoldoutConsumed(): HoldoutConsumed {
  return readJSON<HoldoutConsumed>("final_holdout_consumed.json");
}

/**
 * Dense/Hybrid/Hybrid+Reranker don't get their own aggregated cost/latency
 * fields in phase9_sample_report.json (only Agentic Multi-Hop RAG/Adaptive
 * RAG do, since those are this project's primary comparison) — so, exactly
 * like `scripts/generate_phase9_charts.py` did for the README charts, this
 * derives their mean cost/query and latency/query directly from the raw
 * per-question traces, filtered to the same frozen 50-question sample.
 * Every number here is a plain arithmetic mean over real recorded fields
 * — nothing invented, nothing hardcoded.
 */
export function getDevBaselineCostLatency(): Record<
  "dense" | "hybrid" | "hybrid_reranker",
  BaselineCostLatency
> {
  const sample = readJSON<{ qa_ids: string[] }>("phase9_sample.json");
  const sampleIds = new Set(sample.qa_ids);

  const pipelines = ["dense", "hybrid", "hybrid_reranker"] as const;
  const result = {} as Record<(typeof pipelines)[number], BaselineCostLatency>;

  for (const pipeline of pipelines) {
    const raw = readJSON<{
      records: Array<{ qa_id: string; total_cost_usd: number; total_latency_ms: number }>;
    }>(`phase9_${pipeline}_raw.json`);
    const records = raw.records.filter((r) => sampleIds.has(r.qa_id));
    const n = records.length;
    result[pipeline] = {
      cost: records.reduce((sum, r) => sum + r.total_cost_usd, 0) / n,
      latency: records.reduce((sum, r) => sum + r.total_latency_ms, 0) / n,
    };
  }

  return result;
}

/**
 * A handful of REAL questions from data/processed/dev_subset.json — the
 * development split, never final_holdout — for the live demo's example
 * chips. Selected deterministically (fixed indices into the non-null,
 * question-type-sorted list, capped to a readable length) so the same
 * five questions appear on every build; nothing here is written by hand.
 */
export function getExampleQuestions(): string[] {
  const filePath = path.join(DATA_DIR, "dev_subset.json");
  const raw = fs.readFileSync(filePath, "utf-8");
  const records = JSON.parse(raw) as Array<{ query: string; question_type: string }>;

  const byType = new Map<string, string[]>();
  for (const r of records) {
    if (r.question_type === "null_query" || r.query.length > 220) continue;
    const bucket = byType.get(r.question_type) ?? [];
    bucket.push(r.query);
    byType.set(r.question_type, bucket);
  }

  const examples: string[] = [];
  for (const type of ["inference_query", "comparison_query", "temporal_query"]) {
    const bucket = byType.get(type);
    if (!bucket || bucket.length === 0) continue;
    // Shortest-first within the length-filtered bucket — a scannable example
    // chip, not necessarily the first one encountered in the source file.
    const shortest = [...bucket].sort((a, b) => a.length - b.length)[0];
    examples.push(shortest);
  }
  return examples;
}
