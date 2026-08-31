// Types describe the CANONICAL shape this app works with — every field
// name here uses this project's current pipeline names. The read-only JSON
// artifacts under ../results/*.json are immutable and still use two legacy
// pipeline names (`always_agentic`, `adaptive`); lib/data.ts is the one
// place that reads their raw shape and rekeys it to what's declared here
// (see lib/legacyPipelineNames.ts). Every other field name matches the
// source files verbatim.

export type PipelineKey = "dense" | "hybrid" | "hybrid_reranker" | "agentic_multi_hop" | "adaptive_rag";

export interface DeterministicMetrics {
  normalized_exact_match: number;
  token_f1: number;
  null_query_abstention_accuracy: number;
  n_non_null?: number;
  n_null?: number;
}

export interface SampleJudgeScore {
  mean_judge_score: number;
  n_correct: number;
  n_partially_correct: number;
  n_incorrect: number;
  n_judge_fallbacks: number;
}

export interface HoldoutJudgeScore {
  mean_judge_score_including_fallbacks: number;
  mean_judge_score_excluding_fallbacks: number | null;
  n_correct: number;
  n_partially_correct: number;
  n_incorrect: number;
  n_judge_fallbacks: number;
  fallback_qa_ids: string[];
}

export interface CostLatency {
  agentic_multi_hop_mean_cost_usd: number;
  adaptive_rag_mean_cost_usd: number;
  agentic_multi_hop_mean_latency_ms: number;
  adaptive_rag_mean_latency_ms: number;
  cost_reduction_pct: number;
  latency_reduction_pct: number;
}

export interface BreakdownRow {
  n: number;
  hybrid_reranker_mean_quality?: number;
  agentic_multi_hop_mean_quality: number;
  adaptive_rag_mean_quality: number;
}

export interface UnderRoutedFailure {
  qa_id: string;
  question_type: string;
  hop_count: number;
  route: string;
  adaptive_rag_judge_score: number;
  agentic_multi_hop_judge_score: number;
  adaptive_rag_evidence_coverage: number | null;
  agentic_multi_hop_evidence_coverage: number | null;
}

export interface SampleReport {
  generated_at: string;
  sample_seed: number;
  sample_size: number;
  n_non_null: number;
  n_null: number;
  deterministic_metrics: Partial<Record<PipelineKey, DeterministicMetrics>>;
  judge_scores: Partial<Record<PipelineKey, SampleJudgeScore>>;
  combined_quality_mean: Partial<Record<PipelineKey, number>>;
  adaptive_quality_retention_pct_vs_agentic_multi_hop: number;
  evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
  cost_latency: CostLatency;
  breakdown_by_question_type: Record<string, BreakdownRow>;
  breakdown_by_hop_count: Record<string, BreakdownRow>;
  under_routed_failures: UnderRoutedFailure[];
  judge_call_stats: {
    n_judge_calls: number;
    total_input_tokens: number;
    total_output_tokens: number;
    mean_latency_ms: number;
    n_fallbacks: number;
    total_cost_usd: number | null;
  };
}

export interface HoldoutReport {
  generated_at: string;
  sample_seed: number;
  sample_size: number;
  n_non_null: number;
  n_null: number;
  integrity_check: string;
  pre_access_manifest_generated_at: string;
  deterministic_metrics: Partial<Record<PipelineKey, DeterministicMetrics>>;
  judge_scores: Partial<Record<PipelineKey, HoldoutJudgeScore>>;
  combined_quality_mean: Partial<Record<PipelineKey, number>>;
  adaptive_quality_retention_pct_vs_agentic_multi_hop: number;
  evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
  cost_latency: CostLatency;
  breakdown_by_question_type: Record<string, BreakdownRow>;
  breakdown_by_hop_count: Record<string, BreakdownRow>;
  under_routed_failures: UnderRoutedFailure[];
  development_vs_holdout: {
    development: {
      combined_quality_mean: Partial<Record<PipelineKey, number>>;
      adaptive_quality_retention_pct: number;
      cost_reduction_pct: number;
      latency_reduction_pct: number;
      evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
    };
    holdout: {
      combined_quality_mean: Partial<Record<PipelineKey, number>>;
      adaptive_quality_retention_pct: number;
      cost_reduction_pct: number;
      latency_reduction_pct: number;
      evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
    };
  };
  judge_fallback_total: number;
  total_evaluation_cost: {
    total_pipeline_cost_usd: number;
    total_judge_cost_usd: number | null;
    n_judge_calls: number;
    judge_total_input_tokens: number;
    judge_total_output_tokens: number;
  };
}

export interface RetrievalMethodAggregate {
  "recall@4": number;
  "recall@5": number;
  "recall@10": number;
  "hit@4": number;
  "hit@10": number;
  "complete_evidence@4": number;
  "complete_evidence@10": number;
  "mrr@10": number;
  "ndcg@10": number;
}

export interface RetrievalEval {
  counts: {
    total_development_questions: number;
    non_null_evaluated: number;
    null_query_excluded: number;
  };
  aggregate: Record<"dense" | "bm25" | "hybrid" | "hybrid_reranker", RetrievalMethodAggregate>;
}

export interface RouterModelSide {
  feature_names: string[];
  threshold: number;
  trained_on_n_questions: number;
}

export interface RouterModel {
  cv_seed: number;
  n_splits: number;
  stage1: RouterModelSide;
  stage2: RouterModelSide;
}

export interface HoldoutConsumed {
  status: string;
  sample_seed: number;
  sample_size: number;
  pre_access_manifest_generated_at: string;
  holdout_report_generated_at: string;
  integrity_check: string;
}

export interface BaselineCostLatency {
  cost: number;
  latency: number;
}
