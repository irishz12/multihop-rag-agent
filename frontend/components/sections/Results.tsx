"use client";

import { BarChart, type BarGroup } from "@/components/ui/BarChart";
import { ChartPair } from "@/components/ui/ChartPair";
import { Section } from "@/components/ui/Section";
import {
  PIPELINE_COLOR,
  PIPELINE_LABEL,
  formatPercent,
  formatScore,
  niceTicks,
  titleCaseHopBucket,
  titleCaseQuestionType,
} from "@/lib/format";
import type { BaselineCostLatency, BreakdownRow, HoldoutReport, PipelineKey, SampleReport } from "@/lib/types";

interface ResultsProps {
  sample: SampleReport;
  holdout: HoldoutReport;
  devBaselineCostLatency: Record<"dense" | "hybrid" | "hybrid_reranker", BaselineCostLatency>;
}

const QUESTION_TYPE_ORDER = ["inference_query", "comparison_query", "temporal_query", "null_query"];
const HOP_ORDER = ["hop2", "hop3", "hop4", "null"];

function singleSeriesGroups(entries: Array<[PipelineKey, number]>): BarGroup[] {
  return entries.map(([key, value]) => ({
    label: PIPELINE_LABEL[key],
    bars: [{ key, value, color: PIPELINE_COLOR[key], label: PIPELINE_LABEL[key] }],
  }));
}

function breakdownGroups(
  breakdown: Record<string, BreakdownRow>,
  order: string[],
  labelFn: (key: string) => string,
): BarGroup[] {
  return order
    .filter((key) => breakdown[key])
    .map((key) => {
      const row = breakdown[key];
      return {
        label: `${labelFn(key)} (n=${row.n})`,
        bars: [
          {
            key: "agentic_multi_hop",
            value: row.agentic_multi_hop_mean_quality,
            color: PIPELINE_COLOR.agentic_multi_hop,
            label: PIPELINE_LABEL.agentic_multi_hop,
          },
          {
            key: "adaptive_rag",
            value: row.adaptive_rag_mean_quality,
            color: PIPELINE_COLOR.adaptive_rag,
            label: PIPELINE_LABEL.adaptive_rag,
          },
        ],
      };
    });
}

const AGENTIC_ADAPTIVE_LEGEND = [
  { key: "agentic_multi_hop", label: PIPELINE_LABEL.agentic_multi_hop, color: PIPELINE_COLOR.agentic_multi_hop },
  { key: "adaptive_rag", label: PIPELINE_LABEL.adaptive_rag, color: PIPELINE_COLOR.adaptive_rag },
];

export function Results({ sample, holdout, devBaselineCostLatency }: ResultsProps) {
  // --- answer quality -----------------------------------------------------------------
  const devQualityGroups = singleSeriesGroups([
    ["hybrid_reranker", sample.combined_quality_mean.hybrid_reranker ?? 0],
    ["agentic_multi_hop", sample.combined_quality_mean.agentic_multi_hop ?? 0],
    ["adaptive_rag", sample.combined_quality_mean.adaptive_rag ?? 0],
  ]);
  const holdoutQualityGroups = singleSeriesGroups([
    ["agentic_multi_hop", holdout.combined_quality_mean.agentic_multi_hop ?? 0],
    ["adaptive_rag", holdout.combined_quality_mean.adaptive_rag ?? 0],
  ]);

  // --- evidence coverage ---------------------------------------------------------------
  const devCoverageGroups = singleSeriesGroups(
    (["dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag"] as PipelineKey[]).map(
      (key) => [key, sample.evidence_coverage_mean[key] ?? 0] as [PipelineKey, number],
    ),
  );
  const holdoutCoverageGroups = singleSeriesGroups([
    ["agentic_multi_hop", holdout.evidence_coverage_mean.agentic_multi_hop ?? 0],
    ["adaptive_rag", holdout.evidence_coverage_mean.adaptive_rag ?? 0],
  ]);

  // --- cost / latency per query ----------------------------------------------------------
  const devCostGroups = singleSeriesGroups([
    ["dense", devBaselineCostLatency.dense.cost],
    ["hybrid", devBaselineCostLatency.hybrid.cost],
    ["hybrid_reranker", devBaselineCostLatency.hybrid_reranker.cost],
    ["agentic_multi_hop", sample.cost_latency.agentic_multi_hop_mean_cost_usd],
    ["adaptive_rag", sample.cost_latency.adaptive_rag_mean_cost_usd],
  ]);
  const holdoutCostGroups = singleSeriesGroups([
    ["agentic_multi_hop", holdout.cost_latency.agentic_multi_hop_mean_cost_usd],
    ["adaptive_rag", holdout.cost_latency.adaptive_rag_mean_cost_usd],
  ]);
  const devLatencyGroups = singleSeriesGroups([
    ["dense", devBaselineCostLatency.dense.latency],
    ["hybrid", devBaselineCostLatency.hybrid.latency],
    ["hybrid_reranker", devBaselineCostLatency.hybrid_reranker.latency],
    ["agentic_multi_hop", sample.cost_latency.agentic_multi_hop_mean_latency_ms],
    ["adaptive_rag", sample.cost_latency.adaptive_rag_mean_latency_ms],
  ]);
  const holdoutLatencyGroups = singleSeriesGroups([
    ["agentic_multi_hop", holdout.cost_latency.agentic_multi_hop_mean_latency_ms],
    ["adaptive_rag", holdout.cost_latency.adaptive_rag_mean_latency_ms],
  ]);

  // --- breakdowns --------------------------------------------------------------------------
  const devTypeGroups = breakdownGroups(sample.breakdown_by_question_type, QUESTION_TYPE_ORDER, titleCaseQuestionType);
  const holdoutTypeGroups = breakdownGroups(
    holdout.breakdown_by_question_type,
    QUESTION_TYPE_ORDER,
    titleCaseQuestionType,
  );
  const devHopGroups = breakdownGroups(sample.breakdown_by_hop_count, HOP_ORDER, titleCaseHopBucket);
  const holdoutHopGroups = breakdownGroups(holdout.breakdown_by_hop_count, HOP_ORDER, titleCaseHopBucket);

  // --- development vs holdout summary --------------------------------------------------------
  const DEV_COLOR = "#8FB7DE";
  const HOLDOUT_COLOR = "#0F5AA8";
  const devVsHoldoutGroups: BarGroup[] = [
    {
      label: "Quality retention",
      bars: [
        { key: "dev", value: holdout.development_vs_holdout.development.adaptive_quality_retention_pct, color: DEV_COLOR, label: "Development" },
        { key: "holdout", value: holdout.development_vs_holdout.holdout.adaptive_quality_retention_pct, color: HOLDOUT_COLOR, label: "Final holdout" },
      ],
    },
    {
      label: "Cost reduction",
      bars: [
        { key: "dev", value: holdout.development_vs_holdout.development.cost_reduction_pct, color: DEV_COLOR, label: "Development" },
        { key: "holdout", value: holdout.development_vs_holdout.holdout.cost_reduction_pct, color: HOLDOUT_COLOR, label: "Final holdout" },
      ],
    },
    {
      label: "Latency reduction",
      bars: [
        { key: "dev", value: holdout.development_vs_holdout.development.latency_reduction_pct, color: DEV_COLOR, label: "Development" },
        { key: "holdout", value: holdout.development_vs_holdout.holdout.latency_reduction_pct, color: HOLDOUT_COLOR, label: "Final holdout" },
      ],
    },
  ];

  const devCostTicks = niceTicks(Math.max(...devCostGroups.map((g) => g.bars[0].value)));
  const holdoutCostTicks = niceTicks(holdout.cost_latency.agentic_multi_hop_mean_cost_usd);
  const devLatencyTicks = niceTicks(Math.max(...devLatencyGroups.map((g) => g.bars[0].value)));
  const holdoutLatencyTicks = niceTicks(holdout.cost_latency.agentic_multi_hop_mean_latency_ms);

  return (
    <Section id="results" number="06" title="Results" width="wide">
      <div className="space-y-16">
        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Answer quality</h3>
          <p className="mb-6 text-sm text-ink-muted">
            Judge score for open-ended questions blended with abstention correctness for null questions.
          </p>
          <ChartPair
            leftTitle="Development sample (n = 50)"
            rightTitle="Final holdout (n = 50, one-time)"
            left={
              <BarChart
                groups={devQualityGroups}
                yMax={1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={(v) => formatScore(v, 2)}
                ariaLabel="Development-sample answer quality: Hybrid plus Reranker 0.54, Agentic Multi-Hop RAG 0.64, Adaptive RAG 0.60"
              />
            }
            right={
              <BarChart
                groups={holdoutQualityGroups}
                yMax={1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={(v) => formatScore(v, 2)}
                ariaLabel="Final holdout answer quality: Agentic Multi-Hop RAG 0.70, Adaptive RAG 0.56"
              />
            }
          />
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Evidence coverage</h3>
          <p className="mb-6 text-sm text-ink-muted">
            Fraction of a question&rsquo;s gold source documents that reached the final-answer prompt.
          </p>
          <ChartPair
            leftTitle="Development sample (n = 50)"
            rightTitle="Final holdout (n = 50, one-time)"
            left={
              <BarChart
                groups={devCoverageGroups}
                yMax={1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={formatPercent}
                ariaLabel="Development-sample evidence coverage across five pipelines"
              />
            }
            right={
              <BarChart
                groups={holdoutCoverageGroups}
                yMax={1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={formatPercent}
                ariaLabel="Final holdout evidence coverage: Agentic Multi-Hop RAG 75.9%, Adaptive RAG 68.0%"
              />
            }
          />
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Cost per query</h3>
          <p className="mb-6 text-sm text-ink-muted">Mean Mantle cost per question, in USD.</p>
          <ChartPair
            leftTitle="Development sample (n = 50)"
            rightTitle="Final holdout (n = 50, one-time)"
            left={
              <BarChart
                groups={devCostGroups}
                yMax={devCostTicks[devCostTicks.length - 1]}
                yTicks={devCostTicks}
                formatValue={(v) => `$${v.toFixed(5)}`}
                formatTick={(v) => `$${v.toFixed(4)}`}
                ariaLabel="Development-sample mean cost per query across five pipelines"
              />
            }
            right={
              <BarChart
                groups={holdoutCostGroups}
                yMax={holdoutCostTicks[holdoutCostTicks.length - 1]}
                yTicks={holdoutCostTicks}
                formatValue={(v) => `$${v.toFixed(5)}`}
                formatTick={(v) => `$${v.toFixed(4)}`}
                ariaLabel="Final holdout mean cost per query: Agentic Multi-Hop RAG and Adaptive RAG"
              />
            }
          />
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Latency per query</h3>
          <p className="mb-6 text-sm text-ink-muted">Mean end-to-end latency per question, in milliseconds.</p>
          <ChartPair
            leftTitle="Development sample (n = 50)"
            rightTitle="Final holdout (n = 50, one-time)"
            left={
              <BarChart
                groups={devLatencyGroups}
                yMax={devLatencyTicks[devLatencyTicks.length - 1]}
                yTicks={devLatencyTicks}
                formatValue={(v) => `${Math.round(v)}`}
                formatTick={(v) => v.toLocaleString("en-US")}
                ariaLabel="Development-sample mean latency per query across five pipelines, in milliseconds"
              />
            }
            right={
              <BarChart
                groups={holdoutLatencyGroups}
                yMax={holdoutLatencyTicks[holdoutLatencyTicks.length - 1]}
                yTicks={holdoutLatencyTicks}
                formatValue={(v) => `${Math.round(v)}`}
                formatTick={(v) => v.toLocaleString("en-US")}
                ariaLabel="Final holdout mean latency per query: Agentic Multi-Hop RAG and Adaptive RAG, in milliseconds"
              />
            }
          />
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Quality by query type</h3>
          <p className="mb-6 text-sm text-ink-muted">
            Combined quality, Agentic Multi-Hop RAG vs. Adaptive RAG, by MultiHop-RAG question type.
          </p>
          <ChartPair
            leftTitle="Development sample"
            rightTitle="Final holdout"
            left={
              <BarChart
                groups={devTypeGroups}
                yMax={1.1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={(v) => formatScore(v, 2)}
                legend={AGENTIC_ADAPTIVE_LEGEND}
                ariaLabel="Development-sample quality by query type, Agentic Multi-Hop RAG versus Adaptive RAG"
              />
            }
            right={
              <BarChart
                groups={holdoutTypeGroups}
                yMax={1.1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={(v) => formatScore(v, 2)}
                legend={AGENTIC_ADAPTIVE_LEGEND}
                ariaLabel="Final holdout quality by query type, Agentic Multi-Hop RAG versus Adaptive RAG"
              />
            }
          />
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Quality by hop-count difficulty</h3>
          <p className="mb-6 text-sm text-ink-muted">
            Combined quality by the number of gold documents a question requires.
          </p>
          <ChartPair
            leftTitle="Development sample"
            rightTitle="Final holdout"
            left={
              <BarChart
                groups={devHopGroups}
                yMax={1.1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={(v) => formatScore(v, 2)}
                legend={AGENTIC_ADAPTIVE_LEGEND}
                ariaLabel="Development-sample quality by hop count, Agentic Multi-Hop RAG versus Adaptive RAG"
              />
            }
            right={
              <BarChart
                groups={holdoutHopGroups}
                yMax={1.1}
                yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                formatValue={(v) => formatScore(v, 2)}
                legend={AGENTIC_ADAPTIVE_LEGEND}
                ariaLabel="Final holdout quality by hop count, Agentic Multi-Hop RAG versus Adaptive RAG"
              />
            }
          />
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Development vs. holdout</h3>
          <p className="mb-6 text-sm text-ink-muted">
            Does Adaptive RAG&rsquo;s cost/latency trade-off hold up on unseen data?
          </p>
          <BarChart
            groups={devVsHoldoutGroups}
            yMax={1}
            yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
            formatValue={formatPercent}
            legend={[
              { key: "dev", label: "Development", color: DEV_COLOR },
              { key: "holdout", label: "Final holdout", color: HOLDOUT_COLOR },
            ]}
            ariaLabel="Development versus final holdout: Adaptive RAG quality retention, cost reduction, and latency reduction"
          />
        </div>
      </div>
    </Section>
  );
}
