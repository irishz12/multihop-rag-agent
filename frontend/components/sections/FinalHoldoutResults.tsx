import { MetricStat } from "@/components/ui/MetricStat";
import { Section } from "@/components/ui/Section";
import { formatPercent, formatScore } from "@/lib/format";

interface FinalHoldoutResultsProps {
  agenticQuality: number;
  adaptiveQuality: number;
  agenticCoverage: number;
  adaptiveCoverage: number;
  costReductionPct: number;
  latencyReductionPct: number;
  sampleSize: number;
}

export function FinalHoldoutResults({
  agenticQuality,
  adaptiveQuality,
  agenticCoverage,
  adaptiveCoverage,
  costReductionPct,
  latencyReductionPct,
  sampleSize,
}: FinalHoldoutResultsProps) {
  return (
    <Section id="holdout" number="05" title="Final Holdout Results" width="wide">
      <p className="mb-10 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
        Measured once, on {sampleSize} questions from a split neither the
        router nor any prompt or threshold was ever tuned against. This is
        the number that matters more than anything on the development split.
      </p>

      <div className="grid grid-cols-2 gap-x-8 gap-y-10 sm:grid-cols-3">
        <MetricStat value={formatScore(agenticQuality)} label="Agentic answer quality" system="agentic" />
        <MetricStat value={formatScore(adaptiveQuality)} label="Adaptive answer quality" system="adaptive" />
        <MetricStat value={formatPercent(agenticCoverage)} label="Agentic evidence coverage" system="agentic" />
        <MetricStat value={formatPercent(adaptiveCoverage)} label="Adaptive evidence coverage" system="adaptive" />
        <MetricStat value={formatPercent(costReductionPct)} label="Adaptive cost reduction" system="adaptive" />
        <MetricStat value={formatPercent(latencyReductionPct)} label="Adaptive latency reduction" system="adaptive" />
      </div>

      <p className="mt-10 max-w-[42rem] border-t border-rule pt-8 text-[1.0625rem] leading-relaxed text-ink-muted">
        Agentic Multi-Hop RAG achieved higher answer quality and evidence
        coverage. Adaptive RAG reduced cost and latency at a measurable
        quality trade-off. Neither result is smoothed over — both are held
        side by side in the sections below.
      </p>
    </Section>
  );
}
