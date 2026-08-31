import { Section } from "@/components/ui/Section";
import { formatPercent, formatScore } from "@/lib/format";
import type { HoldoutReport, SampleReport } from "@/lib/types";

interface FailureAnalysisProps {
  sample: SampleReport;
  holdout: HoldoutReport;
}

export function FailureAnalysis({ sample, holdout }: FailureAnalysisProps) {
  const holdoutComparison = holdout.breakdown_by_question_type.comparison_query;
  const holdoutTemporal = holdout.breakdown_by_question_type.temporal_query;
  const devTemporal = sample.breakdown_by_question_type.temporal_query;

  const devUnderRouted = sample.under_routed_failures.length;
  const holdoutUnderRouted = holdout.under_routed_failures.length;

  const devRetention = holdout.development_vs_holdout.development.adaptive_quality_retention_pct;
  const holdoutRetention = holdout.development_vs_holdout.holdout.adaptive_quality_retention_pct;

  return (
    <Section id="failure-analysis" number="08" title="Failure Analysis">
      <div className="space-y-10">
        <div>
          <h3 className="text-base font-medium text-ink">Temporal and comparison questions are the clearest weakness</h3>
          <p className="mt-2 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
            On final holdout, temporal questions score{" "}
            <span className="tnum font-data text-ink">{formatScore(holdoutTemporal.agentic_multi_hop_mean_quality, 2)}</span>{" "}
            for Agentic and{" "}
            <span className="tnum font-data text-ink">{formatScore(holdoutTemporal.adaptive_rag_mean_quality, 2)}</span>{" "}
            for Adaptive — the lowest of any category, on both systems, on both splits (development temporal quality
            for Adaptive was{" "}
            <span className="tnum font-data text-ink">{formatScore(devTemporal.adaptive_rag_mean_quality, 2)}</span>).
            Comparison questions score{" "}
            <span className="tnum font-data text-ink">{formatScore(holdoutComparison.agentic_multi_hop_mean_quality, 2)}</span>{" "}
            (Agentic) and{" "}
            <span className="tnum font-data text-ink">{formatScore(holdoutComparison.adaptive_rag_mean_quality, 2)}</span>{" "}
            (Adaptive) — the second-weakest category, with Adaptive trailing Agentic by a visible margin. Neither
            weakness is new to Adaptive RAG; the router does not compensate for either.
          </p>
        </div>

        <div>
          <h3 className="text-base font-medium text-ink">Adaptive under-routing</h3>
          <p className="mt-2 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
            {devUnderRouted} of {sample.n_non_null} non-null development questions, and {holdoutUnderRouted} of{" "}
            {holdout.n_non_null} on final holdout, were routed to a cheaper path and scored — or covered evidence —
            strictly worse than Agentic would have. Both sets concentrate in comparison and temporal questions: the
            router is paying its cheap-route bet and losing it on exactly the categories already flagged above, not
            a separate, unrelated failure mode.
          </p>
        </div>

        <div>
          <h3 className="text-base font-medium text-ink">The development-to-holdout generalization gap</h3>
          <p className="mt-2 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
            Adaptive&rsquo;s quality retention was{" "}
            <span className="tnum font-data text-ink">{formatPercent(devRetention)}</span> on the development sample
            and <span className="tnum font-data text-ink">{formatPercent(holdoutRetention)}</span> on final holdout —
            a real drop, not sampling noise smoothed away. Cost and latency reductions moved the opposite direction,
            growing larger on holdout. Both numbers moving together — bigger savings, bigger quality gap — is
            consistent with the router escalating slightly less on this split, not with two unrelated metrics
            drifting independently.
          </p>
        </div>
      </div>
    </Section>
  );
}
