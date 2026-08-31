import { Section } from "@/components/ui/Section";
import { formatPercent, formatScore } from "@/lib/format";
import type { HoldoutReport, RetrievalEval } from "@/lib/types";

interface ExperimentJourneyProps {
  retrievalEval: RetrievalEval;
  holdout: HoldoutReport;
}

export function ExperimentJourney({ retrievalEval, holdout }: ExperimentJourneyProps) {
  const steps = [
    {
      title: "Dense retrieval",
      body: "A single embedding-similarity pass over the corpus establishes the floor.",
      metric: `${formatPercent(retrievalEval.aggregate.dense["recall@10"])} Recall@10`,
    },
    {
      title: "Hybrid retrieval",
      body: "Dense and BM25 fused by deterministic reciprocal rank fusion recover evidence neither method alone surfaces.",
      metric: `${formatPercent(retrievalEval.aggregate.hybrid["recall@10"])} Recall@10`,
    },
    {
      title: "Cross-encoder reranking",
      body: "Reordering the fused candidates by a cross-encoder trades a little Recall@10 for a markedly better top-of-ranking precision.",
      metric: `${formatScore(retrievalEval.aggregate.hybrid_reranker["mrr@10"])} MRR@10`,
    },
    {
      title: "Agentic Multi-Hop RAG",
      body: "A bounded, evidence-checking loop replaces one retrieval pass with up to three — the main system, measured once on final holdout.",
      metric: `${formatScore(holdout.combined_quality_mean.agentic_multi_hop ?? 0)} answer quality`,
    },
    {
      title: "Adaptive RAG",
      body: "A learned router sends most questions through a cheaper path and reserves the full loop for the ones that need it.",
      metric: `${formatPercent(holdout.cost_latency.cost_reduction_pct)} cost reduction`,
    },
  ];

  return (
    <Section id="journey" number="07" title="Experiment Journey">
      <ol className="space-y-8">
        {steps.map((step, i) => (
          <li key={step.title} className="flex gap-5">
            <span className="font-data text-sm tabular-nums text-ink-faint">
              {String(i + 1).padStart(2, "0")}
            </span>
            <div className="flex-1 border-b border-rule pb-8 last:border-b-0 last:pb-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <h3 className="text-base font-medium text-ink">{step.title}</h3>
                <span className="tnum font-data text-sm text-ink-muted">{step.metric}</span>
              </div>
              <p className="mt-2 max-w-[38rem] text-sm leading-relaxed text-ink-muted">{step.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </Section>
  );
}
