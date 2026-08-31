import { MetricStat } from "@/components/ui/MetricStat";
import { formatPercent, formatScore } from "@/lib/format";

interface HeroProps {
  agenticQuality: number;
  adaptiveQuality: number;
  retentionPct: number;
}

export function Hero({ agenticQuality, adaptiveQuality, retentionPct }: HeroProps) {
  return (
    <header className="pt-20 pb-16 sm:pt-28 sm:pb-20">
      <div className="mx-auto max-w-[42rem] px-6">
        <p className="font-data text-xs tracking-widest text-ink-faint uppercase">
          Project case study
        </p>
        <h1 className="mt-4 font-display text-4xl leading-[1.1] font-semibold tracking-tight text-ink sm:text-5xl">
          Agentic Multi-Hop RAG
        </h1>
        <p className="mt-6 max-w-[38rem] text-lg leading-relaxed text-ink-muted">
          A single retrieval pass fails when a question&rsquo;s evidence is spread
          across several documents. This system retrieves iteratively instead —
          judging after each hop whether it has enough evidence, and asking a
          focused follow-up question when it doesn&rsquo;t, for up to three hops.
        </p>
        <p className="mt-4 max-w-[38rem] text-lg leading-relaxed text-ink-muted">
          A learned router variant, Adaptive RAG, tests whether that same quality
          can be approximated more cheaply — measured honestly, once, on a final
          holdout set it never saw during development.
        </p>

        <div className="mt-12 grid grid-cols-3 gap-6 border-t border-rule pt-8">
          <MetricStat
            value={formatScore(agenticQuality)}
            label="Agentic answer quality"
            system="agentic"
            detail="final holdout"
          />
          <MetricStat
            value={formatScore(adaptiveQuality)}
            label="Adaptive answer quality"
            system="adaptive"
            detail="final holdout"
          />
          <MetricStat
            value={formatPercent(retentionPct)}
            label="Adaptive quality retention"
            detail="vs. Agentic"
          />
        </div>
      </div>
    </header>
  );
}
