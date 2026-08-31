import { Section } from "@/components/ui/Section";
import { formatPercent } from "@/lib/format";
import type { HoldoutReport } from "@/lib/types";

interface KeyFindingsProps {
  holdout: HoldoutReport;
}

export function KeyFindings({ holdout }: KeyFindingsProps) {
  const fallbackRate = holdout.judge_fallback_total / (holdout.n_non_null * 2);

  const findings = [
    "Agentic Multi-Hop RAG outperforms every single-pass baseline and the Adaptive variant on both answer quality and evidence coverage, on both the development sample and final holdout.",
    "Adaptive RAG recovers most of that quality — 80% on final holdout — at meaningfully lower cost and latency, by routing roughly a quarter of questions away from the full agentic loop.",
    "The quality gap and the cost/latency savings both grew from development to holdout, moving together rather than independently — a real, not noisy, generalization finding.",
  ];

  const limitations = [
    "50 questions per split is enough to see a clear, consistent signal, not enough for tight confidence intervals on any single percentage.",
    "The router's MEDIUM class is rare and hard to hit — a direct consequence of a conservative, safety-first threshold objective, not an oversight.",
    `Judge fallback rate on final holdout was ${formatPercent(fallbackRate)} — sensitivity-checked, and it does not change the headline finding, but it is a real source of noise, not zero.`,
    "Temporal and comparison questions remain unresolved weaknesses for the main agentic system, not just the router built on top of it.",
    "Normalized exact match and token F1 are close to uninformative for this dataset's explanatory gold answers — the judge score and evidence coverage carry the real signal, and both depend on judge-model behavior validated on a small sample rather than cross-checked against a second judge.",
    "The final holdout evaluation is one-time by design — these numbers cannot be improved by further tuning without invalidating the entire held-out measurement.",
  ];

  return (
    <Section id="findings" number="11" title="Key Findings & Limitations">
      <div className="space-y-10">
        <div>
          <h3 className="text-sm font-medium tracking-wide text-ink-faint uppercase">Findings</h3>
          <ul className="mt-4 space-y-4">
            {findings.map((text) => (
              <li key={text} className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
                {text}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="text-sm font-medium tracking-wide text-ink-faint uppercase">Limitations</h3>
          <ul className="mt-4 space-y-4">
            {limitations.map((text) => (
              <li key={text} className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
                {text}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Section>
  );
}
