import { Section } from "@/components/ui/Section";
import type { HoldoutConsumed } from "@/lib/types";

interface MethodologyProps {
  holdoutConsumed: HoldoutConsumed;
  passingTestCount: number;
}

const ITEMS = (holdoutConsumed: HoldoutConsumed, passingTestCount: number) => [
  {
    title: "MultiHop-RAG",
    body: "609 news documents, 2,556 questions labeled inference / comparison / temporal / null, each citing 0–4 supporting source documents — a benchmark built specifically so single-pass retrieval cannot reliably answer every question.",
  },
  {
    title: "Frozen configurations",
    body: "Every model id, retrieval config, router threshold, agent config, and prompt template used for the final holdout evaluation was fixed before that evaluation began — see the pre-access manifest below.",
  },
  {
    title: "An untouched final holdout, until one measurement",
    body: `A configuration manifest was SHA-1 hashed BEFORE the holdout file was read for the first time. The aggregation step re-hashes the same files and would raise on any drift — mechanically checked, not just claimed. Integrity check: "${holdoutConsumed.integrity_check}".`,
  },
  {
    title: "A separate judge model",
    body: "openai.gpt-oss-120b grades every answer — a third model, distinct from both the answer-generation model (Qwen) and the agent controller (GLM), seeing only the question, reference answer, and candidate answer. It never learns which pipeline, route, or model produced what it's grading.",
  },
  {
    title: "Checkpointed, paid evaluation",
    body: "Every live model call is written to disk immediately after it completes. An interrupted run resumes without repeating a completed, already-paid call — verified in practice, not just in theory, across this project's development.",
  },
  {
    title: `${passingTestCount} passing tests`,
    body: "Unit tests run entirely offline against fake models and fixtures; structural guard tests grep the actual source of every live script, proving in code — not just in documentation — that gold answers can never reach a routing or generation decision, and that no script can reach the final holdout file outside its one deliberate, clearly named exception.",
  },
  {
    title: "Consumed",
    body: `final_holdout.json is marked "${holdoutConsumed.status}" (seed ${holdoutConsumed.sample_seed}, ${holdoutConsumed.sample_size} questions) and is not evaluated against again.`,
  },
];

export function Methodology({ holdoutConsumed, passingTestCount }: MethodologyProps) {
  const items = ITEMS(holdoutConsumed, passingTestCount);
  return (
    <Section id="methodology" number="09" title="Evaluation Methodology">
      <dl className="space-y-7">
        {items.map((item) => (
          <div key={item.title}>
            <dt className="text-base font-medium text-ink">{item.title}</dt>
            <dd className="mt-1.5 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">{item.body}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}
