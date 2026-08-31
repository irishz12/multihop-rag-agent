import { Section } from "@/components/ui/Section";

export function BusinessProblem() {
  return (
    <Section id="problem" number="01" title="Business Problem">
      <div className="space-y-5 text-[1.0625rem] leading-relaxed text-ink-muted">
        <p>
          Retrieval-augmented generation typically retrieves once and answers
          once. That works when a single query surfaces every fact a question
          needs. It breaks down for questions whose answer depends on
          synthesizing facts that live in different documents — comparing two
          sources, reconstructing a timeline, or chaining several inferences
          together. A single retrieval pass either misses part of the
          evidence, or surfaces it too far down the ranking to reach the
          model&rsquo;s context window.
        </p>
        <p>
          The failure is quiet. The system doesn&rsquo;t know it is missing
          evidence, so it answers anyway — confidently, and sometimes wrongly.
          There is no built-in mechanism to notice the gap and go look again.
        </p>
        <p>
          This project measures that gap directly, on{" "}
          <a
            href="https://github.com/yixuantt/MultiHop-RAG"
            className="text-ink underline decoration-rule underline-offset-4 hover:decoration-ink"
          >
            MultiHop-RAG
          </a>{" "}
          — a benchmark built specifically so single-pass retrieval cannot
          reliably answer every question — and builds a system that retrieves
          again when the first pass isn&rsquo;t enough, plus a cheaper variant
          that learns when that extra work is actually worth doing.
        </p>
      </div>
    </Section>
  );
}
