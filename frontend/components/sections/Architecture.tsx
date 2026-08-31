import { FlowDiagram } from "@/components/ui/FlowDiagram";
import { Section } from "@/components/ui/Section";

const STEPS = [
  { label: "Question", detail: "text only — no gold answer or evidence" },
  { label: "Dense + BM25", detail: "two independent retrieval passes" },
  { label: "Reciprocal rank fusion", detail: "deterministic RRF, k = 60" },
  { label: "Cross-encoder reranker", detail: "BAAI/bge-reranker-base" },
  { label: "Evidence sufficiency", detail: "one structured GLM judgment" },
  { label: "Follow-up retrieval", detail: "a new focused query" },
  { label: "Final answer", detail: "Qwen, from all merged evidence" },
];

export function Architecture() {
  return (
    <Section id="architecture" number="02" title="Architecture" width="wide">
      <p className="mb-8 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
        Every hop runs the same frozen retrieval pipeline. What changes hop to
        hop is only the query: the original question first, then a controller-
        written follow-up whenever the merged evidence isn&rsquo;t judged
        sufficient yet.
      </p>
      <FlowDiagram
        steps={STEPS}
        loopBack={{
          fromIndex: 5,
          toIndex: 1,
          note: "capped at 3 hops total, then answers from whatever evidence it has",
        }}
        ariaLabel="Agentic Multi-Hop RAG pipeline: question, dense and BM25 retrieval, reciprocal rank fusion, cross-encoder reranking, an evidence-sufficiency judgment, an optional follow-up retrieval hop repeating from dense and BM25 retrieval up to three hops total, then a final answer."
      />
      <div className="mt-10 grid grid-cols-1 gap-8 border-t border-rule pt-8 text-sm text-ink-muted sm:grid-cols-3">
        <div>
          <p className="font-medium text-ink">Hard limits, not just prompts</p>
          <p className="mt-1.5">
            Max 3 retrieval calls, a context token budget, and a wall-clock
            timeout are enforced in code — the loop cannot exceed them
            regardless of what the controller returns.
          </p>
        </div>
        <div>
          <p className="font-medium text-ink">One controller call per hop</p>
          <p className="mt-1.5">
            A single structured call returns whether the evidence is
            sufficient and, if not, the next query — never a separate
            sufficiency check and sub-query call.
          </p>
        </div>
        <div>
          <p className="font-medium text-ink">Same answer model, every path</p>
          <p className="mt-1.5">
            Every pipeline on this page — baseline, agentic, or adaptive —
            answers with the identical model, prompt, and context assembly.
            Only the evidence reaching that call differs.
          </p>
        </div>
      </div>
    </Section>
  );
}
