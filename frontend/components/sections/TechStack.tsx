import { Section } from "@/components/ui/Section";

const GROUPS = [
  {
    title: "Retrieval & indexing",
    items: ["Qdrant", "sentence-transformers (BAAI/bge-base-en-v1.5)", "FastEmbed sparse BM25", "BAAI/bge-reranker-base"],
  },
  {
    title: "Routing & evaluation",
    items: ["scikit-learn (logistic regression router)", "pytest", "NumPy"],
  },
  {
    title: "Language models",
    items: ["qwen.qwen3-next-80b-a3b-instruct — generation", "zai.glm-4.7-flash — agent controller", "openai.gpt-oss-120b — judge"],
  },
  {
    title: "Model access",
    items: ["Amazon Bedrock Mantle", "OpenAI Python SDK (OpenAI-compatible endpoint)"],
  },
  {
    title: "This case study",
    items: ["Next.js (App Router)", "TypeScript", "Tailwind CSS", "Hand-built SVG charts — no charting library"],
  },
];

export function TechStack() {
  return (
    <Section id="tech-stack" number="11" title="Tech Stack" width="wide">
      <div className="grid grid-cols-1 gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
        {GROUPS.map((group) => (
          <div key={group.title}>
            <h3 className="text-sm font-medium text-ink">{group.title}</h3>
            <ul className="mt-3 space-y-1.5">
              {group.items.map((item) => (
                <li key={item} className="font-data text-sm text-ink-muted">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </Section>
  );
}
