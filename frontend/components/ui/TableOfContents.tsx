const SECTIONS = [
  { id: "problem", number: "01", title: "Business Problem" },
  { id: "architecture", number: "02", title: "Architecture" },
  { id: "live-demo", number: "03", title: "Live Agentic RAG Demo" },
  { id: "baselines", number: "04", title: "Baseline Comparison" },
  { id: "holdout", number: "05", title: "Final Holdout Results" },
  { id: "results", number: "06", title: "Results" },
  { id: "journey", number: "07", title: "Experiment Journey" },
  { id: "failure-analysis", number: "08", title: "Failure Analysis" },
  { id: "methodology", number: "09", title: "Evaluation Methodology" },
  { id: "tech-stack", number: "10", title: "Tech Stack" },
  { id: "findings", number: "11", title: "Key Findings & Limitations" },
  { id: "reproducibility", number: "12", title: "Reproducibility" },
];

/** A plain-text contents list, doubling as the page's only navigation — no
 * persistent header bar, no icons, just links to real sections below. */
export function TableOfContents() {
  return (
    <nav aria-label="Table of contents" className="border-t border-rule py-10">
      <div className="mx-auto max-w-[42rem] px-6">
        <ol className="grid grid-cols-1 gap-x-8 gap-y-2 sm:grid-cols-2">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                className="group flex items-baseline gap-3 py-1 text-sm text-ink-muted transition-colors hover:text-ink"
              >
                <span className="font-data text-ink-faint">{s.number}</span>
                <span className="border-b border-transparent group-hover:border-ink-muted">
                  {s.title}
                </span>
              </a>
            </li>
          ))}
        </ol>
      </div>
    </nav>
  );
}
