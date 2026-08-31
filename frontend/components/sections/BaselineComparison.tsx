import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { Section } from "@/components/ui/Section";
import { PIPELINE_LABEL, formatPercent, formatScore } from "@/lib/format";
import type { PipelineKey, RetrievalMethodAggregate } from "@/lib/types";

interface RetrievalRow {
  key: "dense" | "hybrid" | "hybrid_reranker";
  metrics: RetrievalMethodAggregate;
}

interface CoverageRow {
  key: PipelineKey;
  coverage: number;
}

interface BaselineComparisonProps {
  retrievalRows: RetrievalRow[];
  coverageRows: CoverageRow[];
  nNonNullDevelopment: number;
}

export function BaselineComparison({
  retrievalRows,
  coverageRows,
  nNonNullDevelopment,
}: BaselineComparisonProps) {
  const retrievalColumns: DataTableColumn<RetrievalRow>[] = [
    { key: "method", label: "Method", render: (r) => PIPELINE_LABEL[r.key] },
    { key: "recall10", label: "Recall@10", align: "right", render: (r) => formatPercent(r.metrics["recall@10"]) },
    { key: "hit10", label: "Hit@10", align: "right", render: (r) => formatPercent(r.metrics["hit@10"]) },
    { key: "mrr10", label: "MRR@10", align: "right", render: (r) => formatScore(r.metrics["mrr@10"]) },
    { key: "ndcg10", label: "NDCG@10", align: "right", render: (r) => formatScore(r.metrics["ndcg@10"]) },
    {
      key: "ce10",
      label: "Complete-Evidence@10",
      align: "right",
      render: (r) => formatPercent(r.metrics["complete_evidence@10"]),
    },
  ];

  const coverageColumns: DataTableColumn<CoverageRow>[] = [
    { key: "method", label: "Pipeline", render: (r) => PIPELINE_LABEL[r.key] },
    { key: "coverage", label: "Evidence coverage", align: "right", render: (r) => formatPercent(r.coverage) },
  ];

  return (
    <Section id="baselines" number="04" title="Baseline Comparison" width="wide">
      <p className="mb-8 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
        Three single-pass baselines establish how far retrieval alone can go.
        Agentic Multi-Hop RAG is the main system this project builds; Adaptive
        RAG is measured as a cost-optimized alternative to it, not a fourth
        independent baseline.
      </p>

      <h3 className="mb-4 text-sm font-medium text-ink">
        Retrieval quality — {nNonNullDevelopment} non-null development questions
      </h3>
      <DataTable
        columns={retrievalColumns}
        rows={retrievalRows}
        caption="Retrieval quality of the three single-pass baselines on the development split"
      />

      <h3 className="mt-12 mb-4 text-sm font-medium text-ink">
        Evidence coverage, all five pipelines — 50-question development sample
      </h3>
      <DataTable
        columns={coverageColumns}
        rows={coverageRows}
        emphasize={(r) => r.key === "agentic_multi_hop" || r.key === "adaptive_rag"}
        caption="Evidence coverage across all five pipelines on the development sample"
      />
      <p className="mt-4 text-sm text-ink-muted">
        Evidence coverage is the fraction of a question&rsquo;s gold source
        documents that actually reached the final-answer prompt — a stricter,
        end-to-end measure than retrieval-ranking metrics alone.
      </p>
    </Section>
  );
}
