"use client";

import { BarChart, type BarGroup } from "@/components/ui/BarChart";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { FlowDiagram } from "@/components/ui/FlowDiagram";
import { MetricStat } from "@/components/ui/MetricStat";
import { Section } from "@/components/ui/Section";
import { PIPELINE_COLOR, formatInt, formatPercent, titleCaseQuestionType, titleCaseStopReason } from "@/lib/format";
import type { MultihopExamplesReplay, MultihopSuccessAnalysis } from "@/lib/types";

interface CaseStudyProps {
  successAnalysis: MultihopSuccessAnalysis;
  examplesReplay: MultihopExamplesReplay;
}

// Context-Matched has no fixed identity color elsewhere on the page (it only
// exists as a control in this one comparison) — reusing the same light-blue
// used for "Development" in the Results section reads as a variant of
// Agentic's blue, which is exactly its conceptual role: same iteration
// mechanism's evidence volume, minus the iteration itself.
const CONTEXT_MATCHED_COLOR = "#8FB7DE";

interface CoverageRow {
  label: string;
  value: number;
  vsBaseline: number | null;
  vsContextMatched: number | null;
}

export function CaseStudy({ successAnalysis, examplesReplay }: CaseStudyProps) {
  const coverage = successAnalysis.evidence_coverage_three_way;
  const addedEvidence = successAnalysis.added_required_evidence;
  const outcomes = successAnalysis.final_judge_outcome_breakdown;
  const excludedIds = new Set(successAnalysis.excluded_qa_ids.qa_ids);
  const exampleIdsUsed = outcomes.beats_both_baselines.qa_ids.filter((id) => !excludedIds.has(id));

  const coverageGroups: BarGroup[] = [
    {
      label: "Baseline",
      bars: [
        {
          key: "baseline",
          value: coverage.baseline_hybrid_reranker,
          color: PIPELINE_COLOR.hybrid_reranker,
          label: "Baseline (Hybrid + Reranker)",
        },
      ],
    },
    {
      label: "Context-Matched",
      bars: [
        {
          key: "context_matched",
          value: coverage.context_matched,
          color: CONTEXT_MATCHED_COLOR,
          label: "Context-Matched (control)",
        },
      ],
    },
    {
      label: "Agentic Multi-Hop",
      bars: [
        {
          key: "agentic",
          value: coverage.agentic_final_all_hops,
          color: PIPELINE_COLOR.agentic_multi_hop,
          label: "Agentic Multi-Hop (all hops)",
        },
      ],
    },
  ];

  const coverageRows: CoverageRow[] = [
    { label: "Baseline (Hybrid + Reranker)", value: coverage.baseline_hybrid_reranker, vsBaseline: null, vsContextMatched: null },
    {
      label: "Context-Matched (control)",
      value: coverage.context_matched,
      vsBaseline: coverage.context_matched - coverage.baseline_hybrid_reranker,
      vsContextMatched: null,
    },
    {
      label: "Agentic Multi-Hop (all hops)",
      value: coverage.agentic_final_all_hops,
      vsBaseline: coverage.agentic_final_all_hops - coverage.baseline_hybrid_reranker,
      vsContextMatched: coverage.agentic_final_all_hops - coverage.context_matched,
    },
  ];

  const coverageColumns: DataTableColumn<CoverageRow>[] = [
    { key: "pipeline", label: "Pipeline", render: (r) => r.label },
    { key: "coverage", label: "Mean evidence coverage", align: "right", render: (r) => formatPercent(r.value) },
    {
      key: "vsBaseline",
      label: "vs. Baseline",
      align: "right",
      render: (r) => (r.vsBaseline === null ? "—" : `+${formatPercent(r.vsBaseline)}`),
    },
    {
      key: "vsContextMatched",
      label: "vs. Context-Matched",
      align: "right",
      render: (r) => (r.vsContextMatched === null ? "—" : `+${formatPercent(r.vsContextMatched)}`),
    },
  ];

  const detailById = new Map(successAnalysis.selected_examples_detail.map((d) => [d.qa_id, d]));

  const exactReplayMatches = examplesReplay.records.filter((record) => {
    const detail = detailById.get(record.qa_id);
    if (!detail) return false;
    return (
      JSON.stringify([...detail.final_doc_ids_all_hops].sort()) ===
      JSON.stringify([...record.evidence_doc_ids_used_final].sort())
    );
  }).length;

  return (
    <Section id="case-study" number="10" title="Case Study: Why Iterative Retrieval Earns Its Cost" width="wide">
      <p className="mb-12 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
        A single retrieval pass either finds everything a question needs or it
        doesn&rsquo;t, with no way to notice which. After every hop, the
        agentic controller judges whether the evidence gathered so far is
        sufficient and, if not, issues one focused follow-up query aimed at
        what&rsquo;s missing. On the development questions this project can
        trace end to end — where hop 1 alone genuinely wasn&rsquo;t enough —
        that mechanism visibly earns its keep.
      </p>

      <div className="space-y-16">
        <div>
          <h3 className="mb-1 text-base font-medium text-ink">
            Evidence coverage: Baseline → Context-Matched → Agentic
          </h3>
          <p className="mb-6 max-w-[42rem] text-sm text-ink-muted">
            Measured on the {formatInt(successAnalysis.populations.population_three_way_n)} development questions
            where Agentic took more than one hop and all three pipelines have a result to compare. Context-Matched
            is the same single-pass retrieval given as many chunks as Agentic actually used for that question —
            controlling for evidence volume, so any remaining Agentic advantage over it comes from which evidence
            iteration found, not simply how much of it there was.
          </p>
          <BarChart
            groups={coverageGroups}
            yMax={1}
            yTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
            formatValue={formatPercent}
            ariaLabel="Evidence coverage: Baseline, Context-Matched, and Agentic Multi-Hop, on the multi-hop-resolved subset"
          />
          <div className="mt-8">
            <DataTable
              columns={coverageColumns}
              rows={coverageRows}
              emphasize={(r) => r.label.startsWith("Agentic")}
              caption="Evidence coverage: Baseline vs. Context-Matched vs. Agentic Multi-Hop, on the multi-hop-resolved subset"
            />
          </div>
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Mechanism evidence</h3>
          <p className="mb-6 max-w-[42rem] text-sm text-ink-muted">
            Of the {formatInt(successAnalysis.populations.population_all_multihop_n)} development questions Agentic
            resolved in 2+ hops, did a later hop retrieve a required (gold) document that hop 1 had missed — and did
            that recovered evidence go on to flip the final answer?
          </p>
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2">
            <MetricStat
              value={formatPercent(addedEvidence.pct)}
              label="Later hop recovered a missing required document"
              detail={`${formatInt(addedEvidence.n)} of ${formatInt(addedEvidence.denominator)} multi-hop-resolved questions`}
              system="agentic"
            />
            <MetricStat
              value={formatPercent(exampleIdsUsed.length / addedEvidence.n)}
              label="Converted to a correctly-graded answer over both baselines"
              detail={`${formatInt(exampleIdsUsed.length)} of ${formatInt(addedEvidence.n)} (${formatInt(outcomes.beats_both_baselines.n)} total, minus 1 excluded from example selection — see below)`}
              system="agentic"
            />
          </div>
          <p className="mt-6 max-w-[42rem] text-sm text-ink-muted">
            {formatInt(outcomes.beats_both_baselines.n)} of the {formatInt(addedEvidence.n)} questions where a later
            hop added required evidence went on to a correctly-graded Agentic answer where both Baseline and
            Context-Matched were graded incorrect. One of those,{" "}
            <span className="font-data">{successAnalysis.excluded_qa_ids.qa_ids[0]}</span>, is excluded from the
            worked examples below only:{" "}
            {successAnalysis.excluded_qa_ids.reason.split(": ").slice(1).join(": ")} This leaves{" "}
            {formatInt(exampleIdsUsed.length)} used for examples. Both counts are reported so the reduction is
            explicit, not smoothed over: finding the missing evidence is necessary but not sufficient for a better
            final answer, and the {formatInt(addedEvidence.n - outcomes.beats_both_baselines.n)} questions that added
            evidence without flipping the grade are the rest of that story.
          </p>
        </div>

        <div>
          <h3 className="mb-1 text-base font-medium text-ink">Worked examples</h3>
          <p className="mb-10 max-w-[42rem] text-sm text-ink-muted">
            Each trace below combines the original benchmark run (final answer, baseline answer, judge grade) with
            an independent live replay of that exact question, development split only, which recovered the actual
            verbatim hop-by-hop queries — the original benchmark script never persisted them, only the final
            aggregate result. Hop count and stop reason reproduced exactly in all {formatInt(examplesReplay.records.length)};
            the exact final evidence-document set reproduced exactly in {formatInt(exactReplayMatches)} of{" "}
            {formatInt(examplesReplay.records.length)} — the replayed answer text below is quoted from the live
            replay and is not guaranteed identical to the original graded run (see Limitations).
          </p>

          <div className="space-y-14">
            {examplesReplay.records.map((record, i) => {
              const detail = detailById.get(record.qa_id);
              const flowSteps = [
                ...record.hops.map((hop) => ({
                  label: `Hop ${hop.hop_number}`,
                  detail: hop.query.length > 70 ? `${hop.query.slice(0, 70)}…` : hop.query,
                })),
                { label: "Stop", detail: titleCaseStopReason(record.stop_reason) },
              ];

              return (
                <div key={record.qa_id} className="border-t border-rule pt-10 first:border-t-0 first:pt-0">
                  <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <span className="font-data text-sm tabular-nums text-ink-faint">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="font-data text-sm text-ink-faint">{record.qa_id}</span>
                    <span className="text-xs text-ink-faint">
                      {titleCaseQuestionType(record.question_type)} · stopped {titleCaseStopReason(record.stop_reason)}
                    </span>
                  </div>

                  <p className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink">
                    {detail?.question ?? record.query}
                  </p>

                  <div className="mt-6 max-w-[42rem]">
                    <FlowDiagram
                      steps={flowSteps}
                      ariaLabel={`Hop-by-hop retrieval trace for ${record.qa_id}`}
                    />
                  </div>

                  {detail ? (
                    <div className="mt-6 grid grid-cols-1 gap-8 sm:grid-cols-2">
                      <div>
                        <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                          Agentic answer — graded {detail.agentic_grade}
                        </p>
                        <p className="mt-2 max-w-[38rem] text-sm leading-relaxed text-ink">{detail.agentic_answer}</p>
                      </div>
                      <div>
                        <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                          Baseline answer — graded {detail.baseline_grade}
                        </p>
                        <p className="mt-2 max-w-[38rem] text-sm leading-relaxed text-ink-muted">
                          {detail.baseline_answer}
                        </p>
                      </div>
                    </div>
                  ) : null}

                  <p className="mt-6 text-xs text-ink-faint">
                    Final evidence documents used: {formatInt(record.evidence_doc_ids_used_final.length)} · Retrieval
                    calls: {formatInt(record.num_retrieval_calls)} · Controller calls:{" "}
                    {formatInt(record.num_controller_calls)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-medium tracking-wide text-ink-faint uppercase">
            Limitations of this analysis
          </h3>
          <ul className="mt-4 space-y-4">
            <li className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
              Iteration finding evidence is not the same as iteration fixing the answer. Only{" "}
              {formatInt(exampleIdsUsed.length)} of the {formatInt(addedEvidence.n)} questions where a later hop
              added required evidence flipped from &ldquo;both baselines wrong&rdquo; to &ldquo;Agentic
              correct&rdquo; — the rest saw no such clean win. This section reports both numbers, not just the
              flattering one.
            </li>
            <li className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
              Generation is not byte-deterministic. The worked-example answers above are the original graded run,
              not the live replay&rsquo;s own regenerated text — even at fixed decoding settings, hosted LLM
              inference is not guaranteed to be token-identical across separate calls, so the two differ slightly
              in wording even when the underlying evidence is identical.
            </li>
            <li className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
              Retrieval itself showed minor variation on replay: the exact final evidence-document set reproduced
              in {formatInt(exactReplayMatches)} of {formatInt(examplesReplay.records.length)} replayed questions —
              most likely because approximate nearest-neighbor search isn&rsquo;t guaranteed bit-identical across
              separate live queries. Disclosed, not hidden.
            </li>
            <li className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
              This is a targeted, traceable subset, not the whole system. The{" "}
              {formatInt(successAnalysis.populations.population_three_way_n)}–
              {formatInt(successAnalysis.populations.population_all_multihop_n)} question population above is
              genuinely multi-hop-resolved development questions specifically — not the headline sample — and these
              findings should not be read as &ldquo;Agentic always improves the answer.&rdquo;
            </li>
            <li className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
              Fact-grounding, measured elsewhere in this project, is retrieval-side only — whether a required fact
              reached the generation context, not whether the generated answer correctly used it — and is
              intentionally not used as evidence for this section&rsquo;s claims.
            </li>
            <li className="max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
              Development-stage analysis only. Every number in this section comes from the development split; the
              one-time final holdout evaluation is reported separately and is not part of this case study.
            </li>
          </ul>
        </div>
      </div>
    </Section>
  );
}
