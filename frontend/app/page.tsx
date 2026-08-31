import { Architecture } from "@/components/sections/Architecture";
import { BaselineComparison } from "@/components/sections/BaselineComparison";
import { BusinessProblem } from "@/components/sections/BusinessProblem";
import { ExperimentJourney } from "@/components/sections/ExperimentJourney";
import { FailureAnalysis } from "@/components/sections/FailureAnalysis";
import { FinalHoldoutResults } from "@/components/sections/FinalHoldoutResults";
import { Footer } from "@/components/sections/Footer";
import { Hero } from "@/components/sections/Hero";
import { KeyFindings } from "@/components/sections/KeyFindings";
import { LiveDemo } from "@/components/sections/LiveDemo";
import { Methodology } from "@/components/sections/Methodology";
import { Reproducibility } from "@/components/sections/Reproducibility";
import { Results } from "@/components/sections/Results";
import { TechStack } from "@/components/sections/TechStack";
import { TableOfContents } from "@/components/ui/TableOfContents";
import {
  getDevBaselineCostLatency,
  getExampleQuestions,
  getHoldoutConsumed,
  getHoldoutReport,
  getRetrievalEval,
  getRouterModel,
  getSampleReport,
} from "@/lib/data";
import type { PipelineKey } from "@/lib/types";

// This is the ONLY value on the page not read from a results/*.json file —
// the number of passing tests in the Python evaluation suite, which has no
// JSON artifact of its own. Re-verify with `pytest -q` before publishing;
// last verified at the time this page was written.
const LAST_VERIFIED_PASSING_TEST_COUNT = 632;

export default function Home() {
  const sample = getSampleReport();
  const holdout = getHoldoutReport();
  const retrievalEval = getRetrievalEval();
  const routerModel = getRouterModel();
  const holdoutConsumed = getHoldoutConsumed();
  const devBaselineCostLatency = getDevBaselineCostLatency();
  const exampleQuestions = getExampleQuestions();

  const retrievalRows = (["dense", "hybrid", "hybrid_reranker"] as const).map((key) => ({
    key,
    metrics: retrievalEval.aggregate[key],
  }));
  const coverageRows = (["dense", "hybrid", "hybrid_reranker", "agentic_multi_hop", "adaptive_rag"] as PipelineKey[]).map(
    (key) => ({ key, coverage: sample.evidence_coverage_mean[key] ?? 0 }),
  );

  return (
    <main>
      <Hero
        agenticQuality={holdout.combined_quality_mean.agentic_multi_hop ?? 0}
        adaptiveQuality={holdout.combined_quality_mean.adaptive_rag ?? 0}
        retentionPct={holdout.adaptive_quality_retention_pct_vs_agentic_multi_hop}
      />
      <TableOfContents />
      <BusinessProblem />
      <Architecture />
      <LiveDemo exampleQuestions={exampleQuestions} />
      <BaselineComparison
        retrievalRows={retrievalRows}
        coverageRows={coverageRows}
        nNonNullDevelopment={retrievalEval.counts.non_null_evaluated}
      />
      <FinalHoldoutResults
        agenticQuality={holdout.combined_quality_mean.agentic_multi_hop ?? 0}
        adaptiveQuality={holdout.combined_quality_mean.adaptive_rag ?? 0}
        agenticCoverage={holdout.evidence_coverage_mean.agentic_multi_hop ?? 0}
        adaptiveCoverage={holdout.evidence_coverage_mean.adaptive_rag ?? 0}
        costReductionPct={holdout.cost_latency.cost_reduction_pct}
        latencyReductionPct={holdout.cost_latency.latency_reduction_pct}
        sampleSize={holdout.sample_size}
      />
      <Results sample={sample} holdout={holdout} devBaselineCostLatency={devBaselineCostLatency} />
      <ExperimentJourney retrievalEval={retrievalEval} holdout={holdout} />
      <FailureAnalysis sample={sample} holdout={holdout} />
      <Methodology holdoutConsumed={holdoutConsumed} passingTestCount={LAST_VERIFIED_PASSING_TEST_COUNT} />
      <TechStack />
      <KeyFindings holdout={holdout} />
      <Reproducibility sampleSeed={sample.sample_seed} routerCvSeed={routerModel.cv_seed} />
      <Footer />
    </main>
  );
}
