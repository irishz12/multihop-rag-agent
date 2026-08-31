import { Section } from "@/components/ui/Section";

interface ReproducibilityProps {
  sampleSeed: number;
  routerCvSeed: number;
}

const COMMANDS = [
  "python scripts/build_index.py && python scripts/build_hybrid_index.py",
  "python scripts/run_retrieval_eval.py",
  "python scripts/select_phase9_sample.py",
  "python scripts/run_phase9_benchmark.py --pipeline adaptive",
  "python scripts/analyze_phase9_sample.py",
  "pytest",
];

export function Reproducibility({ sampleSeed, routerCvSeed }: ReproducibilityProps) {
  return (
    <Section id="reproducibility" number="13" title="Reproducibility">
      <div className="grid grid-cols-1 gap-10 lg:grid-cols-2">
        <div className="space-y-4 text-[1.0625rem] leading-relaxed text-ink-muted">
          <p>
            Every retrieval, routing, and generation step is deterministic given
            its frozen inputs — no sampling temperature above zero anywhere in
            the pipeline, a fixed cross-validation seed (
            <span className="tnum font-data text-ink">{routerCvSeed}</span>) for
            router training, and a fixed stratified-sampling seed (
            <span className="tnum font-data text-ink">{sampleSeed}</span>) for
            both evaluation samples.
          </p>
          <p>
            Full commands, configuration file layout, and the final holdout
            evaluation&rsquo;s exact sequence live in the project{" "}
            <span className="font-data text-ink">README</span>. This page reads
            the same committed result artifacts directly — it does not call a
            model, and it cannot run or repeat any evaluation.
          </p>
        </div>
        <div>
          <pre
            tabIndex={0}
            aria-label="Key reproduction commands"
            className="overflow-x-auto border border-rule p-4 font-data text-xs leading-relaxed text-ink-muted"
          >
            {COMMANDS.map((cmd) => `${cmd}\n`).join("")}
          </pre>
        </div>
      </div>
    </Section>
  );
}
