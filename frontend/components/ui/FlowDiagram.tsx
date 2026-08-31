interface FlowStep {
  label: string;
  detail?: string;
}

interface FlowDiagramProps {
  steps: FlowStep[];
  loopBack?: { fromIndex: number; toIndex: number; note: string };
  ariaLabel: string;
}

/**
 * A text-first process diagram: labeled steps connected by hairline rules
 * and a plain arrow glyph, no pictograms. Built as an ordered list so a
 * screen reader announces it as the sequence it is, not a picture.
 */
export function FlowDiagram({ steps, loopBack, ariaLabel }: FlowDiagramProps) {
  return (
    <div aria-label={ariaLabel}>
      <ol className="flex flex-col gap-0 md:flex-row md:flex-wrap md:items-stretch md:gap-0">
        {steps.map((step, i) => (
          <li key={step.label} className="flex items-stretch md:contents">
            <div className="flex flex-1 flex-col justify-center border border-rule px-4 py-3 md:w-[9.75rem] md:flex-none">
              <span className="font-data text-[0.65rem] tracking-wide text-ink-faint uppercase">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="mt-1 text-sm font-medium text-ink">{step.label}</span>
              {step.detail ? (
                <span className="mt-0.5 text-xs text-ink-muted">{step.detail}</span>
              ) : null}
            </div>
            {i < steps.length - 1 ? (
              <div
                aria-hidden
                className="flex w-8 shrink-0 items-center justify-center text-ink-faint md:w-8"
              >
                <span className="hidden md:inline">→</span>
                <span className="md:hidden">↓</span>
              </div>
            ) : null}
          </li>
        ))}
      </ol>
      {loopBack ? (
        <p className="mt-4 text-sm text-ink-muted">
          <span aria-hidden className="mr-1.5 text-ink-faint">
            ↻
          </span>
          Step {String(loopBack.fromIndex + 1).padStart(2, "0")} repeats step{" "}
          {String(loopBack.toIndex + 1).padStart(2, "0")} onward — {loopBack.note}
        </p>
      ) : null}
    </div>
  );
}
