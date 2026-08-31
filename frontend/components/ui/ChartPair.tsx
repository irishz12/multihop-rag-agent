import type { ReactNode } from "react";

interface ChartPairProps {
  leftTitle: string;
  rightTitle: string;
  left: ReactNode;
  right: ReactNode;
}

/** Two related charts (development sample / final holdout) shown side by
 * side on wide screens and stacked on narrow ones — reused by every chart
 * in the Results section that has a development-vs-holdout counterpart. */
export function ChartPair({ leftTitle, rightTitle, left, right }: ChartPairProps) {
  return (
    <div className="grid grid-cols-1 gap-10 sm:grid-cols-2 sm:gap-8">
      <div>
        <h4 className="mb-4 text-xs font-medium tracking-wide text-ink-faint uppercase">{leftTitle}</h4>
        {left}
      </div>
      <div>
        <h4 className="mb-4 text-xs font-medium tracking-wide text-ink-faint uppercase">{rightTitle}</h4>
        {right}
      </div>
    </div>
  );
}
