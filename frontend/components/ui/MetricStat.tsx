interface MetricStatProps {
  value: string;
  label: string;
  /** Which system this measurement belongs to — colors the value like the charts do. */
  system?: "agentic" | "adaptive" | "neutral";
  detail?: string;
}

const SYSTEM_COLOR: Record<NonNullable<MetricStatProps["system"]>, string> = {
  agentic: "text-accent",
  adaptive: "text-accent-2",
  neutral: "text-ink",
};

/**
 * One measured value, rendered like an instrument reading: monospace,
 * tabular figures, a quiet label beneath. Used in the hero and the final
 * holdout highlight grid — every value passed in is a formatted string
 * from lib/format.ts, sourced from lib/data.ts, never typed by hand here.
 */
export function MetricStat({ value, label, system = "neutral", detail }: MetricStatProps) {
  return (
    <div>
      <div className={`tnum font-data text-3xl font-medium sm:text-4xl ${SYSTEM_COLOR[system]}`}>
        {value}
      </div>
      <div className="mt-2 text-sm text-ink-muted">{label}</div>
      {detail ? <div className="mt-0.5 text-xs text-ink-faint">{detail}</div> : null}
    </div>
  );
}
