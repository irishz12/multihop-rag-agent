"use client";

import { useId, useState } from "react";

export interface BarDatum {
  key: string;
  value: number;
  color: string;
  label: string;
}

export interface BarGroup {
  label: string;
  bars: BarDatum[];
}

export interface LegendItem {
  key: string;
  label: string;
  color: string;
}

interface BarChartProps {
  groups: BarGroup[];
  yMax: number;
  yTicks: number[];
  formatValue: (value: number) => string;
  formatTick?: (value: number) => string;
  legend?: LegendItem[];
  ariaLabel: string;
  height?: number;
}

const VIEW_WIDTH = 640;

/**
 * One reusable chart for every result comparison on the page — a single
 * bar per group (cost, latency, evidence coverage, answer quality) or
 * several bars per group (query-type / hop-count / development-vs-holdout
 * breakdowns), depending on what `groups` contains. Colors are passed in
 * by the caller and always come from the same fixed Agentic/Adaptive/
 * baseline palette used throughout the page — never re-picked per chart.
 *
 * Hovering or focusing a bar reveals its exact value in the reading line
 * below the chart, rather than a floating tooltip — every bar is a real
 * `<button>`, so this works identically with a mouse, a keyboard, or a
 * screen reader. A visually hidden table beneath the chart carries the
 * same data for anyone not using the visual chart at all.
 */
export function BarChart({
  groups,
  yMax,
  yTicks,
  formatValue,
  formatTick,
  legend,
  ariaLabel,
  height = 260,
}: BarChartProps) {
  const [active, setActive] = useState<{ group: string; bar: BarDatum } | null>(null);
  const titleId = useId();

  const tickFormat = formatTick ?? formatValue;
  // Scale left padding to the widest formatted tick label so longer strings
  // (e.g. "$0.0012") never clip against the SVG's left edge — short labels
  // (e.g. "80%") get a snug axis instead of always paying for the worst case.
  const maxTickChars = Math.max(...yTicks.map((t) => tickFormat(t).length));
  const paddingLeft = 22 + maxTickChars * 6.5;
  const paddingBottom = 40;
  const paddingTop = 12;
  const plotWidth = VIEW_WIDTH - paddingLeft - 12;
  const plotHeight = height - paddingTop - paddingBottom;
  const groupWidth = plotWidth / groups.length;

  const scaleY = (value: number) => plotHeight - (value / yMax) * plotHeight;

  return (
    <div>
      {legend && legend.length > 1 ? (
        <ul className="mb-3 flex flex-wrap gap-x-5 gap-y-1.5">
          {legend.map((item) => (
            <li key={item.key} className="flex items-center gap-2 text-xs text-ink-muted">
              <span
                aria-hidden
                className="inline-block h-2.5 w-2.5 shrink-0"
                style={{ backgroundColor: item.color }}
              />
              {item.label}
            </li>
          ))}
        </ul>
      ) : null}

      <svg
        viewBox={`0 0 ${VIEW_WIDTH} ${height}`}
        role="group"
        aria-labelledby={titleId}
        className="w-full"
        style={{ height: "auto" }}
      >
        {/* role="group", not "img" — this chart contains genuinely focusable
            bars (below), and ARIA's "img" role asserts an atomic graphic
            with no separately-navigable content, which would conflict with
            that. <title> still gives the whole chart one accessible name. */}
        <title id={titleId}>{ariaLabel}</title>

        {/* gridlines + y-axis labels */}
        {yTicks.map((tick) => {
          const y = paddingTop + scaleY(tick);
          return (
            <g key={tick}>
              <line
                x1={paddingLeft}
                x2={VIEW_WIDTH - 12}
                y1={y}
                y2={y}
                stroke="var(--color-rule)"
                strokeWidth={1}
              />
              <text
                x={paddingLeft - 10}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-ink-faint font-data"
                fontSize={11}
              >
                {tickFormat(tick)}
              </text>
            </g>
          );
        })}

        {/* bars */}
        {groups.map((group, gi) => {
          const groupX = paddingLeft + gi * groupWidth;
          const barGap = 6;
          const barWidth = (groupWidth - barGap * (group.bars.length + 1)) / group.bars.length;

          return (
            <g key={group.label}>
              {group.bars.map((bar, bi) => {
                const barHeight = (bar.value / yMax) * plotHeight;
                const x = groupX + barGap + bi * (barWidth + barGap);
                const y = paddingTop + plotHeight - barHeight;
                const isActive = active?.bar.key === bar.key + group.label;

                return (
                  <g key={bar.key}>
                    <rect
                      x={x}
                      y={y}
                      width={Math.max(barWidth, 1)}
                      height={Math.max(barHeight, 0)}
                      fill={bar.color}
                      opacity={active && !isActive ? 0.45 : 1}
                      style={{ transition: "opacity 150ms ease" }}
                    />
                    <text
                      x={x + barWidth / 2}
                      y={y - 6}
                      textAnchor="middle"
                      className="fill-ink-muted font-data"
                      fontSize={10}
                    >
                      {formatValue(bar.value)}
                    </text>
                    {/* Invisible, focusable hit target covering the full column height — bigger
                        than the visible bar so short bars are still easy to hover/tap/focus. */}
                    <rect
                      x={x - barGap / 2}
                      y={paddingTop}
                      width={barWidth + barGap}
                      height={plotHeight}
                      fill="transparent"
                      tabIndex={0}
                      role="button"
                      aria-label={`${group.label}, ${bar.label}: ${formatValue(bar.value)}`}
                      onMouseEnter={() => setActive({ group: group.label, bar: { ...bar, key: bar.key + group.label } })}
                      onMouseLeave={() => setActive(null)}
                      onFocus={() => setActive({ group: group.label, bar: { ...bar, key: bar.key + group.label } })}
                      onBlur={() => setActive(null)}
                      className="cursor-pointer outline-none"
                    />
                  </g>
                );
              })}
              <text
                x={groupX + groupWidth / 2}
                y={height - paddingBottom + 18}
                textAnchor="middle"
                className="fill-ink-muted"
                fontSize={11}
              >
                {group.label}
              </text>
            </g>
          );
        })}

        <line
          x1={paddingLeft}
          x2={paddingLeft}
          y1={paddingTop}
          y2={paddingTop + plotHeight}
          stroke="var(--color-rule)"
          strokeWidth={1}
        />
      </svg>

      <p className="tnum mt-2 h-5 font-data text-xs text-ink-muted" aria-live="polite">
        {active ? `${active.group} — ${active.bar.label}: ${formatValue(active.bar.value)}` : " "}
      </p>

      <table className="sr-only">
        <caption>{ariaLabel}</caption>
        <thead>
          <tr>
            <th scope="col">Group</th>
            <th scope="col">Series</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {groups.flatMap((group) =>
            group.bars.map((bar) => (
              <tr key={group.label + bar.key}>
                <td>{group.label}</td>
                <td>{bar.label}</td>
                <td>{formatValue(bar.value)}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}
