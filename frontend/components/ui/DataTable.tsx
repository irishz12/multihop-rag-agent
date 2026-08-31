import type { ReactNode } from "react";

export interface DataTableColumn<Row> {
  key: string;
  label: string;
  align?: "left" | "right";
  render: (row: Row) => ReactNode;
}

interface DataTableProps<Row> {
  columns: DataTableColumn<Row>[];
  rows: Row[];
  /** Predicate marking a row as one of the two systems under comparison. */
  emphasize?: (row: Row) => boolean;
  caption?: string;
}

/**
 * One reusable table for every tabular comparison on the page. Hairline
 * rules only — a top rule on the header, a rule under every row, nothing
 * else. Emphasized rows (Agentic / Adaptive) get medium weight, not a
 * background fill or border, to keep the "no cards" rule intact.
 */
export function DataTable<Row>({ columns, rows, emphasize, caption }: DataTableProps<Row>) {
  return (
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label={caption ?? "Data table"}>
      <table className="w-full border-collapse text-left">
        {/* No forced min-width here — a short two-column table (e.g. evidence
            coverage) should stay comfortably narrow on mobile; a wide table
            (e.g. the six-column retrieval comparison) gets its natural
            content width and the surrounding overflow-x-auto scrolls it. */}
        <caption className="sr-only">{caption ?? "Data table"}</caption>
        <thead>
          <tr className="border-t border-b border-rule">
            {columns.map((col) => (
              <th
                key={col.key}
                scope="col"
                className={`py-3 pr-6 text-xs font-medium tracking-wide text-ink-faint uppercase ${
                  col.align === "right" ? "text-right" : "text-left"
                }`}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const isEmphasized = emphasize?.(row) ?? false;
            return (
              <tr key={i} className="border-b border-rule">
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`py-3 pr-6 text-sm ${col.align === "right" ? "text-right tnum font-data" : ""} ${
                      isEmphasized ? "font-medium text-ink" : "text-ink-muted"
                    }`}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
