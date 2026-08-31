import type { ReactNode } from "react";

interface SectionProps {
  id: string;
  number: string;
  title: string;
  /** Narrow = comfortable reading measure for prose. Wide = room for tables/charts. */
  width?: "narrow" | "wide";
  children: ReactNode;
}

/**
 * The one structural device used for every section on the page: a hairline
 * top rule, a numbered eyebrow (the content really is a 12-part sequence,
 * so the numbering encodes real information), and a title. No cards, no
 * shadows, no background panels — sections are separated by rhythm and a
 * single 1px rule, nothing else.
 */
export function Section({ id, number, title, width = "narrow", children }: SectionProps) {
  return (
    <section id={id} className="scroll-mt-20 border-t border-rule py-16 sm:py-20">
      <div
        className={
          width === "narrow"
            ? "mx-auto max-w-[42rem] px-6"
            : "mx-auto max-w-[68rem] px-6"
        }
      >
        <div className="mb-10 flex items-baseline gap-4 sm:mb-12">
          <span className="font-data text-sm tabular-nums text-ink-faint">{number}</span>
          <h2 className="font-display text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
            {title}
          </h2>
        </div>
        {children}
      </div>
    </section>
  );
}
