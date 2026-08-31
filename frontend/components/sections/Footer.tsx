export function Footer() {
  return (
    <footer className="border-t border-rule py-10">
      <div className="mx-auto max-w-[42rem] px-6 text-sm text-ink-faint">
        <p>
          Agentic Multi-Hop RAG — a case study built on{" "}
          <a
            href="https://github.com/yixuantt/MultiHop-RAG"
            className="underline decoration-rule underline-offset-4 hover:decoration-ink-muted"
          >
            MultiHop-RAG
          </a>{" "}
          (Tang &amp; Yang, 2024, COLM 2024). All figures on this page are read
          directly from the project&rsquo;s committed evaluation artifacts.
        </p>
        <p className="mt-4">
          RISHIKESH K G —{" "}
          <a
            href="mailto:irishz121212@gmail.com"
            className="underline decoration-rule underline-offset-4 hover:decoration-ink-muted"
          >
            irishz121212@gmail.com
          </a>
        </p>
      </div>
    </footer>
  );
}
