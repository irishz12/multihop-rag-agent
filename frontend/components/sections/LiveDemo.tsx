"use client";

import { useState } from "react";

import { Section } from "@/components/ui/Section";

interface HopInfo {
  hop_number: number;
  query: string;
  new_chunks: number;
}

interface AskResponse {
  answer: string;
  hops: HopInfo[];
  retrieval_calls: number;
  controller_calls: number;
  documents_used: string[];
  latency_ms: number;
  estimated_cost_usd: number | null;
  stop_reason: string;
}

type Status = "idle" | "loading" | "success" | "error";

const API_URL = process.env.NEXT_PUBLIC_DEMO_API_URL ?? "http://localhost:8000";
const MAX_QUESTION_LENGTH = 500;
const CLIENT_TIMEOUT_MS = 95_000;

// `next dev` sets NODE_ENV to "development"; `next build` (this static site's
// only deployment path) always sets it to "production" — Next.js inlines
// this at build time, so the branch below is resolved once, at build time,
// with no runtime env lookup and no extra configuration for whoever deploys
// this. The interactive demo needs a live FastAPI backend on localhost, which
// only exists when this repository is cloned and run locally; the deployed
// static site has no backend to call, so it shows a note instead.
const IS_STATIC_DEPLOYMENT = process.env.NODE_ENV === "production";

interface LiveDemoProps {
  exampleQuestions: string[];
}

/**
 * The one interactive section on the page — everything else here is a
 * plain fetch to the existing, unmodified Agentic Multi-Hop RAG pipeline
 * (see backend/app.py). This component owns no retrieval/generation logic
 * of its own; it only renders the fields that endpoint already returns.
 */
export function LiveDemo({ exampleQuestions }: LiveDemoProps) {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  async function submit(rawQuestion: string) {
    const trimmed = rawQuestion.trim();
    if (!trimmed) {
      setStatus("error");
      setErrorMessage("Type a question first.");
      return;
    }
    if (trimmed.length > MAX_QUESTION_LENGTH) {
      setStatus("error");
      setErrorMessage(`Keep questions under ${MAX_QUESTION_LENGTH} characters.`);
      return;
    }

    setStatus("loading");
    setErrorMessage("");
    setResponse(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CLIENT_TIMEOUT_MS);

    try {
      const res = await fetch(`${API_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const body: unknown = await res.json().catch(() => null);
        setStatus("error");
        setErrorMessage(describeErrorResponse(res.status, body));
        return;
      }

      const data = (await res.json()) as AskResponse;
      setResponse(data);
      setStatus("success");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setErrorMessage("The pipeline took too long to respond. Try a shorter or simpler question.");
      } else {
        setErrorMessage("Can't reach the demo backend right now. It may be offline.");
      }
      setStatus("error");
    } finally {
      clearTimeout(timeoutId);
    }
  }

  function handleExampleClick(exampleQuestion: string) {
    setQuestion(exampleQuestion);
    void submit(exampleQuestion);
  }

  return (
    <Section id="live-demo" number="03" title="Live Agentic RAG Demo" width="wide">
      <p className="mb-8 max-w-[42rem] text-[1.0625rem] leading-relaxed text-ink-muted">
        Ask a real question against the same indexed corpus, through the same
        frozen pipeline measured throughout this page. This runs live — it
        retrieves, reranks, checks evidence sufficiency, and answers with the
        exact code behind every number above it.
      </p>

      {IS_STATIC_DEPLOYMENT ? (
        <p className="max-w-[42rem] border-t border-rule pt-8 text-[1.0625rem] leading-relaxed text-ink-muted">
          Live demo available locally from the repository. This section calls
          a separate FastAPI backend (<span className="font-data text-ink">backend/</span>) that
          only runs alongside a local clone — it isn&rsquo;t part of this
          static deployment. Clone the repository and follow{" "}
          <span className="font-data text-ink">backend/README.md</span> to ask
          a real question against the live pipeline.
        </p>
      ) : (
        <>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submit(question);
            }}
            className="max-w-[42rem]"
          >
            <label htmlFor="live-demo-question" className="mb-2 block text-sm font-medium text-ink">
              Your question
            </label>
            <textarea
              id="live-demo-question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              maxLength={MAX_QUESTION_LENGTH}
              rows={3}
              placeholder="Ask something a single source might not fully answer…"
              className="w-full resize-none border border-rule bg-transparent p-3 text-[1.0625rem] text-ink placeholder:text-ink-faint focus-visible:outline-2 focus-visible:outline-accent"
            />
            <div className="mt-3 flex items-center justify-between">
              <button
                type="submit"
                disabled={status === "loading"}
                className="border border-ink px-5 py-2 text-sm font-medium text-ink transition-colors hover:bg-ink hover:text-bg disabled:cursor-not-allowed disabled:opacity-40"
              >
                {status === "loading" ? "Asking…" : "Ask"}
              </button>
              <span className="tnum font-data text-xs text-ink-faint">
                {question.length}/{MAX_QUESTION_LENGTH}
              </span>
            </div>
          </form>

          {exampleQuestions.length > 0 ? (
            <div className="mt-6 max-w-[42rem]">
              <p className="mb-2 text-xs font-medium tracking-wide text-ink-faint uppercase">
                Or try a real development-split question
              </p>
              <ul className="space-y-1.5">
                {exampleQuestions.map((exampleQuestion) => (
                  <li key={exampleQuestion}>
                    <button
                      type="button"
                      onClick={() => handleExampleClick(exampleQuestion)}
                      disabled={status === "loading"}
                      className="text-left text-sm text-ink-muted underline decoration-rule underline-offset-4 hover:text-ink hover:decoration-ink disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {exampleQuestion}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="mt-10 max-w-[42rem] border-t border-rule pt-8" aria-live="polite">
            {status === "loading" ? (
              <p className="text-sm text-ink-muted">
                Retrieving, reranking, and checking evidence sufficiency — this can take up to
                half a minute if the pipeline needs a follow-up hop.
              </p>
            ) : null}

            {status === "error" ? (
              <p className="text-sm text-ink-muted">
                <span className="font-medium text-accent-2">Couldn&rsquo;t complete that request.</span>{" "}
                {errorMessage}
              </p>
            ) : null}

            {status === "success" && response ? (
              <div>
                <p className="text-[1.0625rem] leading-relaxed text-ink">{response.answer}</p>

                <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-4 text-xs text-ink-muted sm:grid-cols-4">
                  <div>
                    <dt className="tracking-wide text-ink-faint uppercase">Retrieval hops</dt>
                    <dd className="tnum font-data mt-1 text-sm text-ink">{response.retrieval_calls}</dd>
                  </div>
                  <div>
                    <dt className="tracking-wide text-ink-faint uppercase">Controller calls</dt>
                    <dd className="tnum font-data mt-1 text-sm text-ink">{response.controller_calls}</dd>
                  </div>
                  <div>
                    <dt className="tracking-wide text-ink-faint uppercase">Latency</dt>
                    <dd className="tnum font-data mt-1 text-sm text-ink">
                      {(response.latency_ms / 1000).toFixed(1)}s
                    </dd>
                  </div>
                  <div>
                    <dt className="tracking-wide text-ink-faint uppercase">Estimated cost</dt>
                    <dd className="tnum font-data mt-1 text-sm text-ink">
                      {response.estimated_cost_usd != null ? `$${response.estimated_cost_usd.toFixed(5)}` : "—"}
                    </dd>
                  </div>
                </dl>

                {response.documents_used.length > 0 ? (
                  <div className="mt-6">
                    <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                      Source documents used
                    </p>
                    <ul className="mt-2 space-y-1">
                      {response.documents_used.map((title) => (
                        <li key={title} className="text-sm text-ink-muted">
                          {title}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {response.hops.length > 1 ? (
                  <div className="mt-6">
                    <p className="text-xs font-medium tracking-wide text-ink-faint uppercase">
                      Follow-up queries
                    </p>
                    <ol className="mt-2 space-y-1">
                      {response.hops.slice(1).map((hop) => (
                        <li key={hop.hop_number} className="text-sm text-ink-muted">
                          Hop {hop.hop_number}: {hop.query}
                        </li>
                      ))}
                    </ol>
                  </div>
                ) : null}

                <p className="mt-6 text-xs text-ink-faint">
                  Stop reason: <span className="font-data">{response.stop_reason}</span>
                </p>
              </div>
            ) : null}
          </div>
        </>
      )}
    </Section>
  );
}

function describeErrorResponse(status: number, body: unknown): string {
  const detail = extractDetail(body);
  if (detail) return detail;
  if (status === 429) return "Too many questions in a row — wait a moment and try again.";
  if (status === 422) return "That question isn't valid — try rephrasing it.";
  if (status === 504) return "The pipeline took too long to respond.";
  return "The demo couldn't answer that just now.";
}

function extractDetail(body: unknown): string | null {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0 && typeof detail[0]?.msg === "string") {
      return detail[0].msg;
    }
  }
  return null;
}
