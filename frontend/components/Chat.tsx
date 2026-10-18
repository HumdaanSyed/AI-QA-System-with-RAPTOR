"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamQuery, type Citation } from "@/lib/api";

type Turn = { question: string; answer: string; citations: Citation[]; error?: string };

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Turn [1] / [1, 2] markers into #cite-n links, leaving code untouched. */
function linkCitations(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?```|`[^`\n]*`)/)
    .map((part, i) =>
      i % 2
        ? part
        : part.replace(CITATION, (_, ids: string) =>
            ids
              .split(",")
              .map((n) => `[\\[${n.trim()}\\]](#cite-${n.trim()})`)
              .join(""),
          ),
    )
    .join("");
}

function pages(c: Citation) {
  return c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}–${c.page_end}`;
}

function Answer({ turn, onCite }: { turn: Turn; onCite: (c: Citation) => void }) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            const id = href?.startsWith("#cite-") ? Number(href.slice(6)) : null;
            const citation = turn.citations.find((c) => c.id === id);
            if (id === null) return <a href={href}>{children}</a>;
            return citation ? (
              <button
                onClick={() => onCite(citation)}
                aria-label={`Source ${id}: ${citation.doc_name}, ${pages(citation)}`}
                className="mx-0.5 rounded bg-blue-100 px-1 text-xs font-semibold text-blue-800 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-100"
              >
                {id}
              </button>
            ) : (
              <span className="text-xs text-zinc-500">[{id}]</span>
            );
          },
        }}
      >
        {linkCitations(turn.answer)}
      </ReactMarkdown>
    </div>
  );
}

export function Chat({ kbId }: { kbId: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<Citation | null>(null);
  const controller = useRef<AbortController | null>(null);

  // Stop a running answer when the panel goes away (e.g. another knowledge base is picked).
  useEffect(() => () => controller.current?.abort(), []);

  function update(index: number, change: Partial<Turn> | ((t: Turn) => Partial<Turn>)) {
    setTurns((all) =>
      all.map((t, i) => (i === index ? { ...t, ...(typeof change === "function" ? change(t) : change) } : t)),
    );
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q) return;
    const index = turns.length;
    setTurns((all) => [...all, { question: q, answer: "", citations: [] }]);
    setQuestion("");
    setBusy(true);
    controller.current = new AbortController();
    try {
      const citations = await streamQuery(
        kbId,
        q,
        (text) => update(index, (t) => ({ answer: t.answer + text })),
        controller.current.signal,
      );
      update(index, { citations });
    } catch (e) {
      update(index, { error: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="chat-heading" className="flex min-h-0 flex-col gap-3">
      <h2 id="chat-heading" className="text-lg font-semibold">
        Ask
      </h2>
      <div aria-live="polite" className="flex flex-col gap-6">
        {turns.length === 0 && (
          <p className="text-sm text-zinc-500">
            Answers come only from your documents, with numbered sources you can click.
          </p>
        )}
        {turns.map((turn, i) => (
          <article key={i} className="flex flex-col gap-2">
            <p className="self-end rounded-lg bg-blue-600 px-3 py-2 text-sm text-white">{turn.question}</p>
            {turn.answer ? (
              <Answer turn={turn} onCite={setSource} />
            ) : (
              !turn.error && <p className="text-sm text-zinc-500">Searching your documents…</p>
            )}
            {turn.error && (
              <p role="alert" className="text-sm text-red-600">
                {turn.error}
              </p>
            )}
          </article>
        ))}
      </div>

      {source && (
        <aside
          aria-label="Source"
          className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-sm dark:border-zinc-800 dark:bg-zinc-900"
        >
          <div className="mb-2 flex items-start justify-between gap-2">
            <p className="font-medium">
              [{source.id}] {source.doc_name} · {pages(source)}
              {source.level > 0 && (
                <span className="ml-2 rounded bg-amber-100 px-1.5 text-xs text-amber-900">
                  RAPTOR summary
                </span>
              )}
            </p>
            <button onClick={() => setSource(null)} aria-label="Close source" className="text-zinc-500 hover:text-zinc-900">
              ✕
            </button>
          </div>
          <p className="max-h-64 overflow-auto whitespace-pre-wrap text-zinc-700 dark:text-zinc-300">{source.text}</p>
        </aside>
      )}

      <form onSubmit={ask} className="flex gap-2">
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <input
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          maxLength={2000}
          placeholder="Ask a question about your documents"
          className="flex-1 rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {busy ? "Answering…" : "Ask"}
        </button>
      </form>
    </section>
  );
}
