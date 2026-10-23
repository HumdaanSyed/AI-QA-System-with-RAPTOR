"use client";

import { useEffect, useState } from "react";
import { api, type Doc, type DocStatus } from "@/lib/api";

const LABELS: Record<DocStatus, string> = {
  queued: "Queued",
  parsing: "Reading PDF",
  embedding: "Embedding",
  building_tree: "Building RAPTOR tree",
  ready: "Ready",
  failed: "Failed",
};

const isWorking = (d: Doc) => d.status !== "ready" && d.status !== "failed";

const statusText = (d: Doc) => `${LABELS[d.status]}${d.status === "embedding" ? ` ${d.progress}%` : ""}`;

// "Quorvia_Employee_Handbook.pdf" -> "Quorvia Employee Handbook"
const displayName = (filename: string) => filename.replace(/\.pdf$/i, "").replaceAll("_", " ");

function Icon({ path }: { path: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d={path} />
    </svg>
  );
}

const REINDEX = "M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6";
const TRASH = "M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6";

/** Compact document list shown under the selected knowledge base in the sidebar. */
export function Documents({ kbId }: { kbId: string }) {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0); // bump to reload the list
  const reload = () => setVersion((v) => v + 1);

  useEffect(() => {
    let active = true; // ignore responses that arrive after a newer request
    api.status(kbId).then(
      (status) => active && setDocs(status.documents),
      (e: Error) => active && setError(e.message),
    );
    return () => {
      active = false;
    };
  }, [kbId, version]);

  // Poll while any document is still being processed.
  const working = docs.some(isWorking);
  useEffect(() => {
    if (!working) return;
    const timer = setInterval(reload, 2000);
    return () => clearInterval(timer);
  }, [working]);

  async function run(action: () => Promise<unknown>) {
    setError(null);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    }
    reload();
  }

  async function upload(files: FileList | null) {
    for (const file of Array.from(files ?? [])) await run(() => api.upload(kbId, file));
  }

  return (
    <section aria-label="Documents" className="ml-3 mt-1 flex flex-col gap-1 border-l border-zinc-300 pl-2 dark:border-zinc-700">
      <ul aria-live="polite" className="flex flex-col">
        {docs.length === 0 && <li className="px-1.5 py-1 text-xs text-zinc-500">No documents yet.</li>}
        {docs.map((doc) => (
          <li key={doc.id} className="group rounded px-1.5 py-1 hover:bg-zinc-200 dark:hover:bg-zinc-800">
            <div className="flex items-center gap-2">
              <span
                title={statusText(doc)}
                className={`h-2 w-2 shrink-0 rounded-full ${
                  doc.status === "ready" ? "bg-green-500" : doc.status === "failed" ? "bg-red-500" : "animate-pulse bg-blue-500"
                }`}
              />
              <span className="sr-only">{statusText(doc)}:</span>
              <span className="min-w-0 flex-1 truncate text-xs" title={doc.filename}>
                {displayName(doc.filename)}
              </span>
              <span className="flex shrink-0 gap-0.5 opacity-60 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                <button
                  onClick={() => run(() => api.reindexDoc(kbId, doc.id))}
                  disabled={isWorking(doc)}
                  aria-label={`Re-index ${doc.filename}`}
                  title="Re-index"
                  className="rounded p-0.5 hover:bg-zinc-300 hover:text-blue-700 disabled:opacity-30 dark:hover:bg-zinc-700 dark:hover:text-blue-400"
                >
                  <Icon path={REINDEX} />
                </button>
                <button
                  onClick={() => {
                    if (confirm(`Delete "${doc.filename}" and everything indexed from it?`))
                      void run(() => api.deleteDoc(kbId, doc.id));
                  }}
                  aria-label={`Delete ${doc.filename}`}
                  title="Delete"
                  className="rounded p-0.5 hover:bg-zinc-300 hover:text-red-700 dark:hover:bg-zinc-700 dark:hover:text-red-400"
                >
                  <Icon path={TRASH} />
                </button>
              </span>
            </div>
            {isWorking(doc) && (
              <div className="ml-4 mt-1 flex items-center gap-2">
                <div
                  role="progressbar"
                  aria-label={`${doc.filename} progress`}
                  aria-valuenow={doc.status === "embedding" ? doc.progress : undefined}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="h-1 flex-1 overflow-hidden rounded bg-zinc-300 dark:bg-zinc-700"
                >
                  <div
                    className={`h-full bg-blue-500 ${doc.status === "embedding" ? "" : "w-1/3 animate-pulse"}`}
                    style={doc.status === "embedding" ? { width: `${doc.progress}%` } : undefined}
                  />
                </div>
                <span className="shrink-0 text-[10px] text-zinc-500">{statusText(doc)}</span>
              </div>
            )}
            {doc.error && <p className="ml-4 mt-0.5 text-[11px] text-red-600">{doc.error}</p>}
          </li>
        ))}
      </ul>
      {error && (
        <p role="alert" className="px-1.5 text-xs text-red-600">
          {error}
        </p>
      )}
      <label className="cursor-pointer self-start rounded px-1.5 py-1 text-xs font-medium text-blue-700 focus-within:outline-2 focus-within:outline-blue-600 hover:bg-zinc-200 dark:text-blue-400 dark:hover:bg-zinc-800">
        + Upload PDFs
        <input
          type="file"
          accept="application/pdf"
          multiple
          className="sr-only"
          onChange={(e) => {
            void upload(e.target.files);
            e.target.value = "";
          }}
        />
      </label>
    </section>
  );
}
