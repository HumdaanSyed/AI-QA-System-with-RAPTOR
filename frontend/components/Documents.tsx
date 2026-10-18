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
    <section aria-labelledby="docs-heading" className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 id="docs-heading" className="text-lg font-semibold">
          Documents
        </h2>
        <label className="cursor-pointer whitespace-nowrap rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white focus-within:outline-2 focus-within:outline-blue-600 hover:bg-blue-700">
          Upload PDFs
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
      </div>
      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}
      <ul aria-live="polite" className="flex flex-col gap-2">
        {docs.length === 0 && <li className="text-sm text-zinc-500">Upload a PDF to get started.</li>}
        {docs.map((doc) => (
          <li
            key={doc.id}
            className="flex flex-col gap-2 rounded-md border border-zinc-200 p-3 dark:border-zinc-800"
          >
            <div className="flex flex-col gap-0.5">
              <span className="truncate text-sm font-medium" title={doc.filename}>
                {doc.filename}
              </span>
              <span
                className={`text-xs ${
                  doc.status === "failed"
                    ? "text-red-600"
                    : doc.status === "ready"
                      ? "text-green-700 dark:text-green-500"
                      : "text-zinc-500"
                }`}
              >
                {LABELS[doc.status]}
                {doc.status === "embedding" && ` ${doc.progress}%`}
              </span>
            </div>
            {isWorking(doc) && (
              <div
                role="progressbar"
                aria-label={`${doc.filename} progress`}
                aria-valuenow={doc.status === "embedding" ? doc.progress : undefined}
                aria-valuemin={0}
                aria-valuemax={100}
                className="h-1.5 overflow-hidden rounded bg-zinc-200 dark:bg-zinc-800"
              >
                <div
                  className={`h-full bg-blue-600 ${doc.status === "embedding" ? "" : "w-1/3 animate-pulse"}`}
                  style={doc.status === "embedding" ? { width: `${doc.progress}%` } : undefined}
                />
              </div>
            )}
            {doc.error && <p className="text-xs text-red-600">{doc.error}</p>}
            <div className="flex gap-3 text-xs">
              <button
                onClick={() => run(() => api.reindexDoc(kbId, doc.id))}
                disabled={isWorking(doc)}
                className="text-blue-700 hover:underline disabled:opacity-40 dark:text-blue-400"
              >
                Re-index
              </button>
              <button
                onClick={() => {
                  if (confirm(`Delete "${doc.filename}" and everything indexed from it?`))
                    void run(() => api.deleteDoc(kbId, doc.id));
                }}
                className="text-red-700 hover:underline dark:text-red-400"
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
