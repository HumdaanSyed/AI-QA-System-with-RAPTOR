"use client";

import { useEffect, useState } from "react";
import { Chat } from "@/components/Chat";
import { Documents } from "@/components/Documents";
import { KnowledgeBases } from "@/components/KnowledgeBases";
import { api, type KnowledgeBase } from "@/lib/api";

export default function Home() {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const kb = kbs.find((k) => k.id === selectedId);

  useEffect(() => {
    api
      .listKBs()
      .then((list) => {
        setKbs(list);
        setSelectedId(list[0]?.id ?? null);
      })
      .catch(() => setError("Can't reach the API. Is the backend running on port 8000?"));
  }, []);

  async function rename() {
    if (!kb) return;
    const name = prompt("Rename knowledge base", kb.name)?.trim();
    if (!name || name === kb.name) return;
    try {
      const updated = await api.renameKB(kb.id, name);
      setKbs((all) => all.map((k) => (k.id === kb.id ? updated : k)));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove() {
    if (!kb || !confirm(`Delete "${kb.name}" and all of its documents?`)) return;
    try {
      await api.deleteKB(kb.id);
      const rest = kbs.filter((k) => k.id !== kb.id);
      setKbs(rest);
      setSelectedId(rest[0]?.id ?? null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="flex h-dvh">
      {sidebarOpen && (
        <aside
          id="sidebar"
          className="fixed inset-y-0 left-0 z-20 w-72 shrink-0 overflow-y-auto border-r border-zinc-200 bg-zinc-100 p-4 shadow-lg md:static md:shadow-none dark:border-zinc-800 dark:bg-zinc-950"
        >
          <div className="mb-6 flex items-center justify-between">
            <h1 className="text-xl font-bold">RAPTOR RAG</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              aria-label="Hide sidebar"
              aria-controls="sidebar"
              aria-expanded
              className="rounded p-1 text-zinc-500 hover:bg-zinc-200 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
            >
              «
            </button>
          </div>
          <KnowledgeBases
            items={kbs}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onCreated={(created) => {
              setKbs((all) => [...all, created]);
              setSelectedId(created.id);
            }}
            selectedContent={kb && <Documents key={`docs-${kb.id}`} kbId={kb.id} />}
          />
        </aside>
      )}

      <main className="flex min-w-0 flex-1 flex-col overflow-y-auto p-4 md:p-8">
        <header className="mx-auto mb-6 flex w-full max-w-3xl flex-wrap items-center gap-3">
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              aria-label="Show sidebar"
              aria-controls="sidebar"
              aria-expanded={false}
              className="rounded border border-zinc-300 px-2 py-1 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              ☰
            </button>
          )}
          {kb && (
            <>
              <h2 className="text-2xl font-semibold">{kb.name}</h2>
              <span className="rounded bg-zinc-200 px-2 py-0.5 text-xs dark:bg-zinc-800">
                RAPTOR {kb.raptor_enabled ? "on" : "off"}
              </span>
              <div className="ml-auto flex gap-3 text-sm">
                <button onClick={rename} className="hover:underline">
                  Rename
                </button>
                <button onClick={remove} className="text-red-700 hover:underline dark:text-red-400">
                  Delete
                </button>
              </div>
            </>
          )}
        </header>
        <div className="mx-auto w-full max-w-3xl">
          {error && (
            <p role="alert" className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">
              {error}
            </p>
          )}
          {kb ? (
            <Chat key={`chat-${kb.id}`} kbId={kb.id} />
          ) : (
            !error && <p className="text-zinc-500">Create a knowledge base to get started.</p>
          )}
        </div>
      </main>
    </div>
  );
}
