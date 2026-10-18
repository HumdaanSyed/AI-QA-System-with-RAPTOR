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
    <div className="flex min-h-dvh flex-col md:flex-row">
      <aside className="border-b border-zinc-200 bg-zinc-100 p-4 md:w-72 md:shrink-0 md:border-b-0 md:border-r dark:border-zinc-800 dark:bg-zinc-950">
        <h1 className="mb-6 text-xl font-bold">RAPTOR RAG</h1>
        <KnowledgeBases
          items={kbs}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onCreated={(created) => {
            setKbs((all) => [...all, created]);
            setSelectedId(created.id);
          }}
        />
      </aside>

      <main className="flex-1 p-4 md:p-8">
        {error && (
          <p role="alert" className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">
            {error}
          </p>
        )}
        {kb ? (
          <div className="mx-auto flex max-w-6xl flex-col gap-6">
            <header className="flex flex-wrap items-center gap-3">
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
            </header>
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
              <Documents key={`docs-${kb.id}`} kbId={kb.id} />
              <Chat key={`chat-${kb.id}`} kbId={kb.id} />
            </div>
          </div>
        ) : (
          !error && <p className="text-zinc-500">Create a knowledge base to get started.</p>
        )}
      </main>
    </div>
  );
}
