"use client";

import { useState } from "react";
import { api, type KnowledgeBase } from "@/lib/api";

type Props = {
  items: KnowledgeBase[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreated: (kb: KnowledgeBase) => void;
};

export function KnowledgeBases({ items, selectedId, onSelect, onCreated }: Props) {
  const [name, setName] = useState("");
  const [raptor, setRaptor] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function create(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      onCreated(await api.createKB(name.trim(), raptor));
      setName("");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <nav aria-label="Knowledge bases" className="flex flex-col gap-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">Knowledge bases</h2>
      <ul className="flex flex-col gap-1">
        {items.map((kb) => (
          <li key={kb.id}>
            <button
              onClick={() => onSelect(kb.id)}
              aria-current={kb.id === selectedId ? "page" : undefined}
              className={`w-full rounded-md px-3 py-2 text-left text-sm focus-visible:outline-2 focus-visible:outline-blue-600 ${
                kb.id === selectedId
                  ? "bg-blue-600 text-white"
                  : "hover:bg-zinc-200 dark:hover:bg-zinc-800"
              }`}
            >
              {kb.name}
            </button>
          </li>
        ))}
        {items.length === 0 && <li className="px-3 text-sm text-zinc-500">None yet.</li>}
      </ul>

      <form onSubmit={create} className="flex flex-col gap-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
        <label htmlFor="kb-name" className="text-sm font-medium">
          New knowledge base
        </label>
        <input
          id="kb-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={100}
          required
          placeholder="e.g. Company handbook"
          className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={raptor} onChange={(e) => setRaptor(e.target.checked)} />
          Build RAPTOR summary trees
        </label>
        <button
          type="submit"
          disabled={!name.trim()}
          className="rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          Create
        </button>
        {error && (
          <p role="alert" className="text-sm text-red-600">
            {error}
          </p>
        )}
      </form>
    </nav>
  );
}
