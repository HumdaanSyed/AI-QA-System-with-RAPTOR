export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type KnowledgeBase = {
  id: string;
  name: string;
  created_at: string;
  raptor_enabled: boolean;
};

export type DocStatus = "queued" | "parsing" | "embedding" | "building_tree" | "ready" | "failed";

export type Doc = {
  id: string;
  kb_id: string;
  filename: string;
  status: DocStatus;
  progress: number;
  error: string | null;
  created_at: string;
};

export type Citation = {
  id: number;
  doc_id: string;
  doc_name: string;
  level: number; // 0 = passage, 1+ = RAPTOR summary
  page_start: number;
  page_end: number;
  text: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const detail = (await response.json().catch(() => null))?.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const api = {
  listKBs: () => request<KnowledgeBase[]>("/kb"),
  createKB: (name: string, raptorEnabled: boolean) =>
    request<KnowledgeBase>("/kb", json("POST", { name, raptor_enabled: raptorEnabled })),
  renameKB: (id: string, name: string) => request<KnowledgeBase>(`/kb/${id}`, json("PATCH", { name })),
  deleteKB: (id: string) => request<void>(`/kb/${id}`, { method: "DELETE" }),
  status: (id: string) => request<{ kb: KnowledgeBase; documents: Doc[] }>(`/kb/${id}/status`),
  upload: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Doc>(`/kb/${id}/documents`, { method: "POST", body: form });
  },
  deleteDoc: (kbId: string, docId: string) =>
    request<void>(`/kb/${kbId}/documents/${docId}`, { method: "DELETE" }),
  history: (kbId: string) =>
    request<{ question: string; answer: string; citations: Citation[] }[]>(`/kb/${kbId}/history`),
  clearHistory: (kbId: string) => request<void>(`/kb/${kbId}/history`, { method: "DELETE" }),
  reindexDoc: (kbId: string, docId: string) =>
    request<Doc>(`/kb/${kbId}/documents/${docId}/reindex`, { method: "POST" }),
};

/** Ask a question; calls onToken as the answer streams and resolves with its citations. */
export async function streamQuery(
  kbId: string,
  question: string,
  onToken: (text: string) => void,
  signal?: AbortSignal,
): Promise<Citation[]> {
  const response = await fetch(`${API_URL}/kb/${kbId}/query`, { ...json("POST", { question }), signal });
  if (!response.ok || !response.body) throw new Error(`Request failed (${response.status})`);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let citations: Citation[] = [];
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return citations;
    buffer += value;
    // Server-sent events are separated by a blank line; JSON data never contains one.
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1];
      const data = JSON.parse(/^data: (.*)$/m.exec(block)?.[1] ?? "{}");
      if (event === "token") onToken(data.text);
      else if (event === "citations") citations = data.citations;
      else if (event === "error") throw new Error(data.message);
    }
  }
}
