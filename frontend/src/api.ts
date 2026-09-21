const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type DocumentMeta = {
  id: string;
  filename: string;
  title: string;
  char_count: number;
  chunk_count: number;
  uploaded_at: string;
};

export type Source = {
  id: number;
  filename: string;
  chunk_index: number;
  snippet: string;
  score: number;
  doc_id?: string | null;
};

export type ChatResponse = {
  answer: string;
  plan: string;
  sources: Source[];
  tool_trace: Array<{ tool: string; args: Record<string, unknown>; output: string }>;
  mode: string;
  steps: string[];
};

export type Status = {
  app: string;
  mode: string;
  documents: number;
  chunks: number;
  openai_configured: boolean;
  tamus_configured?: boolean;
  llm_configured?: boolean;
};

export type EvalSummary = {
  created_at: string;
  mode: string;
  baseline_score: number;
  improved_score: number;
  improvement_pct: number;
  notes: string;
  cases: Array<Record<string, unknown>>;
  report_path?: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  status: () => request<Status>("/api/status"),
  documents: () => request<DocumentMeta[]>("/api/documents"),
  upload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentMeta>("/api/documents/upload", { method: "POST", body: form });
  },
  ingestText: (title: string, text: string) =>
    request<DocumentMeta>("/api/documents/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text }),
    }),
  deleteDocument: (id: string) =>
    request<{ deleted: boolean }>(`/api/documents/${id}`, { method: "DELETE" }),
  chat: (question: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  runEval: () => request<EvalSummary>("/api/eval/run", { method: "POST" }),
  latestEval: () => request<EvalSummary | null>("/api/eval/latest"),
};
