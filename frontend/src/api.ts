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
  grounded?: boolean;
  confidence?: number;
  memory_used?: boolean;
};

export type StreamHandlers = {
  onStatus?: (step: string) => void;
  onPlan?: (plan: string, memoryUsed: boolean) => void;
  onTool?: (tool: { tool: string; args: Record<string, unknown>; output: string }) => void;
  onSources?: (sources: Source[]) => void;
  onToken?: (text: string) => void;
  onReplace?: (answer: string) => void;
  onFinal?: (result: ChatResponse) => void;
  onError?: (detail: string) => void;
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
  dataset?: string | null;
  case_count?: number | null;
  domain_baseline_score?: number | null;
  domain_improved_score?: number | null;
  domain_improvement_pct?: number | null;
  hf_baseline_score?: number | null;
  hf_improved_score?: number | null;
  hf_improvement_pct?: number | null;
  citation_score?: number | null;
  citation_coverage?: number | null;
  citation_validity?: number | null;
  citation_support?: number | null;
  refusal_accuracy?: number | null;
  unanswerable_baseline_score?: number | null;
  unanswerable_improved_score?: number | null;
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

function parseSseChunk(buffer: string): { events: Array<{ event: string; data: string }>; rest: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  const events: Array<{ event: string; data: string }> = [];
  for (const block of parts) {
    if (!block.trim()) continue;
    let event = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    events.push({ event, data: dataLines.join("\n") });
  }
  return { events, rest };
}

async function chatStream(
  question: string,
  history: Array<{ role: string; content: string }>,
  handlers: StreamHandlers,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : "Stream failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: ChatResponse | null = null;
  let streamError: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseChunk(buffer);
    buffer = parsed.rest;
    for (const evt of parsed.events) {
      let data: Record<string, unknown> = {};
      try {
        data = evt.data ? (JSON.parse(evt.data) as Record<string, unknown>) : {};
      } catch {
        continue;
      }
      if (evt.event === "status" && typeof data.step === "string") handlers.onStatus?.(data.step);
      else if (evt.event === "plan") {
        handlers.onPlan?.(String(data.plan ?? ""), Boolean(data.memory_used));
      } else if (evt.event === "tool") {
        handlers.onTool?.(data as { tool: string; args: Record<string, unknown>; output: string });
      } else if (evt.event === "sources" && Array.isArray(data.sources)) {
        handlers.onSources?.(data.sources as Source[]);
      } else if (evt.event === "token" && typeof data.text === "string") {
        handlers.onToken?.(data.text);
      } else if (evt.event === "replace" && typeof data.answer === "string") {
        handlers.onReplace?.(data.answer);
      } else if (evt.event === "final") {
        finalResult = data as unknown as ChatResponse;
        handlers.onFinal?.(finalResult);
      } else if (evt.event === "error") {
        streamError = String(data.detail ?? "Stream error");
        handlers.onError?.(streamError);
      }
    }
  }

  if (streamError) throw new Error(streamError);
  if (!finalResult) throw new Error("Stream ended without a final answer");
  return finalResult;
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
  chat: (question: string, history: Array<{ role: string; content: string }> = []) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    }),
  chatStream,
  runEval: () => request<EvalSummary>("/api/eval/run", { method: "POST" }),
  latestEval: () => request<EvalSummary | null>("/api/eval/latest"),
};
