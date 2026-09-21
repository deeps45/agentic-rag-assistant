import { useEffect, useRef, useState, useTransition } from "react";
import type { FormEvent } from "react";
import {
  ArrowUpRight,
  FileText,
  LoaderCircle,
  Sparkles,
  Trash2,
  Upload,
  Waypoints,
} from "lucide-react";
import { api } from "./api";
import type { ChatResponse, DocumentMeta, EvalSummary, Status } from "./api";

type ChatTurn = {
  role: "user" | "assistant";
  content: string;
  meta?: ChatResponse;
};

const SUGGESTIONS = [
  "What is retrieval augmented generation and why use it?",
  "How does FAISS help in a knowledge assistant?",
  "What steps does an agentic RAG assistant take?",
  "How can evaluation improve answer relevance?",
];

function formatPct(n: number) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [evalReport, setEvalReport] = useState<EvalSummary | null>(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const [textTitle, setTextTitle] = useState("");
  const [textBody, setTextBody] = useState("");
  const [pending, startTransition] = useTransition();
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function refresh() {
    const [s, d, e] = await Promise.all([
      api.status(),
      api.documents(),
      api.latestEval().catch(() => null),
    ]);
    startTransition(() => {
      setStatus(s);
      setDocs(d);
      setEvalReport(e);
    });
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  async function onUpload(file: File) {
    setError(null);
    setBusy(true);
    try {
      await api.upload(file);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function onIngestText(e: FormEvent) {
    e.preventDefault();
    if (!textTitle.trim() || !textBody.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.ingestText(textTitle.trim(), textBody.trim());
      setTextTitle("");
      setTextBody("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setBusy(false);
    }
  }

  async function onDelete(id: string) {
    setBusy(true);
    try {
      await api.deleteDocument(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setBusy(false);
    }
  }

  async function ask(q: string) {
    const cleaned = q.trim();
    if (!cleaned || busy) return;
    setQuestion("");
    setBusy(true);
    setError(null);
    const history = turns.map((t) => ({ role: t.role, content: t.content }));
    setTurns((prev) => [
      ...prev,
      { role: "user", content: cleaned },
      {
        role: "assistant",
        content: "",
        meta: {
          answer: "",
          plan: "",
          sources: [],
          tool_trace: [],
          mode: status?.mode ?? "…",
          steps: [],
          grounded: true,
          confidence: 0,
          memory_used: false,
        },
      },
    ]);
    try {
      const res = await api.chatStream(cleaned, history, {
        onStatus: (step) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && last.meta) {
              last.meta = { ...last.meta, steps: [...(last.meta.steps || []), step] };
            }
            return next;
          });
        },
        onPlan: (plan, memoryUsed) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && last.meta) {
              last.meta = { ...last.meta, plan, memory_used: memoryUsed };
            }
            return next;
          });
        },
        onTool: (tool) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && last.meta) {
              last.meta = {
                ...last.meta,
                tool_trace: [...(last.meta.tool_trace || []), tool],
              };
            }
            return next;
          });
        },
        onSources: (sources) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant" && last.meta) {
              last.meta = { ...last.meta, sources };
            }
            return next;
          });
        },
        onToken: (text) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              last.content = `${last.content}${text}`;
            }
            return next;
          });
        },
        onReplace: (answer) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              last.content = answer;
            }
            return next;
          });
        },
        onFinal: (result) => {
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              last.content = result.answer;
              last.meta = result;
            }
            return next;
          });
        },
      });
      void res;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Chat failed";
      setError(message);
      setTurns((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          last.content =
            message.includes("rate-limited") || message.includes("429")
              ? "The LLM provider is temporarily rate-limited. Please wait a few seconds and try again."
              : `Could not synthesize an answer: ${message}`;
        }
        return next;
      });
    } finally {
      setBusy(false);
    }
  }

  async function onEval() {
    setEvalBusy(true);
    setError(null);
    try {
      const report = await api.runEval();
      setEvalReport(report);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setEvalBusy(false);
    }
  }

  const latest = [...turns].reverse().find((t) => t.role === "assistant" && t.meta);

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-5 pb-2 pt-6 md:px-8">
        <div className="flex items-center gap-3">
          <div className="relative grid h-11 w-11 place-items-center rounded-2xl bg-[var(--accent)] text-[var(--signal)]">
            <Waypoints className="h-5 w-5" />
            <span className="orbit absolute inset-[-6px] rounded-full border border-dashed border-[var(--accent)]/30" />
          </div>
          <div>
            <p className="display text-2xl leading-none">Groundline</p>
            <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
              Agentic RAG
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
          <span className="hidden rounded-full border border-[var(--line)] bg-white/50 px-3 py-1 sm:inline">
            {status ? `${status.documents} docs · ${status.chunks} chunks` : "Loading…"}
          </span>
          <span className="rounded-full bg-[var(--accent)] px-3 py-1 text-[var(--signal)]">
            {status?.mode ?? "…"}
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 pb-16 md:px-8">
        <section className="relative mt-6 overflow-hidden rounded-[2rem] border border-[var(--line)] bg-[var(--accent)] text-[#f4fff7]">
          <div
            className="absolute inset-0 opacity-70"
            style={{
              backgroundImage:
                "linear-gradient(120deg, rgba(200,242,91,0.18), transparent 40%), radial-gradient(circle at 80% 20%, rgba(255,255,255,0.16), transparent 35%), url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22800%22 height=%22400%22 viewBox=%220 0 800 400%22%3E%3Cpath d=%22M0 280C120 220 180 160 280 170C420 185 460 260 600 230C700 210 760 150 800 120V400H0Z%22 fill=%22%230a2c2d%22/%3E%3Cpath d=%22M0 320C140 270 220 240 340 250C480 265 520 310 650 290C720 280 770 250 800 230V400H0Z%22 fill=%22%23081f20%22/%3E%3C/svg%3E')",
              backgroundSize: "cover",
              backgroundPosition: "center bottom",
            }}
          />
          <div className="relative grid gap-8 px-6 py-10 md:grid-cols-[1.2fr_0.8fr] md:px-10 md:py-14">
            <div>
              <p className="rise text-sm uppercase tracking-[0.22em] text-[var(--signal)]">
                Knowledge assistant
              </p>
              <h1 className="display rise rise-delay-1 mt-3 max-w-xl text-5xl leading-[0.95] md:text-6xl">
                Groundline
              </h1>
              <p className="rise rise-delay-2 mt-4 max-w-lg text-base text-[#d7ebe4] md:text-lg">
                Plan, retrieve, tool-call, and synthesize source-grounded answers across your
                documents — with measurable relevance gains.
              </p>
              <div className="rise rise-delay-3 mt-7 flex flex-wrap gap-3">
                <a
                  href="#ask"
                  className="inline-flex items-center gap-2 rounded-full bg-[var(--signal)] px-5 py-2.5 text-sm font-semibold text-[var(--accent)] transition hover:brightness-105"
                >
                  Ask the corpus <ArrowUpRight className="h-4 w-4" />
                </a>
                <a
                  href="#ingest"
                  className="inline-flex items-center gap-2 rounded-full border border-white/25 px-5 py-2.5 text-sm text-white/90"
                >
                  Ingest documents
                </a>
              </div>
            </div>
            <div className="rise rise-delay-2 flex flex-col justify-end gap-3 self-end">
              <div className="signal-bar h-1 w-40 rounded-full bg-[var(--signal)]" />
              <p className="display text-3xl text-[var(--signal)]">
                {evalReport ? formatPct(evalReport.improvement_pct) : "+25% target"}
              </p>
              <p className="text-sm text-[#c5ddd6]">
                Measured answer-relevance lift vs. naive top-1 chunk baseline after agentic
                retrieval tuning.
              </p>
            </div>
          </div>
        </section>

        {error && (
          <div className="mt-5 rounded-2xl border border-red-300/60 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        )}

        <div className="mt-8 grid gap-6 lg:grid-cols-[0.95fr_1.35fr]">
          <section
            id="ingest"
            className="rounded-[1.6rem] border border-[var(--line)] bg-[var(--panel)] p-5 backdrop-blur md:p-6"
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="display text-3xl">Corpus</h2>
                <p className="mt-1 text-sm text-[var(--muted)]">
                  Upload files or paste text for semantic indexing.
                </p>
              </div>
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-full bg-[var(--accent)] px-4 py-2 text-sm text-[var(--signal)] disabled:opacity-60"
              >
                <Upload className="h-4 w-4" /> Upload
              </button>
              <input
                ref={fileRef}
                type="file"
                className="hidden"
                accept=".txt,.md,.markdown,.pdf,.docx,.csv"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void onUpload(file);
                  e.target.value = "";
                }}
              />
            </div>

            <form onSubmit={onIngestText} className="mt-5 space-y-3">
              <input
                value={textTitle}
                onChange={(e) => setTextTitle(e.target.value)}
                placeholder="Document title"
                className="w-full rounded-xl border border-[var(--line)] bg-white/70 px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
              />
              <textarea
                value={textBody}
                onChange={(e) => setTextBody(e.target.value)}
                placeholder="Paste knowledge text to index…"
                rows={4}
                className="w-full resize-y rounded-xl border border-[var(--line)] bg-white/70 px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
              />
              <button
                type="submit"
                disabled={busy || !textTitle.trim() || !textBody.trim()}
                className="rounded-full border border-[var(--line)] px-4 py-2 text-sm font-medium disabled:opacity-50"
              >
                Index text
              </button>
            </form>

            <ul className="mt-5 max-h-72 space-y-2 overflow-auto pr-1">
              {docs.length === 0 && (
                <li className="rounded-xl border border-dashed border-[var(--line)] px-3 py-6 text-center text-sm text-[var(--muted)]">
                  {pending ? "Loading documents…" : "No documents yet. Sample docs seed on first API start."}
                </li>
              )}
              {docs.map((doc) => (
                <li
                  key={doc.id}
                  className="flex items-start justify-between gap-3 rounded-xl border border-[var(--line)] bg-white/55 px-3 py-3"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 shrink-0 text-[var(--accent)]" />
                      <p className="truncate text-sm font-semibold">{doc.title}</p>
                    </div>
                    <p className="mt-1 text-xs text-[var(--muted)]">
                      {doc.filename} · {doc.chunk_count} chunks · {doc.char_count.toLocaleString()} chars
                    </p>
                  </div>
                  <button
                    type="button"
                    aria-label={`Delete ${doc.title}`}
                    onClick={() => void onDelete(doc.id)}
                    className="rounded-lg p-1.5 text-[var(--muted)] hover:bg-black/5 hover:text-red-700"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </li>
              ))}
            </ul>

            <div className="mt-6 border-t border-[var(--line)] pt-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h3 className="display text-2xl">Evaluation</h3>
                  <p className="text-sm text-[var(--muted)]">
                    Retrieval + answer relevance vs. baseline.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void onEval()}
                  disabled={evalBusy}
                  className="inline-flex items-center gap-2 rounded-full border border-[var(--line)] px-4 py-2 text-sm disabled:opacity-60"
                >
                  {evalBusy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                  Run eval
                </button>
              </div>
              {evalReport ? (
                <div className="mt-4 space-y-3">
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="rounded-xl bg-[var(--warm)] px-2 py-3">
                      <p className="text-[11px] uppercase tracking-wider text-[var(--muted)]">Baseline</p>
                      <p className="mt-1 text-lg font-semibold">{evalReport.baseline_score.toFixed(2)}</p>
                    </div>
                    <div className="rounded-xl bg-[var(--warm)] px-2 py-3">
                      <p className="text-[11px] uppercase tracking-wider text-[var(--muted)]">Improved</p>
                      <p className="mt-1 text-lg font-semibold">{evalReport.improved_score.toFixed(2)}</p>
                    </div>
                    <div className="rounded-xl bg-[var(--accent)] px-2 py-3 text-[var(--signal)]">
                      <p className="text-[11px] uppercase tracking-wider opacity-80">Lift</p>
                      <p className="mt-1 text-lg font-semibold">{formatPct(evalReport.improvement_pct)}</p>
                    </div>
                  </div>
                  {evalReport.dataset && (
                    <p className="text-xs text-[var(--muted)]">
                      {evalReport.dataset}
                      {evalReport.case_count ? ` · ${evalReport.case_count} cases` : ""}
                      {typeof evalReport.domain_improvement_pct === "number"
                        ? ` · domain ${formatPct(evalReport.domain_improvement_pct)}`
                        : ""}
                      {typeof evalReport.hf_improvement_pct === "number"
                        ? ` · HF QA ${formatPct(evalReport.hf_improvement_pct)}`
                        : ""}
                      {typeof evalReport.citation_score === "number"
                        ? ` · cite ${(evalReport.citation_score * 100).toFixed(0)}%`
                        : ""}
                      {typeof evalReport.refusal_accuracy === "number"
                        ? ` · refuse ${(evalReport.refusal_accuracy * 100).toFixed(0)}%`
                        : ""}
                    </p>
                  )}
                </div>
              ) : (
                <p className="mt-3 text-sm text-[var(--muted)]">
                  No report yet. Run the pipeline to measure relevance lift.
                </p>
              )}
            </div>
          </section>

          <section
            id="ask"
            className="flex min-h-[34rem] flex-col rounded-[1.6rem] border border-[var(--line)] bg-[var(--panel)] backdrop-blur"
          >
            <div className="border-b border-[var(--line)] px-5 py-4 md:px-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="display text-3xl">Conversational query</h2>
                  <p className="mt-1 text-sm text-[var(--muted)]">
                  Multi-step plan → hybrid retrieval → re-rank → streamed grounded synthesis.
                  </p>
                </div>
                {turns.length > 0 && (
                  <button
                    type="button"
                    onClick={() => {
                      setTurns([]);
                      setError(null);
                    }}
                    disabled={busy}
                    className="shrink-0 rounded-full border border-[var(--line)] px-3 py-1.5 text-xs text-[var(--muted)] hover:border-[var(--accent)] hover:text-[var(--ink)] disabled:opacity-50"
                  >
                    Clear chat
                  </button>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => void ask(s)}
                    className="rounded-full border border-[var(--line)] bg-white/60 px-3 py-1 text-left text-xs text-[var(--muted)] hover:border-[var(--accent)] hover:text-[var(--ink)]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4 md:px-6">
              {turns.length === 0 && (
                <div className="rounded-2xl border border-dashed border-[var(--line)] px-4 py-10 text-center">
                  <p className="display text-2xl">Ask a multi-step document question</p>
                  <p className="mx-auto mt-2 max-w-md text-sm text-[var(--muted)]">
                    Answers cite retrieved chunks. Open the trace below each reply to inspect
                    planning and tool execution.
                  </p>
                </div>
              )}
              {turns.map((turn, idx) => (
                <div
                  key={`${turn.role}-${idx}`}
                  className={`max-w-[95%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    turn.role === "user"
                      ? "ml-auto bg-[var(--accent)] text-[#f3fff6]"
                      : "bg-white/70 text-[var(--ink)]"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{turn.content}</p>
                  {turn.meta && (
                    <details className="mt-3 rounded-xl border border-[var(--line)] bg-[var(--warm)]/70 px-3 py-2">
                      <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
                        Plan · tools · sources
                        {typeof turn.meta.confidence === "number" && (
                          <span className="ml-2 normal-case tracking-normal text-[var(--accent)]">
                            · conf {(turn.meta.confidence * 100).toFixed(0)}%
                            {turn.meta.grounded === false ? " · ungrounded" : " · grounded"}
                            {turn.meta.memory_used ? " · memory" : ""}
                          </span>
                        )}
                      </summary>
                      <div className="mt-2 space-y-3 text-xs text-[var(--muted)]">
                        <div>
                          <p className="font-semibold text-[var(--ink)]">Plan</p>
                          <pre className="mt-1 whitespace-pre-wrap font-sans">{turn.meta.plan}</pre>
                        </div>
                        <div>
                          <p className="font-semibold text-[var(--ink)]">Tool trace</p>
                          <ul className="mt-1 space-y-1">
                            {turn.meta.tool_trace.map((t, i) => (
                              <li key={`${t.tool}-${i}`}>
                                <span className="font-medium text-[var(--accent)]">{t.tool}</span>
                                <span className="block opacity-80">{t.output.slice(0, 220)}…</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="font-semibold text-[var(--ink)]">Sources</p>
                          <ul className="mt-1 space-y-2">
                            {turn.meta.sources.map((s) => (
                              <li key={s.id} className="rounded-lg bg-white/70 px-2 py-1.5">
                                <span className="font-medium text-[var(--ink)]">
                                  [{s.id}] {s.filename}
                                </span>
                                <span className="block">{s.snippet}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </details>
                  )}
                </div>
              ))}
              {busy && (
                <div className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1.5 text-xs text-[var(--muted)]">
                  <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> Streaming plan · retrieve · answer…
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            <form
              className="border-t border-[var(--line)] p-4 md:p-5"
              onSubmit={(e) => {
                e.preventDefault();
                void ask(question);
              }}
            >
              <div className="flex gap-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask across your knowledge base…"
                  className="flex-1 rounded-full border border-[var(--line)] bg-white/80 px-4 py-3 text-sm outline-none focus:border-[var(--accent)]"
                />
                <button
                  type="submit"
                  disabled={busy || !question.trim()}
                  className="rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-[var(--signal)] disabled:opacity-50"
                >
                  Ask
                </button>
              </div>
            </form>
          </section>
        </div>

        {latest?.meta && (
          <section className="mt-6 rounded-[1.6rem] border border-[var(--line)] bg-white/55 p-5 md:p-6">
            <h2 className="display text-3xl">Source panel</h2>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Evidence used in the latest grounded response.
            </p>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              {latest.meta.sources.map((s) => (
                <article key={s.id} className="rounded-2xl border border-[var(--line)] bg-[var(--warm)]/60 p-4">
                  <p className="text-xs uppercase tracking-wider text-[var(--muted)]">
                    [{s.id}] {s.filename} · chunk {s.chunk_index}
                  </p>
                  <p className="mt-2 text-sm leading-relaxed">{s.snippet}</p>
                </article>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
