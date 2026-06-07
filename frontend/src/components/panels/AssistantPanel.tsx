import { useRef, useState, useEffect } from "react";
import { Sparkles, Send, User, Loader2 } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import {
  assistantChat,
  type ChatMessage,
  type ChatSource,
} from "@/api/assistant";

interface Turn {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  grounded?: boolean;
}

const SUGGESTIONS = [
  "What cell types are in my data?",
  "What marks the clusters?",
  "What analysis steps were run?",
  "Are there any batch effects to worry about?",
];

export function AssistantPanel() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const datasetName = useDatasetStore((s) => s.currentDataset?.name);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns, busy]);

  async function send(query: string) {
    const q = query.trim();
    if (!q || busy || !datasetId) return;
    setInput("");
    const history: ChatMessage[] = turns.map((t) => ({
      role: t.role,
      content: t.content,
    }));
    setTurns((prev) => [...prev, { role: "user", content: q }]);
    setBusy(true);
    try {
      const res = await assistantChat(datasetId, q, history);
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          grounded: res.grounded,
        },
      ]);
    } catch (err) {
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry — the assistant failed: ${
            err instanceof Error ? err.message : String(err)
          }`,
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  if (!datasetId) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        Load a dataset to chat with the AI co-pilot.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 border-b px-4 py-3">
        <Sparkles className="h-5 w-5 text-primary" />
        <div>
          <h2 className="text-sm font-semibold">AI Co-pilot</h2>
          <p className="text-xs text-muted-foreground">
            Grounded in {datasetName ?? "your dataset"}&apos;s analysis &amp; results — every
            answer cites the facts it used.
          </p>
        </div>
      </div>

      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
        {turns.length === 0 && (
          <div className="mx-auto max-w-xl pt-8 text-center">
            <Sparkles className="mx-auto mb-3 h-8 w-8 text-primary/60" />
            <p className="mb-4 text-sm text-muted-foreground">
              Ask about your clusters, markers, the steps that were run, or
              single-cell analysis in general. Answers are grounded in this
              dataset and cite the facts they use.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t, i) => (
          <div key={i} className="flex gap-3">
            <div
              className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                t.role === "user"
                  ? "bg-muted text-muted-foreground"
                  : "bg-primary/15 text-primary"
              }`}
            >
              {t.role === "user" ? (
                <User className="h-4 w-4" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
            </div>
            <div className="min-w-0 flex-1">
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {t.content}
              </div>
              {t.role === "assistant" && t.grounded === false && (
                <div className="mt-1 text-xs italic text-amber-600">
                  Language model not configured — showing grounded facts directly.
                </div>
              )}
              {t.sources && t.sources.length > 0 && (
                <details className="mt-2 text-xs">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                    {t.sources.length} grounding source
                    {t.sources.length === 1 ? "" : "s"}
                  </summary>
                  <ul className="mt-1 space-y-1 border-l-2 border-muted pl-3">
                    {t.sources.map((s, j) => (
                      <li key={j}>
                        <code className="rounded bg-muted px-1 py-0.5 text-[10px] text-primary">
                          {s.ref}
                        </code>{" "}
                        <span className="text-muted-foreground">{s.detail}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}

        {busy && (
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/15 text-primary">
              <Sparkles className="h-4 w-4" />
            </div>
            <Loader2 className="h-4 w-4 animate-spin" />
            Thinking…
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t p-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="flex items-end gap-2"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            rows={1}
            placeholder="Ask about your data…"
            className="max-h-32 flex-1 resize-none rounded-lg border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
