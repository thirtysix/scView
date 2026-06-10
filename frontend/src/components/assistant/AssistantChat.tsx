import { useRef, useState, useEffect, useCallback } from "react";
import { Sparkles, Send, User, Loader2 } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { PANEL_LABELS, type PanelId } from "@/lib/constants";
import {
  assistantChat,
  assistantChatStream,
  type AssistantAction,
  type ChatMessage,
  type ChatSource,
  type ViewContext,
} from "@/api/assistant";
import { MarkdownLite } from "@/components/assistant/MarkdownLite";

interface Turn {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  grounded?: boolean;
  route?: string[];
  followups?: string[];
}

const ROUTE_LABELS: Record<string, string> = {
  app: "app & library",
  data: "your dataset",
  tutorials: "methods docs",
  literature: "literature",
};

const SUGGESTIONS = [
  "What cell types are in my data?",
  "What marks this cluster?",
  "What analysis steps were run?",
  "Should I run batch integration?",
];

const SUGGESTIONS_NO_DATASET = [
  "How do I load my data?",
  "What file formats does scView support?",
  "What can scView do?",
  "What datasets are already available?",
];

/** Snapshot what the user is currently looking at, so deictic questions resolve. */
function currentViewContext(): ViewContext {
  const v = useViewStore.getState();
  const sel = useSelectionStore.getState();
  const settings = useSettingsStore.getState();
  const uv = useUnifiedViewStore.getState();
  return {
    panel: PANEL_LABELS[v.activePanel],
    color_by: settings.colorBy || undefined,
    highlighted: sel.highlightedGroup,
    overlay: uv.overlayLabel || undefined,
  };
}

/** The chat surface (conversation + composer), shared by the full panel and the
 *  floating drawer. Reads the current dataset + view context from the stores. */
export function AssistantChat() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns, busy]);

  // Replace the last turn (the streaming assistant placeholder) via a mapper.
  const patchLast = (fn: (t: Turn) => Turn) =>
    setTurns((prev) => prev.map((t, i) => (i === prev.length - 1 ? fn(t) : t)));

  async function send(query: string) {
    const q = query.trim();
    if (!q || busy) return;  // datasetId may be null → app-level co-pilot
    setInput("");
    const history: ChatMessage[] = turns.map((t) => ({ role: t.role, content: t.content }));
    const vc = currentViewContext();
    // push the user turn + an empty assistant placeholder to stream into
    setTurns((prev) => [...prev, { role: "user", content: q }, { role: "assistant", content: "" }]);
    setBusy(true);
    try {
      await assistantChatStream(datasetId, q, history, vc, {
        onSources: (ev) =>
          patchLast((t) => ({ ...t, sources: ev.sources, route: ev.route, grounded: ev.grounded })),
        onDelta: (text) => patchLast((t) => ({ ...t, content: t.content + text })),
        onDone: (ev) => {
          patchLast((t) => ({ ...t, followups: ev.followups }));
          executeActions(ev.actions);
        },
        onError: (msg) =>
          patchLast((t) => ({ ...t, content: t.content || `Sorry — ${msg}` })),
      });
    } catch {
      // Streaming unavailable → fall back to the non-streaming endpoint.
      try {
        const res = await assistantChat(datasetId, q, history, vc);
        patchLast(() => ({
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          grounded: res.grounded,
          route: res.route,
          followups: res.followups,
        }));
        executeActions(res.actions);
      } catch (e2) {
        patchLast((t) => ({
          ...t,
          content: `Sorry — the assistant failed: ${
            e2 instanceof Error ? e2.message : String(e2)
          }`,
        }));
      }
    } finally {
      setBusy(false);
    }
  }

  // Execute allow-listed UI actions the co-pilot returned for a natural-language command.
  const executeActions = useCallback((actions?: AssistantAction[]) => {
    if (!actions?.length) return;
    for (const a of actions) {
      if (a.type === "set_color_by" && a.column) {
        useSettingsStore.getState().setColorBy(a.column);
        useSelectionStore.getState().setHighlight(null);
        useViewStore.getState().setPanel("unified");
      } else if (a.type === "highlight_cluster" && a.column && a.value) {
        useSettingsStore.getState().setColorBy(a.column);
        useSelectionStore.getState().setHighlight({ column: a.column, value: a.value });
        useViewStore.getState().setPanel("unified");
      } else if (a.type === "open_panel" && a.panel) {
        useViewStore.getState().setPanel(a.panel as PanelId);
      }
    }
  }, []);

  // Clicking a result citation chip jumps to that cluster/gene in the Unified View.
  const handleCitation = useCallback((tag: string) => {
    const parts = tag.split(":");
    if (parts[0] !== "result") return;
    const sub = parts[1];
    const value = parts.slice(2).join(":");
    const ds = useDatasetStore.getState().currentDataset;
    if (sub === "groups" && value) {
      useSettingsStore.getState().setColorBy(value);
      useSelectionStore.getState().setHighlight(null);
    } else if (value) {
      // find the obs column whose categories include this value (e.g. cluster name)
      const col = ds?.obs_columns?.find((c) => (c.values ?? []).includes(value))?.name;
      if (col) {
        useSettingsStore.getState().setColorBy(col);
        useSelectionStore.getState().setHighlight({ column: col, value });
      }
    }
    useViewStore.getState().setPanel("unified");
    useViewStore.getState().setCopilotOpen(false);
  }, []);

  const suggestions = datasetId ? SUGGESTIONS : SUGGESTIONS_NO_DATASET;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
        {turns.length === 0 && (
          <div className="mx-auto max-w-xl pt-6 text-center">
            <Sparkles className="mx-auto mb-3 h-8 w-8 text-primary/60" />
            <p className="mb-4 text-sm text-muted-foreground">
              {datasetId
                ? "Ask about your clusters, markers, the steps that were run, or single-cell analysis in general. Answers are grounded in this dataset (and the methods/literature corpora) and cite their sources."
                : "New here? Ask how to load your data, what formats are supported, or what scView can do — I can help you get started."}
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {suggestions.map((s) => (
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
              {t.role === "user" ? <User className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
            </div>
            <div className="min-w-0 flex-1">
              {t.role === "assistant" ? (
                t.content ? (
                  <MarkdownLite text={t.content} onCitation={handleCitation} />
                ) : (
                  <div className="flex items-center gap-2 py-1 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Thinking…
                  </div>
                )
              ) : (
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{t.content}</div>
              )}
              {t.role === "assistant" && t.grounded === false && (
                <div className="mt-1 text-xs italic text-amber-600">
                  Language model not configured — showing grounded facts directly.
                </div>
              )}
              {t.role === "assistant" && t.route && t.route.length > 0 && (
                <div className="mt-1 flex flex-wrap items-center gap-1 text-[10px] text-muted-foreground">
                  <span>via</span>
                  {t.route.map((r) => (
                    <span key={r} className="rounded bg-muted px-1.5 py-0.5">
                      {ROUTE_LABELS[r] ?? r}
                    </span>
                  ))}
                </div>
              )}
              {t.sources && t.sources.length > 0 && (
                <details className="mt-2 text-xs">
                  <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                    {t.sources.length} grounding source{t.sources.length === 1 ? "" : "s"}
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
              {t.role === "assistant" &&
                i === turns.length - 1 &&
                !busy &&
                t.followups &&
                t.followups.length > 0 && (
                  <div className="mt-2 flex flex-col items-start gap-1">
                    {t.followups.map((q) => (
                      <button
                        key={q}
                        onClick={() => send(q)}
                        className="rounded-full border px-2.5 py-1 text-left text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}
            </div>
          </div>
        ))}
      </div>

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
