import { useRef, useState, useEffect, useCallback } from "react";
import { Sparkles, Send, User, Loader2, AlertTriangle, Clock, Check, X, Trash2 } from "lucide-react";
import { getDataset } from "@/api/datasets";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { PANEL_LABELS, type PanelId } from "@/lib/constants";
import {
  assistantChat,
  assistantChatStream,
  runPipelineSteps,
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
  pendingAction?: AssistantAction; // a confirm-gated mutating action awaiting the user
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
// Per-dataset conversation threads survive reloads + dataset switches.
const threadKey = (datasetId: string | null) => `scview.chat.v1.${datasetId ?? "__app__"}`;

export function AssistantChat() {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns, busy]);

  // Load this dataset's saved thread when the dataset changes (and on mount).
  const key = threadKey(datasetId);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      setTurns(raw ? (JSON.parse(raw) as Turn[]) : []);
    } catch {
      setTurns([]);
    }
  }, [key]);

  // Persist the thread once idle (avoid hammering storage during streaming).
  useEffect(() => {
    if (busy) return;
    try {
      if (turns.length) localStorage.setItem(key, JSON.stringify(turns));
      else localStorage.removeItem(key);
    } catch {
      /* storage full / unavailable — non-fatal */
    }
  }, [turns, busy, key]);

  const clearChat = useCallback(() => {
    setTurns([]);
    try {
      localStorage.removeItem(key);
    } catch {
      /* non-fatal */
    }
  }, [key]);

  // An "ask about this" affordance elsewhere in the UI queues a question here.
  const pendingAsk = useViewStore((s) => s.pendingAsk);
  const clearPendingAsk = useViewStore((s) => s.clearPendingAsk);
  useEffect(() => {
    if (!pendingAsk) return;
    clearPendingAsk();
    send(pendingAsk);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingAsk]);

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
      if (a.requires_confirm) {
        // Mutating action: don't run it — surface a Confirm card on the turn.
        setTurns((prev) =>
          prev.map((t, i) => (i === prev.length - 1 ? { ...t, pendingAction: a } : t)),
        );
        continue;
      }
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
      } else if (a.type === "set_embedding" && a.embedding) {
        useSettingsStore.getState().setEmbedding(a.embedding);
        useViewStore.getState().setPanel("unified");
      } else if (a.type === "set_subtab" && a.subtab) {
        useUnifiedViewStore
          .getState()
          .setActiveSubtab(a.subtab as "markers" | "expression" | "genesets" | "enrichment");
        useViewStore.getState().setPanel("unified");
      } else if (a.type === "set_groupby" && a.column) {
        useUnifiedViewStore.getState().setGroupByColumn(a.column);
        useViewStore.getState().setPanel("unified");
      } else if (a.type === "clear_highlight") {
        useSelectionStore.getState().setHighlight(null);
      } else if (a.type === "clear_overlay") {
        useUnifiedViewStore.getState().clearOverlayState();
      } else if (a.type === "show_gene" && a.gene) {
        useViewStore.getState().setPendingGene(a.gene);
        useViewStore.getState().setPanel("expression");
      }
    }
  }, []);

  // Run a confirm-gated mutating action (after the user clicks Confirm).
  const runPendingAction = useCallback(
    async (action: AssistantAction) => {
      if (!datasetId || !action.step) return;
      setConfirming(true);
      setTurns((prev) =>
        prev.map((t, i) =>
          i === prev.length - 1
            ? { ...t, pendingAction: undefined, content: t.content + "\n\n_Running…_" }
            : t,
        ),
      );
      try {
        const res = await runPipelineSteps(datasetId, [action.step], action.params ?? {});
        const updated = await getDataset(datasetId);
        useDatasetStore.getState().setCurrentDataset(updated);
        const err =
          res.errors && Object.keys(res.errors).length ? Object.values(res.errors)[0] : "";
        setTurns((prev) =>
          prev.map((t, i) =>
            i === prev.length - 1
              ? { ...t, content: t.content.replace("_Running…_", err ? `❌ ${err}` : "✓ Done.") }
              : t,
          ),
        );
        const tgt = action.params?.annotation_target as string | undefined;
        if (!err && action.step === "cell_type_annotation" && tgt) {
          useSettingsStore.getState().setColorBy(tgt);
          useViewStore.getState().setPanel("unified");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setTurns((prev) =>
          prev.map((t, i) =>
            i === prev.length - 1 ? { ...t, content: t.content.replace("_Running…_", `❌ ${msg}`) } : t,
          ),
        );
      } finally {
        setConfirming(false);
      }
    },
    [datasetId],
  );

  const cancelPendingAction = useCallback(() => {
    setTurns((prev) =>
      prev.map((t, i) => (i === prev.length - 1 ? { ...t, pendingAction: undefined } : t)),
    );
  }, []);

  // Clicking a result citation chip jumps to that cluster/gene in the Unified View.
  const handleCitation = useCallback((tag: string) => {
    const ds = useDatasetStore.getState().currentDataset;
    // 1) jump to the longest categorical group value mentioned anywhere in the tag —
    //    tolerant of the model reformatting the citation (e.g. "...cluster='NK'").
    const tokenHit = (v: string) => {
      const i = tag.indexOf(v);
      if (i < 0) return false;
      const bnd = (ch?: string) => !ch || !/[A-Za-z0-9]/.test(ch);
      return bnd(tag[i - 1]) && bnd(tag[i + v.length]);
    };
    let best: { col: string; val: string } | null = null;
    for (const c of ds?.obs_columns ?? []) {
      for (const v of c.values ?? []) {
        if (tokenHit(v) && (!best || v.length > best.val.length)) best = { col: c.name, val: v };
      }
    }
    const parts = tag.split(":");
    if (best) {
      useSettingsStore.getState().setColorBy(best.col);
      useSelectionStore.getState().setHighlight({ column: best.col, value: best.val });
    } else if (parts[0] === "result" && parts[1] === "groups" && parts[2]) {
      useSettingsStore.getState().setColorBy(parts.slice(2).join(":"));
      useSelectionStore.getState().setHighlight(null);
    } else {
      return; // nothing resolvable — leave the view untouched
    }
    useViewStore.getState().setPanel("unified");
    useViewStore.getState().setCopilotOpen(false);
  }, []);

  const suggestions = datasetId ? SUGGESTIONS : SUGGESTIONS_NO_DATASET;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {turns.length > 0 && (
        <div className="flex justify-end border-b px-3 py-1.5">
          <button
            onClick={clearChat}
            disabled={busy}
            title="Clear this conversation"
            className="flex items-center gap-1 rounded px-1.5 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-40"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
        </div>
      )}
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
              {t.role === "assistant" && t.pendingAction && (
                <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs">
                  <div className="mb-1.5 font-medium text-amber-900">{t.pendingAction.label}</div>
                  {t.pendingAction.advisory && (
                    <div className="mb-1 flex items-start gap-1.5 text-amber-800">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                      <span>{t.pendingAction.advisory}</span>
                    </div>
                  )}
                  {t.pendingAction.estimate && (
                    <div className="mb-2 flex items-center gap-1.5 text-amber-700">
                      <Clock className="h-3.5 w-3.5" />
                      <span>Estimated time: {t.pendingAction.estimate}</span>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button
                      type="button"
                      disabled={confirming}
                      onClick={() => runPendingAction(t.pendingAction!)}
                      className="inline-flex items-center gap-1 rounded-md bg-primary px-3 py-1 font-medium text-white hover:bg-primary/90 disabled:opacity-50"
                    >
                      {confirming ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Check className="h-3.5 w-3.5" />
                      )}
                      Confirm &amp; run
                    </button>
                    <button
                      type="button"
                      disabled={confirming}
                      onClick={cancelPendingAction}
                      className="inline-flex items-center gap-1 rounded-md border border-slate-300 px-3 py-1 font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                    >
                      <X className="h-3.5 w-3.5" />
                      Cancel
                    </button>
                  </div>
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
