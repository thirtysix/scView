import { apiFetch } from "./client";
import { API_BASE } from "@/lib/constants";

export interface ChatSource {
  kind: string; // "dataset" | "preprocessing" | "provenance" | "result" | "literature" | "tutorial"
  ref: string;
  detail: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/** What the user is currently looking at, so deictic questions resolve. */
export interface ViewContext {
  panel?: string;
  color_by?: string;
  highlighted?: { column: string; value: string } | null;
  overlay?: string;
}

/** An allow-listed UI action the co-pilot can request (natural-language commands). */
export interface AssistantAction {
  type:
    | "set_color_by"
    | "highlight_cluster"
    | "open_panel"
    | "set_embedding"
    | "set_subtab"
    | "set_groupby"
    | "clear_highlight"
    | "clear_overlay"
    | "show_gene"
    | "annotate_cell_types"
    | "cluster"
    | "detect_doublets"
    | "compute_markers"
    | "run_enrichment";
  column?: string;
  value?: string;
  panel?: string;
  embedding?: string;
  subtab?: string;
  gene?: string;
  label?: string;
  // Confirm-gated mutating actions:
  requires_confirm?: boolean;
  step?: string;
  params?: Record<string, unknown>;
  advisory?: string;
  estimate?: string;
}

/** Run pipeline step(s) on a dataset (used when the user confirms a mutating action). */
export async function runPipelineSteps(
  datasetId: string,
  steps: string[],
  params: Record<string, unknown>
): Promise<{ steps_run: string[]; errors: Record<string, string> }> {
  return apiFetch(`/datasets/${datasetId}/assessment/run`, {
    method: "POST",
    body: JSON.stringify({ steps, params }),
  });
}

/** A deterministic one-line "I notice…" nudge surfaced when a dataset opens. */
export interface DatasetInsight {
  insight: string;
  question?: string | null;
  severity?: "info" | "suggestion";
}

export async function getInsight(datasetId: string): Promise<DatasetInsight> {
  return apiFetch<DatasetInsight>(`/datasets/${datasetId}/assistant/insight`);
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  grounded: boolean;
  raw_response?: string;
  route?: string[]; // knowledge sources consulted: app/data/tutorials/literature
  followups?: string[]; // suggested next questions
  actions?: AssistantAction[]; // UI actions to execute
}

// With no dataset loaded, hit the app-level endpoint so the co-pilot can still
// help a newcomer (what scView does, how to load data).
const chatPath = (datasetId: string | null, stream: boolean) =>
  (datasetId ? `/datasets/${datasetId}/assistant` : `/assistant`) +
  (stream ? "/chat-stream" : "/chat");

export async function assistantChat(
  datasetId: string | null,
  query: string,
  history: ChatMessage[] = [],
  viewContext?: ViewContext
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>(chatPath(datasetId, false), {
    method: "POST",
    body: JSON.stringify({ query, history, view_context: viewContext ?? null }),
  });
}

export interface StreamHandlers {
  onSources?: (ev: { sources: ChatSource[]; route: string[]; grounded: boolean }) => void;
  onDelta?: (text: string) => void;
  onDone?: (ev: { followups: string[]; actions: AssistantAction[] }) => void;
  onError?: (msg: string) => void;
}

/** Stream a chat answer via SSE. Resolves when the stream completes; throws on a
 *  transport/HTTP error so the caller can fall back to the non-streaming call. */
export async function assistantChatStream(
  datasetId: string | null,
  query: string,
  history: ChatMessage[],
  viewContext: ViewContext | undefined,
  handlers: StreamHandlers
): Promise<void> {
  const res = await fetch(`${API_BASE}${chatPath(datasetId, true)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history, view_context: viewContext ?? null }),
  });
  if (!res.ok || !res.body) throw new Error(`API error ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      const line = raw.startsWith("data:") ? raw.slice(5).trim() : raw;
      if (!line) continue;
      let ev: { type: string; [k: string]: unknown };
      try {
        ev = JSON.parse(line);
      } catch {
        continue;
      }
      if (ev.type === "sources") {
        handlers.onSources?.(ev as never);
      } else if (ev.type === "delta") {
        handlers.onDelta?.(String(ev.text ?? ""));
      } else if (ev.type === "done") {
        handlers.onDone?.({
          followups: (ev.followups as string[]) ?? [],
          actions: (ev.actions as AssistantAction[]) ?? [],
        });
      } else if (ev.type === "error") {
        handlers.onError?.(String(ev.error ?? "stream error"));
      }
    }
  }
}
