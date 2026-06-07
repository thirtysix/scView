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

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  grounded: boolean;
  raw_response?: string;
  route?: string[]; // knowledge sources consulted: app/data/tutorials/literature
  followups?: string[]; // suggested next questions
}

export async function assistantChat(
  datasetId: string,
  query: string,
  history: ChatMessage[] = [],
  viewContext?: ViewContext
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>(`/datasets/${datasetId}/assistant/chat`, {
    method: "POST",
    body: JSON.stringify({ query, history, view_context: viewContext ?? null }),
  });
}

export interface StreamHandlers {
  onSources?: (ev: { sources: ChatSource[]; route: string[]; grounded: boolean }) => void;
  onDelta?: (text: string) => void;
  onDone?: (ev: { followups: string[] }) => void;
  onError?: (msg: string) => void;
}

/** Stream a chat answer via SSE. Resolves when the stream completes; throws on a
 *  transport/HTTP error so the caller can fall back to the non-streaming call. */
export async function assistantChatStream(
  datasetId: string,
  query: string,
  history: ChatMessage[],
  viewContext: ViewContext | undefined,
  handlers: StreamHandlers
): Promise<void> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/assistant/chat-stream`, {
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
        handlers.onDone?.({ followups: (ev.followups as string[]) ?? [] });
      } else if (ev.type === "error") {
        handlers.onError?.(String(ev.error ?? "stream error"));
      }
    }
  }
}
