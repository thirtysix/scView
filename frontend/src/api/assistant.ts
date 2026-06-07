import { apiFetch } from "./client";

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
