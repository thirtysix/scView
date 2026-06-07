import { apiFetch } from "./client";

export interface ChatSource {
  kind: string; // "dataset" | "preprocessing" | "provenance" | "result"
  ref: string;
  detail: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  grounded: boolean;
  raw_response?: string;
}

export async function assistantChat(
  datasetId: string,
  query: string,
  history: ChatMessage[] = []
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>(`/datasets/${datasetId}/assistant/chat`, {
    method: "POST",
    body: JSON.stringify({ query, history }),
  });
}
