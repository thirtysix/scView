import { apiFetch } from "./client";
import { errorMessage } from "@/lib/errorMessage";

/** The server's reason from an "API error N: {detail}" message, else the message. */
function reason(e: unknown): Error {
  return new Error(errorMessage(e));
}

/**
 * One obs column's value per cell, in cell order, as text (missing: null):
 * `GET /datasets/{id}/metadata/{column}?format=json`. The gates' CSV counts
 * cells per sample with it.
 */
export async function obsColumnValues(datasetId: string, column: string): Promise<(string | null)[]> {
  try {
    const res = await apiFetch<{ column: string; values: unknown[] }>(
      `/datasets/${datasetId}/metadata/${encodeURIComponent(column)}?format=json`,
    );
    return (res.values ?? []).map((v) => (v == null ? null : String(v)));
  } catch (e) {
    throw reason(e);
  }
}
