import { apiFetch } from "./client";

/** The server's reason from an "API error N: {detail}" message, else the message. */
function reason(e: unknown): Error {
  const raw = e instanceof Error ? e.message : String(e);
  const body = raw.replace(/^API error \d+: /, "");
  try {
    const detail = JSON.parse(body).detail;
    if (typeof detail === "string") return new Error(detail);
  } catch {
    /* not JSON */
  }
  return new Error(body || raw);
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
