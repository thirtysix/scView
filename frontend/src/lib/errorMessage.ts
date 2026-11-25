/** The thrown value's own text: an Error's message, else the value as a string. */
function rawText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** The server's reason from an apiFetch error ("API error 400: {\"detail\": ...}") as plain text.
 * A 422 validation list gives its messages. Any other error gives its own message. */
export function errorMessage(e: unknown): string {
  const body = rawText(e).replace(/^API error \d+: /, "");
  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0 && detail.every((d) => typeof d?.msg === "string")) {
      return detail.map((d: { msg: string }) => d.msg.replace(/^Value error, /, "")).join("; ");
    }
    if (detail != null) return JSON.stringify(detail);
  } catch {
    // not JSON: the text itself
  }
  return body;
}

/** The HTTP status of an apiFetch error ("API error 404: ..."), else null. */
export function errorStatus(e: unknown): number | null {
  const m = /^API error (\d+):/.exec(rawText(e));
  return m ? Number(m[1]) : null;
}

/** Statuses a query never asks again for: 507 is the backend refusing work that does not fit
 * in its memory, and asking again at once only queues the same refusal (a retry storm on a
 * page that fires a dozen requests). */
const NO_RETRY = new Set([507]);

/** React Query's `retry`: once, except for a refusal asking again cannot change. */
export function retryQuery(failureCount: number, error: unknown): boolean {
  const status = errorStatus(error);
  return failureCount < 1 && (status === null || !NO_RETRY.has(status));
}
