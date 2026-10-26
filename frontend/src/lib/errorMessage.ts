/** The server's reason from an apiFetch error ("API error 400: {\"detail\": ...}") as plain text. */
export function errorMessage(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e);
  const body = raw.replace(/^API error \d+: /, "");
  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail != null) return JSON.stringify(detail);
  } catch {
    // not JSON: the text itself
  }
  return body;
}
