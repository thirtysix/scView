import { API_BASE } from "@/lib/constants";

/** Shown instead of the HTML page nginx sends when the backend is down or restarting. */
export const BACKEND_DOWN_MESSAGE =
  "scView's backend did not answer: it is restarting or has stopped. If it was working on a " +
  "large dataset it may have run out of memory. Wait a minute and try again.";

/** An error response's body to show: the backend's own text, or a plain sentence for nginx's error page. */
export async function readError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  if ([502, 503, 504].includes(res.status) && (!text.trim() || /^\s*<(!doctype|html)/i.test(text)))
    return BACKEND_DOWN_MESSAGE;
  return text;
}

export async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const error = await readError(res);
    throw new Error(`API error ${res.status}: ${error}`);
  }
  // No-content responses (e.g. 204 from DELETE) have no body to parse.
  if (res.status === 204 || res.headers.get("content-length") === "0") {
    return undefined as T;
  }
  return res.json();
}

export async function apiFetchBinary(
  path: string,
  options?: RequestInit
): Promise<ArrayBuffer> {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const error = await readError(res);
    throw new Error(`API error ${res.status}: ${error}`);
  }
  return res.arrayBuffer();
}

export async function apiUpload(
  path: string,
  file: File
): Promise<{ id: string; name: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const error = await readError(res);
    throw new Error(`Upload error ${res.status}: ${error}`);
  }
  return res.json();
}
