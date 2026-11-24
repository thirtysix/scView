import type { UploadProgress } from "@/lib/uploadProgress";
import { BACKEND_DOWN_MESSAGE } from "./client";

// --- Uploads with byte progress (fetch has none): one XMLHttpRequest per request ---

/** Bytes of this request sent so far, of its total (the multipart body counts its headers too). */
export type RequestProgress = (loaded: number, total: number) => void;

/** Progress over a whole upload, and the signal that cancels it. */
export interface UploadOptions {
  onProgress?: (p: UploadProgress) => void;
  signal?: AbortSignal;
}

const HTML_BODY = /^\s*<(!doctype|html)/i;

/** The detail sentence of an error body, or the body itself. An nginx error page (HTML) becomes a
 *  sentence: the backend is down (502/503/504), or the file is over a proxy's size limit (413). */
export function uploadErrorText(status: number, body: string): string {
  const page = !body.trim() || HTML_BODY.test(body);
  if ([502, 503, 504].includes(status) && page) return BACKEND_DOWN_MESSAGE;
  if (status === 413 && page) return "The file is larger than this server accepts (HTTP 413).";
  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  } catch {
    /* not JSON */
  }
  return body || `The upload failed (HTTP ${status}).`;
}

/**
 * POST a body (a FormData, or one File as the raw body) with upload progress. The browser reads
 * files from disk as it sends them. Rejects with the server's sentence on an error status, with a
 * plain sentence when the connection breaks, and with an AbortError when `signal` aborts.
 */
export function xhrUpload<T>(
  url: string,
  body: FormData | File,
  opts: { onProgress?: RequestProgress; signal?: AbortSignal; what?: string; contentType?: string } = {},
): Promise<T> {
  const { onProgress, signal, what = "the files", contentType } = opts;
  return new Promise<T>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const xhr = new XMLHttpRequest();
    const onAbort = () => xhr.abort();
    const done = () => signal?.removeEventListener("abort", onAbort);
    xhr.open("POST", url);
    if (contentType) xhr.setRequestHeader("Content-Type", contentType);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      done();
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch {
          reject(new Error(`The server's answer to the upload of ${what} could not be read.`));
        }
      } else reject(new Error(uploadErrorText(xhr.status, xhr.responseText)));
    };
    xhr.onerror = () => {
      done();
      reject(new Error(`The upload of ${what} stopped before the end: check the connection and send it again.`));
    };
    xhr.onabort = () => {
      done();
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort);
    xhr.send(body);
  });
}

/** Whether an error is the abort of a cancelled upload (not a failure to show). */
export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}
