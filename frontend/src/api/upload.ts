import { fileAt, type UploadProgress } from "@/lib/uploadProgress";
import { BACKEND_DOWN_MESSAGE, LargeDatasetRefusal, largeRefusalMessage } from "./client";

// --- Uploads with byte progress (fetch has none): one XMLHttpRequest per request ---

/** Bytes of this request sent so far, of its total (the multipart body counts its headers too). */
export type RequestProgress = (loaded: number, total: number) => void;

/** Progress over a whole upload, and the signal that cancels it. */
export interface UploadOptions {
  onProgress?: (p: UploadProgress) => void;
  signal?: AbortSignal;
}

const HTML_BODY = /^\s*<(!doctype|html)/i;

/** An error status from an upload: the message is the sentence to show (`uploadErrorText`),
 *  `status` and `body` are the answer as it came. */
export class UploadHttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: string,
  ) {
    super(message);
    this.name = "UploadHttpError";
  }

  /** The body as `readError` gives it: a proxy's HTML error page (or none) as the sentence. */
  get text(): string {
    return !this.body.trim() || HTML_BODY.test(this.body) ? this.message : this.body;
  }
}

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
      } else reject(new UploadHttpError(uploadErrorText(xhr.status, xhr.responseText), xhr.status, xhr.responseText));
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

/** `xhrUpload` with apiFetch's errors: a large-dataset refusal, else "API error <status>: <body>"
 *  (so `errorMessage` and `errorStatus` read it as they read an apiFetch error). A proxy's HTML
 *  error page becomes its sentence. */
export async function apiUpload<T>(
  url: string,
  body: FormData | File,
  opts: { onProgress?: RequestProgress; signal?: AbortSignal; what?: string; contentType?: string } = {},
): Promise<T> {
  try {
    return await xhrUpload<T>(url, body, opts);
  } catch (e) {
    if (!(e instanceof UploadHttpError)) throw e;
    const refusal = largeRefusalMessage(e.status, e.body);
    if (refusal) throw new LargeDatasetRefusal(refusal);
    throw new Error(`API error ${e.status}: ${e.text}`);
  }
}

/** Bytes of the request as bytes of its files: a multipart body is a little larger than its
 *  files, so the share of the request sent is taken as the share of the files sent. */
export function filesProgress(files: File[], onProgress?: (p: UploadProgress) => void): RequestProgress | undefined {
  if (!onProgress) return undefined;
  const sizes = files.map((f) => f.size);
  const all = sizes.reduce((a, b) => a + b, 0);
  return (loaded, total) => {
    const sent = total > 0 ? Math.min(all, (loaded / total) * all) : 0;
    const i = fileAt(sizes, sent);
    onProgress({ loaded: sent, total: all, file: files[i]?.name ?? "", fileIndex: i + 1, fileCount: files.length });
  };
}

/** What an upload of these files is called in messages: the name, or "3 files". */
export function filesWhat(files: File[]): string {
  return files.length === 1 ? files[0]!.name : `${files.length} files`;
}

/** Files sent one request each, in order, with progress over the whole set. Each answer goes to
 *  `each` as it comes; an abort or an error stops the rest. */
export async function uploadInTurn<T>(
  files: File[],
  opts: UploadOptions,
  send: (file: File, opts: UploadOptions) => Promise<T>,
  each?: (answer: T, file: File) => void,
): Promise<T[]> {
  const all = files.reduce((a, f) => a + f.size, 0);
  const out: T[] = [];
  let before = 0;
  for (const [i, file] of files.entries()) {
    const at = before;
    const answer = await send(file, {
      signal: opts.signal,
      onProgress:
        opts.onProgress &&
        ((p) =>
          opts.onProgress!({ loaded: at + p.loaded, total: all, file: file.name, fileIndex: i + 1, fileCount: files.length })),
    });
    out.push(answer);
    each?.(answer, file);
    before += file.size;
  }
  return out;
}
