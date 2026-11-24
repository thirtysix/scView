import { vi } from "vitest";

/** A stand-in XMLHttpRequest for upload tests: records each request and lets the test send
 *  progress, an answer, a network error or nothing. */
export class FakeXhr {
  static requests: FakeXhr[] = [];
  /** Called on send: answer here for a request that should finish by itself. */
  static onSend: ((x: FakeXhr) => void) | null = null;

  method = "";
  url = "";
  headers: Record<string, string> = {};
  body: unknown = null;
  status = 0;
  responseText = "";
  aborted = false;
  upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = {
    onprogress: null,
  };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(k: string, v: string) {
    this.headers[k] = v;
  }
  send(body: unknown) {
    this.body = body;
    FakeXhr.requests.push(this);
    FakeXhr.onSend?.(this);
  }
  abort() {
    this.aborted = true;
    this.onabort?.();
  }

  progress(loaded: number, total: number) {
    this.upload.onprogress?.({ lengthComputable: true, loaded, total });
  }
  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = typeof body === "string" ? body : JSON.stringify(body);
    this.onload?.();
  }
  fail() {
    this.onerror?.();
  }
}

/** Put FakeXhr in place of XMLHttpRequest (undo with vi.unstubAllGlobals()). */
export function installFakeXhr(onSend: ((x: FakeXhr) => void) | null = null): typeof FakeXhr {
  FakeXhr.requests = [];
  FakeXhr.onSend = onSend;
  vi.stubGlobal("XMLHttpRequest", FakeXhr);
  return FakeXhr;
}
