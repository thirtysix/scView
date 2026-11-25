import { afterEach, describe, expect, it, vi } from "vitest";
import { installFakeXhr } from "@/test/fakeXhr";
import type { UploadProgress } from "@/lib/uploadProgress";
import { BACKEND_DOWN_MESSAGE } from "./client";
import { uploadDataset } from "./datasets";
import { isAbort, uploadErrorText, xhrUpload } from "./upload";

describe("uploadErrorText", () => {
  it("gives the server's sentence, or a plain one for a proxy's error page", () => {
    expect(uploadErrorText(409, JSON.stringify({ detail: "Already staged." }))).toBe("Already staged.");
    expect(uploadErrorText(413, JSON.stringify({ detail: "File exceeds maximum upload size of 2000 MB." }))).toBe(
      "File exceeds maximum upload size of 2000 MB.",
    );
    expect(uploadErrorText(413, "<html><body>413 Request Entity Too Large</body></html>")).toBe(
      "The file is larger than this server accepts (HTTP 413).",
    );
    expect(uploadErrorText(502, "<!DOCTYPE html><h1>Bad gateway</h1>")).toBe(BACKEND_DOWN_MESSAGE);
    expect(uploadErrorText(504, "")).toBe(BACKEND_DOWN_MESSAGE);
    expect(uploadErrorText(500, "Internal Server Error")).toBe("Internal Server Error");
    expect(uploadErrorText(500, "")).toBe("The upload failed (HTTP 500).");
  });
});

describe("xhrUpload", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("passes progress on and resolves with the answer", async () => {
    const xhr = installFakeXhr((x) => {
      x.progress(10, 100);
      x.respond(201, { id: "d1" });
    });
    const seen: [number, number][] = [];
    const out = await xhrUpload<{ id: string }>("/api/v1/x", new FormData(), { onProgress: (l, t) => seen.push([l, t]) });
    expect(out).toEqual({ id: "d1" });
    expect(seen).toEqual([[10, 100]]);
    expect(xhr.requests[0]!.method).toBe("POST");
  });

  it("rejects with a plain sentence when the connection breaks", async () => {
    installFakeXhr((x) => x.fail());
    await expect(xhrUpload("/x", new FormData(), { what: "sample.rds" })).rejects.toThrow(
      "The upload of sample.rds stopped before the end: check the connection and send it again.",
    );
  });

  it("rejects with the error page as a sentence (413, 502)", async () => {
    installFakeXhr((x) => x.respond(413, "<html>too large</html>"));
    await expect(xhrUpload("/x", new FormData())).rejects.toThrow("larger than this server accepts");
    installFakeXhr((x) => x.respond(502, "<html>bad gateway</html>"));
    await expect(xhrUpload("/x", new FormData())).rejects.toThrow(BACKEND_DOWN_MESSAGE);
  });

  it("aborts the request on the signal, and never sends with a signal already aborted", async () => {
    const xhr = installFakeXhr();
    const abort = new AbortController();
    const sent = xhrUpload("/x", new FormData(), { signal: abort.signal });
    abort.abort();
    const err = await sent.catch((e: unknown) => e);
    expect(isAbort(err)).toBe(true);
    expect(xhr.requests[0]!.aborted).toBe(true);

    const again = await xhrUpload("/x", new FormData(), { signal: abort.signal }).catch((e: unknown) => e);
    expect(isAbort(again)).toBe(true);
    expect(xhr.requests).toHaveLength(1);
  });

  it("says so when a success answer is not JSON", async () => {
    installFakeXhr((x) => x.respond(200, "<html>login</html>"));
    await expect(xhrUpload("/x", new FormData(), { what: "sample.rds" })).rejects.toThrow(
      "The server's answer to the upload of sample.rds could not be read.",
    );
  });
});

describe("uploadDataset", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends the file as form field 'file' and reports bytes of the file, not of the request", async () => {
    const xhr = installFakeXhr((x) => {
      x.progress(550, 1100); // the multipart body is a little larger than the file
      x.progress(1100, 1100);
      x.respond(201, { id: "d1", name: "sample", status: "converting" });
    });
    const file = new File(["x".repeat(1000)], "sample.rds");
    const seen: UploadProgress[] = [];
    const out = await uploadDataset(file, { onProgress: (p) => seen.push(p) });
    expect(out.status).toBe("converting");
    expect(xhr.requests[0]!.url).toBe("/api/v1/datasets/upload");
    expect((xhr.requests[0]!.body as FormData).get("file")).toBeInstanceOf(File);
    expect(seen).toEqual([
      { loaded: 500, total: 1000, file: "sample.rds", fileIndex: 1, fileCount: 1 },
      { loaded: 1000, total: 1000, file: "sample.rds", fileIndex: 1, fileCount: 1 },
    ]);
  });
});
