import { describe, it, expect, vi, afterEach } from "vitest";
import { installFakeXhr } from "@/test/fakeXhr";
import type { UploadProgress } from "@/lib/uploadProgress";
import { filePath, UPLOAD_BATCH, uploadIngestFiles } from "./ingest";

describe("ingest api", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends one path per file, in order, so same-named files in two folders stay apart", async () => {
    const xhr = installFakeXhr((x) => x.respond(200, { session_id: "s" }));
    const a = new File(["filtered"], "barcodes.tsv.gz");
    const b = new File(["raw"], "barcodes.tsv.gz");
    await uploadIngestFiles("s1", [
      { file: a, path: "outs/filtered_feature_bc_matrix/barcodes.tsv.gz" },
      { file: b, path: "outs/raw_feature_bc_matrix/barcodes.tsv.gz" },
    ]);
    const req = xhr.requests[0]!;
    expect(req.method).toBe("POST");
    expect(req.url).toContain("/ingest/session/s1/files");
    const form = req.body as FormData;
    expect(form.getAll("files")).toHaveLength(2);
    expect(form.getAll("paths")).toEqual([
      "outs/filtered_feature_bc_matrix/barcodes.tsv.gz",
      "outs/raw_feature_bc_matrix/barcodes.tsv.gz",
    ]);
  });

  it("sends a large folder in batches the server's parser accepts", async () => {
    const xhr = installFakeXhr((x) => x.respond(200, { session_id: "s" }));
    const items = Array.from({ length: UPLOAD_BATCH * 2 + 1 }, (_, i) => ({ file: new File(["x"], `f${i}`), path: `d/f${i}` }));
    await uploadIngestFiles("s1", items);
    const sizes = xhr.requests.map((r) => (r.body as FormData).getAll("paths").length);
    expect(sizes).toEqual([UPLOAD_BATCH, UPLOAD_BATCH, 1]);
  });

  it("reports progress over the whole set, across batches, with the file being sent", async () => {
    const seen: UploadProgress[] = [];
    // each batch: half its request body, then all of it, then the answer
    installFakeXhr((x) => {
      x.progress(500, 1000);
      x.progress(1000, 1000);
      x.respond(200, { session_id: "s" });
    });
    const items = Array.from({ length: UPLOAD_BATCH + 100 }, (_, i) => ({ file: new File(["xx"], `f${i}`), path: `d/f${i}` }));
    await uploadIngestFiles("s1", items, { onProgress: (p) => seen.push(p) });
    const total = (UPLOAD_BATCH + 100) * 2;
    expect(seen.every((p) => p.total === total && p.fileCount === UPLOAD_BATCH + 100)).toBe(true);
    // half of batch 1 (400 files of 2 bytes): 400 bytes, the file at byte 400 is the 201st
    expect(seen[0]).toMatchObject({ loaded: UPLOAD_BATCH, fileIndex: UPLOAD_BATCH / 2 + 1, file: `f${UPLOAD_BATCH / 2}` });
    // never goes backwards, and ends at every byte on the last file
    expect(seen.map((p) => p.loaded)).toEqual([...seen.map((p) => p.loaded)].sort((a, b) => a - b));
    expect(seen.at(-1)).toMatchObject({ loaded: total, fileIndex: UPLOAD_BATCH + 100 });
  });

  it("surfaces the server's sentence for a refused upload", async () => {
    installFakeXhr((x) => x.respond(409, { detail: "A different file is already staged as s/barcodes.tsv." }));
    const send = () => uploadIngestFiles("s1", [{ file: new File(["x"], "b"), path: "s/barcodes.tsv" }]);
    await expect(send()).rejects.toThrow(/^A different file is already staged as s\/barcodes.tsv.$/);
    installFakeXhr((x) => x.respond(502, "<html>bad gateway</html>"));
    await expect(send()).rejects.toThrow(/backend did not answer/);
  });

  it("stops at the batch being sent when cancelled; no later batch goes", async () => {
    const xhr = installFakeXhr((x) => {
      if (xhr.requests.length === 1) x.respond(200, { session_id: "s" });
    });
    const abort = new AbortController();
    const items = Array.from({ length: UPLOAD_BATCH * 3 }, (_, i) => ({ file: new File(["x"], `f${i}`), path: `d/f${i}` }));
    const sent = uploadIngestFiles("s1", items, { signal: abort.signal });
    await vi.waitFor(() => expect(xhr.requests).toHaveLength(2));
    abort.abort();
    await expect(sent).rejects.toMatchObject({ name: "AbortError" });
    expect(xhr.requests[1]!.aborted).toBe(true);
    expect(xhr.requests).toHaveLength(2);
  });

  it("names a file by its folder path when it has one", () => {
    expect(filePath({ name: "barcodes.tsv.gz", rel_path: "o/raw_feature_bc_matrix/barcodes.tsv.gz" })).toBe(
      "o/raw_feature_bc_matrix/barcodes.tsv.gz",
    );
    expect(filePath({ name: "a.h5ad" })).toBe("a.h5ad");
  });
});
