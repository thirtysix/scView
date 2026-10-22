import { describe, it, expect, vi, afterEach } from "vitest";
import { filePath, UPLOAD_BATCH, uploadIngestFiles } from "./ingest";

describe("ingest api", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends one path per file, in order, so same-named files in two folders stay apart", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ session_id: "s" }), { status: 200 }));
    vi.stubGlobal("fetch", f);
    const a = new File(["filtered"], "barcodes.tsv.gz");
    const b = new File(["raw"], "barcodes.tsv.gz");
    await uploadIngestFiles("s1", [
      { file: a, path: "outs/filtered_feature_bc_matrix/barcodes.tsv.gz" },
      { file: b, path: "outs/raw_feature_bc_matrix/barcodes.tsv.gz" },
    ]);
    const [url, init] = f.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toContain("/ingest/session/s1/files");
    const form = init.body as FormData;
    expect(form.getAll("files")).toHaveLength(2);
    expect(form.getAll("paths")).toEqual([
      "outs/filtered_feature_bc_matrix/barcodes.tsv.gz",
      "outs/raw_feature_bc_matrix/barcodes.tsv.gz",
    ]);
  });

  it("sends a large folder in batches the server's parser accepts", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ session_id: "s" }), { status: 200 }));
    vi.stubGlobal("fetch", f);
    const items = Array.from({ length: UPLOAD_BATCH * 2 + 1 }, (_, i) => ({ file: new File(["x"], `f${i}`), path: `d/f${i}` }));
    await uploadIngestFiles("s1", items);
    const sizes = f.mock.calls.map((c) => ((c as unknown as [string, RequestInit])[1].body as FormData).getAll("paths").length);
    expect(sizes).toEqual([UPLOAD_BATCH, UPLOAD_BATCH, 1]);
  });

  it("surfaces the server's sentence for a refused upload", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(JSON.stringify({ detail: "A different file is already staged as s/barcodes.tsv." }), { status: 409 })));
    const send = () => uploadIngestFiles("s1", [{ file: new File(["x"], "b"), path: "s/barcodes.tsv" }]);
    await expect(send()).rejects.toThrow(/^A different file is already staged as s\/barcodes.tsv.$/);
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>bad gateway</html>", { status: 502 })));
    await expect(send()).rejects.toThrow(/backend did not answer/);
  });

  it("names a file by its folder path when it has one", () => {
    expect(filePath({ name: "barcodes.tsv.gz", rel_path: "o/raw_feature_bc_matrix/barcodes.tsv.gz" })).toBe(
      "o/raw_feature_bc_matrix/barcodes.tsv.gz",
    );
    expect(filePath({ name: "a.h5ad" })).toBe("a.h5ad");
  });
});
