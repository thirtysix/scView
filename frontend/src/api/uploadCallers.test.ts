import { afterEach, describe, expect, it, vi } from "vitest";
import { installFakeXhr } from "@/test/fakeXhr";
import type { UploadProgress } from "@/lib/uploadProgress";
import { errorMessage, errorStatus } from "@/lib/errorMessage";
import { LargeDatasetRefusal } from "./client";
import { attachAtacMetrics } from "./atac";
import { uploadFragmentFile } from "./atacFragments";
import { runDeconvolution } from "./deconvolution";
import { ApiDetailError, importLabelFile } from "./labels";
import { uploadOutlines, uploadTranscripts } from "./transcripts";
import { isAbort, uploadInTurn } from "./upload";
import { attachVdj } from "./vdj";
import { attachVelocity } from "./velocity";

afterEach(() => vi.unstubAllGlobals());

const file = (name: string, size = 1000) => new File(["x".repeat(size)], name);
const LABEL_OPTS = {
  column: "my_labels",
  barcodeColumn: "barcode",
  labelColumn: "label",
  stripSuffix: false,
  minMatch: 0.9,
  dryRun: true,
};

const DECONV_OPTS = { celltype_column: "cell_type", donor_column: null, method: "auto" as const };

/** Every upload goes through one XMLHttpRequest with byte progress (bytes of the file, not of
 *  the multipart body), keeps its AbortSignal, and keeps the error words its callers read. */
describe("upload callers", () => {
  it("multipart callers: URL, form fields, no Content-Type of their own, progress in file bytes", async () => {
    const cases: [string, (o: { onProgress: (p: UploadProgress) => void }) => Promise<unknown>, string][] = [
      ["/api/v1/datasets/d1/atac/metrics", (o) => attachAtacMetrics("d1", file("per_barcode_metrics.csv"), o), "file"],
      ["/api/v1/datasets/d1/deconvolution/upload", (o) => runDeconvolution("d1", file("bulk.tsv"), DECONV_OPTS, o), "file"],
      ["/api/v1/datasets/d1/vdj/upload", (o) => attachVdj("d1", file("contigs.csv"), { sample: "s1" }, o), "file"],
      ["/api/v1/datasets/d1/velocity/attach", (o) => attachVelocity("d1", [file("sample.loom")], o), "files"],
    ];
    for (const [url, send, field] of cases) {
      const xhr = installFakeXhr((x) => {
        x.progress(550, 1100);
        x.respond(200, { ok: true });
      });
      const seen: UploadProgress[] = [];
      await send({ onProgress: (p) => seen.push(p) });
      const req = xhr.requests[0]!;
      expect(req.url).toBe(url);
      expect(req.headers).toEqual({}); // the browser sets the multipart boundary
      expect((req.body as FormData).get(field)).toBeInstanceOf(File);
      expect(seen).toEqual([{ loaded: 500, total: 1000, file: expect.any(String), fileIndex: 1, fileCount: 1 }]);
    }
  });

  it("the ATAC metrics file goes as multipart (it was sent with a JSON Content-Type before)", async () => {
    const xhr = installFakeXhr((x) => x.respond(200, { n_cells: 10, n_matched: 9, file: "m.csv" }));
    await expect(attachAtacMetrics("d1", file("m.csv"))).resolves.toEqual({ n_cells: 10, n_matched: 9, file: "m.csv" });
    expect(xhr.requests[0]!.headers["Content-Type"]).toBeUndefined();
  });

  it("deconvolution without a file sends only the options, with no progress", async () => {
    const xhr = installFakeXhr((x) => {
      x.progress(10, 10);
      x.respond(200, { id: "r1" });
    });
    const onProgress = vi.fn();
    await runDeconvolution("d1", null, DECONV_OPTS, { onProgress });
    expect((xhr.requests[0]!.body as FormData).get("file")).toBeNull();
    expect(JSON.parse((xhr.requests[0]!.body as FormData).get("options") as string)).toEqual(DECONV_OPTS);
    expect(onProgress).not.toHaveBeenCalled();
  });

  it("several files in one request: progress names the file going now", async () => {
    installFakeXhr((x) => {
      x.progress(1500, 2000); // 75% of the body = 1500 of the files' 2000 bytes: the second file
      x.respond(200, { attached: true, source: {} });
    });
    const seen: UploadProgress[] = [];
    await attachVelocity("d1", [file("a.loom"), file("b.loom")], { onProgress: (p) => seen.push(p) });
    expect(seen).toEqual([{ loaded: 1500, total: 2000, file: "b.loom", fileIndex: 2, fileCount: 2 }]);
  });

  it("raw-body callers send the File itself as octet-stream, with the name in the query", async () => {
    const cases: [string, (o: { onProgress: (p: UploadProgress) => void }) => Promise<unknown>][] = [
      ["/api/v1/datasets/d1/atac/fragments?name=atac_fragments.tsv.gz", (o) => uploadFragmentFile("d1", file("atac_fragments.tsv.gz"), o)],
      ["/api/v1/datasets/d1/spatial/transcripts?name=transcripts.parquet", (o) => uploadTranscripts("d1", file("transcripts.parquet"), o)],
      ["/api/v1/datasets/d1/spatial/outlines?name=cell_boundaries.parquet", (o) => uploadOutlines("d1", file("cell_boundaries.parquet"), o)],
    ];
    for (const [url, send] of cases) {
      const xhr = installFakeXhr((x) => {
        x.progress(250, 1000);
        x.respond(200, {});
      });
      const seen: number[] = [];
      await send({ onProgress: (p) => seen.push(p.loaded) });
      expect(xhr.requests[0]!.url).toBe(url);
      expect(xhr.requests[0]!.headers["Content-Type"]).toBe("application/octet-stream");
      expect(xhr.requests[0]!.body).toBeInstanceOf(File);
      expect(seen).toEqual([250]);
    }
  });

  it("every caller rejects with an AbortError when its signal aborts", async () => {
    const sends: ((signal: AbortSignal) => Promise<unknown>)[] = [
      (signal) => attachAtacMetrics("d1", file("m.csv"), { signal }),
      (signal) => runDeconvolution("d1", file("b.tsv"), DECONV_OPTS, { signal }),
      (signal) => attachVdj("d1", file("c.csv"), {}, { signal }),
      (signal) => attachVelocity("d1", [file("a.loom")], { signal }),
      (signal) => importLabelFile("d1", file("l.csv"), LABEL_OPTS, { signal }),
      (signal) => uploadFragmentFile("d1", file("f.tsv.gz"), { signal }),
      (signal) => uploadTranscripts("d1", file("t.parquet"), { signal }),
      (signal) => uploadOutlines("d1", file("cell_boundaries.parquet"), { signal }),
    ];
    for (const send of sends) {
      const xhr = installFakeXhr();
      const ctl = new AbortController();
      const sent = send(ctl.signal);
      ctl.abort();
      expect(isAbort(await sent.catch((e: unknown) => e))).toBe(true);
      expect(xhr.requests[0]!.aborted).toBe(true);
    }
  });

  it("apiFetch-style callers keep 'API error <status>: <body>' and the large-dataset refusal", async () => {
    installFakeXhr((x) => x.respond(422, { detail: "No cell matched." }));
    const err = await attachVdj("d1", file("c.csv")).catch((e: unknown) => e);
    expect((err as Error).message).toBe('API error 422: {"detail":"No cell matched."}');
    expect(errorMessage(err)).toBe("No cell matched.");
    expect(errorStatus(err)).toBe(422);

    installFakeXhr((x) => x.respond(413, "<html><body>413 Request Entity Too Large</body></html>"));
    const big = await attachVelocity("d1", [file("a.loom")]).catch((e: unknown) => e);
    expect(errorMessage(big)).toBe("The file is larger than this server accepts (HTTP 413).");

    installFakeXhr((x) => x.respond(422, { detail: "Not available for large datasets yet.", large_dataset: true }));
    const large = await attachAtacMetrics("d1", file("m.csv")).catch((e: unknown) => e);
    expect(large).toBeInstanceOf(LargeDatasetRefusal);
  });

  it("the label import keeps its ApiDetailError with the status (null when the link breaks)", async () => {
    installFakeXhr((x) => x.respond(422, { detail: "Only 3 of 10 rows name a cell." }));
    const err = (await importLabelFile("d1", file("l.csv"), LABEL_OPTS).catch((e: unknown) => e)) as ApiDetailError;
    expect(err).toBeInstanceOf(ApiDetailError);
    expect(err.message).toBe("Only 3 of 10 rows name a cell.");
    expect(err.status).toBe(422);

    installFakeXhr((x) => x.fail());
    const broken = (await importLabelFile("d1", file("l.csv"), LABEL_OPTS).catch((e: unknown) => e)) as ApiDetailError;
    expect(broken).toBeInstanceOf(ApiDetailError);
    expect(broken.status).toBeNull();
    expect(broken.message).toMatch(/upload of l.csv stopped before the end/);
  });
});

describe("uploadInTurn", () => {
  it("sends the files one after another with progress over the set, and hands on each answer", async () => {
    let n = 0;
    installFakeXhr((x) => {
      x.progress(500, 1000);
      x.respond(200, { n: ++n });
    });
    const seen: UploadProgress[] = [];
    const answers: number[] = [];
    const out = await uploadInTurn(
      [file("atac_fragments.tsv.gz.tbi", 1000), file("atac_fragments.tsv.gz", 3000)],
      { onProgress: (p) => seen.push(p) },
      (f, one) => uploadFragmentFile("d1", f, one),
      (a) => answers.push((a as unknown as { n: number }).n),
    );
    expect(out).toHaveLength(2);
    expect(answers).toEqual([1, 2]);
    expect(seen).toEqual([
      { loaded: 500, total: 4000, file: "atac_fragments.tsv.gz.tbi", fileIndex: 1, fileCount: 2 },
      { loaded: 2500, total: 4000, file: "atac_fragments.tsv.gz", fileIndex: 2, fileCount: 2 },
    ]);
  });

  it("stops at an abort: the next file is never sent", async () => {
    const xhr = installFakeXhr();
    const ctl = new AbortController();
    const sent = uploadInTurn([file("a"), file("b")], { signal: ctl.signal }, (f, one) => uploadOutlines("d1", f, one));
    ctl.abort();
    expect(isAbort(await sent.catch((e: unknown) => e))).toBe(true);
    expect(xhr.requests).toHaveLength(1);
  });
});
