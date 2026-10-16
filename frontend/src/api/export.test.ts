import { describe, it, expect, vi, afterEach } from "vitest";
import { downloadTable, filenameFromDisposition, h5adDownloadUrl } from "./export";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("filenameFromDisposition", () => {
  it("reads quoted, bare and RFC 5987 filenames", () => {
    expect(filenameFromDisposition('attachment; filename="scview_pbmc_metadata.csv"', "x")).toBe(
      "scview_pbmc_metadata.csv",
    );
    expect(filenameFromDisposition("attachment; filename=a.xlsx", "x")).toBe("a.xlsx");
    expect(filenameFromDisposition("attachment; filename*=utf-8''H%C3%A4m.h5ad", "x")).toBe("Häm.h5ad");
  });

  it("falls back when the header is missing or has no name", () => {
    expect(filenameFromDisposition(null, "fallback.csv")).toBe("fallback.csv");
    expect(filenameFromDisposition("attachment", "fallback.csv")).toBe("fallback.csv");
  });
});

describe("export API", () => {
  it("links the .h5ad directly so the browser streams it", () => {
    expect(h5adDownloadUrl("abc")).toBe("/api/v1/datasets/abc/export/h5ad");
  });

  it("downloads a table under the server's filename", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("grouping,group,gene\n", {
        headers: { "content-disposition": 'attachment; filename="scview_pbmc_markers.csv"' },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    URL.createObjectURL = vi.fn(() => "blob:test");
    URL.revokeObjectURL = vi.fn();
    const clicked: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push(this.download);
    });

    await downloadTable("abc", "markers", "csv");

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/datasets/abc/export");
    expect(JSON.parse(init.body)).toEqual({ format: "csv", data_type: "markers", params: {} });
    expect(clicked).toEqual(["scview_pbmc_markers.csv"]);
  });

  it("throws the server's reason when the export fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "No marker genes found in this dataset." }), { status: 404 }),
      ),
    );
    await expect(downloadTable("abc", "markers", "xlsx")).rejects.toThrow("No marker genes found in this dataset.");
  });
});
