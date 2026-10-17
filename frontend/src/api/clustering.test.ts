import { describe, it, expect, vi, afterEach } from "vitest";
import { keepHvgSetting, runHvgSensitivity, runSweep, subclusterGroup, type HvgRow, type SweepRow } from "./clustering";

afterEach(() => vi.unstubAllGlobals());

/** A streamed response delivering `chunks` one read at a time. */
function streamed(chunks: string[], status = 200): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(ctrl) {
      for (const c of chunks) ctrl.enqueue(enc.encode(c));
      ctrl.close();
    },
  });
  return new Response(body, { status, headers: { "Content-Type": "text/event-stream" } });
}

const row = (resolution: number): SweepRow => ({
  resolution,
  n_clusters: 3,
  sizes: [5, 3, 2],
  silhouette: 0.2,
  ari_previous: null,
  column: `scview_leiden_r${resolution}`,
  exists: false,
});

describe("runSweep", () => {
  it("reports the start and each result, even split mid-line", async () => {
    const text =
      `event: start\ndata: ${JSON.stringify({ method: "leiden", n_cells: 10, resolutions: [0.2, 1], representation: "X_pca", silhouette_cells: 10 })}\n\n` +
      `event: result\ndata: ${JSON.stringify(row(0.2))}\n\n` +
      `event: result\ndata: ${JSON.stringify(row(1))}\n\nevent: complete\ndata: {}\n\n`;
    const fetchMock = vi.fn().mockResolvedValue(streamed([text.slice(0, 50), text.slice(50, 140), text.slice(140)]));
    vi.stubGlobal("fetch", fetchMock);
    const onStart = vi.fn();
    const results: SweepRow[] = [];
    await runSweep("ds", { resolutions: [0.2, 1] }, { onStart, onResult: (r) => results.push(r) });
    expect(onStart).toHaveBeenCalledWith(expect.objectContaining({ method: "leiden", resolutions: [0.2, 1] }));
    expect(results.map((r) => r.resolution)).toEqual([0.2, 1]);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/datasets/ds/clustering/sweep");
    expect(JSON.parse(init.body)).toEqual({ resolutions: [0.2, 1] });
  });

  it("rejects with the server's reason", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "This dataset has no neighbour graph." }), { status: 400 })),
    );
    await expect(runSweep("ds", { resolutions: [1] }, { onResult: () => {} })).rejects.toThrow(/no neighbour graph/);
  });

  it("rejects on an error event or a stream cut short", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed([`event: error\ndata: {"error":"boom"}\n\n`])));
    await expect(runSweep("ds", { resolutions: [1] }, { onResult: () => {} })).rejects.toThrow("boom");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed([`event: result\ndata: ${JSON.stringify(row(1))}\n\n`])));
    await expect(runSweep("ds", { resolutions: [1] }, { onResult: () => {} })).rejects.toThrow(/before it finished/);
  });
});

describe("subclusterGroup", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts the column, group and resolution", async () => {
    const body = { column: "cl_sub", group: "B", n_cells: 220, n_subclusters: 2, groups: ["B.0", "B.1"],
                   sizes: [120, 100], representation: "X_pca", batch_key: null };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const res = await subclusterGroup("ds", "cl", "B");
    expect(res.groups).toEqual(["B.0", "B.1"]);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toContain("/datasets/ds/clustering/subcluster");
    expect(JSON.parse(init.body)).toEqual({ column: "cl", group: "B", resolution: 0.5 });
  });
});

describe("runHvgSensitivity", () => {
  const hvgRow = { key: "seurat_1000", flavor: "seurat", n_top_genes: 1000, ari: 0.9 } as HvgRow;

  it("reports the start and each setting's result", async () => {
    const text =
      `event: start\ndata: ${JSON.stringify({ n_cells: 10, reference: "cl" })}\n\n` +
      `event: result\ndata: ${JSON.stringify(hvgRow)}\n\nevent: complete\ndata: {}\n\n`;
    const fetchMock = vi.fn().mockResolvedValue(streamed([text.slice(0, 30), text.slice(30)]));
    vi.stubGlobal("fetch", fetchMock);
    const onStart = vi.fn();
    const results: HvgRow[] = [];
    const settings = [{ flavor: "seurat" as const, n_top_genes: 1000 }];
    await runHvgSensitivity("ds", { settings }, { onStart, onResult: (r) => results.push(r) });
    expect(onStart).toHaveBeenCalledWith(expect.objectContaining({ reference: "cl" }));
    expect(results.map((r) => r.key)).toEqual(["seurat_1000"]);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/datasets/ds/clustering/hvg-sensitivity");
    expect(JSON.parse(init.body)).toEqual({ settings });
  });

  it("rejects with the server's reason or a stream cut short", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "no clustering to compare with" }), { status: 400 })),
    );
    await expect(runHvgSensitivity("ds", { settings: [] }, { onResult: () => {} })).rejects.toThrow(/no clustering/);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed([`event: result\ndata: ${JSON.stringify(hvgRow)}\n\n`])));
    await expect(runHvgSensitivity("ds", { settings: [] }, { onResult: () => {} })).rejects.toThrow(
      "The check stopped before it finished.",
    );
  });

  it("keeps a setting by its flavour and gene count", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ column: "c", created: true, n_clusters: 3 }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await keepHvgSetting("ds", hvgRow);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toContain("/datasets/ds/clustering/hvg-sensitivity/keep");
    expect(JSON.parse(init.body)).toEqual({ flavor: "seurat", n_top_genes: 1000 });
  });
});
