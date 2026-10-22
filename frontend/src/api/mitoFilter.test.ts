import { describe, it, expect, vi, afterEach } from "vitest";
import { getMitoRefilter, runMitoRefilter } from "./mitoFilter";
import { getFilterPreview, type FilterThresholds } from "./qc";

function sse(text: string, status = 200): Response {
  const body = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode(text));
      c.close();
    },
  });
  return new Response(body, { status });
}

afterEach(() => vi.unstubAllGlobals());

const T: FilterThresholds = { min_genes: 200, max_genes: 0, min_counts: 0, max_pct_mt: 20, drop_doublets: false };

describe("mito filter API", () => {
  it("asks the filter preview for per-type bounds only when they are chosen", async () => {
    const fetchSpy = vi.fn<(url: string) => Promise<Response>>(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchSpy);
    await getFilterPreview("d1", T);
    expect(String(fetchSpy.mock.calls[0]![0])).not.toContain("mito_mode");
    await getFilterPreview("d1", { ...T, qc_mito_mode: "per_type", qc_mito_type_column: "Cell type" });
    const url = new URL(String(fetchSpy.mock.calls[1]![0]), "http://x");
    expect(url.searchParams.get("mito_mode")).toBe("per_type");
    expect(url.searchParams.get("mito_type_column")).toBe("Cell type");
    await getFilterPreview("d1", { ...T, qc_mito_mode: "per_type" });
    expect(new URL(String(fetchSpy.mock.calls[2]![0]), "http://x").searchParams.has("mito_type_column")).toBe(false);
  });

  it("sends the floor only when it is on (preview, re-filter preview and run)", async () => {
    const fetchSpy = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(async () =>
      new Response("{}", { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchSpy);
    await getFilterPreview("d1", { ...T, qc_mito_mode: "per_type" });
    expect(new URL(String(fetchSpy.mock.calls[0]![0]), "http://x").searchParams.has("mito_type_floor")).toBe(false);
    await getFilterPreview("d1", { ...T, qc_mito_mode: "per_type", qc_mito_type_floor: true });
    expect(new URL(String(fetchSpy.mock.calls[1]![0]), "http://x").searchParams.get("mito_type_floor")).toBe("true");
    await getFilterPreview("d1", { ...T, qc_mito_mode: "global", qc_mito_type_floor: true }); // one rule: no floor
    expect(new URL(String(fetchSpy.mock.calls[2]![0]), "http://x").searchParams.has("mito_type_floor")).toBe(false);
    await getMitoRefilter("d1", "cell_type", true);
    expect(String(fetchSpy.mock.calls[3]![0])).toContain("/qc/mito-refilter?column=cell_type&floor=true");
    await getMitoRefilter("d1");
    expect(String(fetchSpy.mock.calls[4]![0])).toMatch(/\/qc\/mito-refilter$/);
    fetchSpy.mockImplementation(async () => sse('event: complete\ndata: {"steps_run": []}\n\n'));
    await runMitoRefilter("d1", "cell_type", () => {}, true);
    expect(JSON.parse(String(fetchSpy.mock.calls[5]![1]!.body))).toEqual({ column: "cell_type", floor: true });
  });

  it("previews with the column and gives the backend's own reason on a refusal", async () => {
    const fetchSpy = vi.fn<(url: string) => Promise<Response>>(async () =>
      new Response(JSON.stringify({ detail: "No original upload is stored for this dataset." }), { status: 409 }),
    );
    vi.stubGlobal("fetch", fetchSpy);
    await expect(getMitoRefilter("d1", "scview_leiden_r0.5")).rejects.toThrow(/^No original upload is stored/);
    expect(String(fetchSpy.mock.calls[0]![0])).toContain("/datasets/d1/qc/mito-refilter?column=scview_leiden_r0.5");
  });

  it("reads the run's events and resolves with the completed run", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => sse(
      'event: refilter\ndata: {"stage": "load", "message": "Reading the original upload"}\n\n' +
        ': keep-alive\n\n' +
        'event: step_start\ndata: {"step": "filtering", "index": 0, "total": 2}\n\n' +
        'event: complete\ndata: {"steps_run": ["filtering", "pca"]}\n\n',
    )));
    const seen: string[] = [];
    const out = await runMitoRefilter("d1", "cell_type", (e) => seen.push(e.event));
    expect(seen).toEqual(["refilter", "step_start", "complete"]);
    expect(out).toEqual({ steps_run: ["filtering", "pca"] });
  });

  it("rejects on an error event, a cut stream and a refusal", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => sse('event: error\ndata: {"error": "boom"}\n\n')));
    await expect(runMitoRefilter("d1", undefined, () => {})).rejects.toThrow("boom");
    vi.stubGlobal("fetch", vi.fn(async () => sse('event: refilter\ndata: {"stage": "load"}\n\n')));
    await expect(runMitoRefilter("d1", undefined, () => {})).rejects.toThrow(/nothing was saved/);
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Column 'x' not found." }), { status: 404 })));
    await expect(runMitoRefilter("d1", "x", () => {})).rejects.toThrow("Column 'x' not found.");
  });
});
