import { describe, it, expect, vi, afterEach } from "vitest";
import { getAnnotationConfidence, makeUnknownColumn, runAnnotationConfidence } from "./annotation";

afterEach(() => vi.unstubAllGlobals());

function ok(body: unknown) {
  return vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(body), { status: 200 })));
}

describe("annotation confidence client", () => {
  it("asks for a column's last check, escaping the name", async () => {
    const fetchMock = ok({ available: true, result: null });
    vi.stubGlobal("fetch", fetchMock);
    await getAnnotationConfidence("ds", "cell type/fine");
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/v1/datasets/ds/annotation/confidence?column=cell%20type%2Ffine");
    await getAnnotationConfidence("ds");
    expect(fetchMock.mock.calls[1]![0]).toBe("/api/v1/datasets/ds/annotation/confidence");
  });

  it("posts the column and alpha to run a check", async () => {
    const fetchMock = ok({ column: "cluster", alpha: 0.05 });
    vi.stubGlobal("fetch", fetchMock);
    await runAnnotationConfidence("ds", "cluster", 0.05);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/datasets/ds/annotation/confidence");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ column: "cluster", alpha: 0.05 });
  });

  it("posts the column for the Unknown copy, and rejects with the server's reason", async () => {
    const fetchMock = ok({ column: "cluster", unknown_column: "scview_conf_cluster_unknown", n_unknown: 9, n_cells: 90 });
    vi.stubGlobal("fetch", fetchMock);
    const r = await makeUnknownColumn("ds", "cluster");
    expect(r.unknown_column).toBe("scview_conf_cluster_unknown");
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/v1/datasets/ds/annotation/confidence/unknown");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Run the confidence check on 'cluster' first." }), { status: 409 })));
    await expect(makeUnknownColumn("ds", "cluster")).rejects.toThrow(/409.*first/);
  });
});
