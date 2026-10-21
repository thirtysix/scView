import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clusterReportHtmlUrl,
  fetchClusterReport,
  fetchGeneReport,
  geneReportHtmlUrl,
  ReportError,
} from "./report";

afterEach(() => vi.unstubAllGlobals());

const respond = (body: unknown, status = 200) =>
  vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(body), { status })));

describe("report client", () => {
  it("builds escaped report and download URLs", async () => {
    const f = respond({ kind: "cluster" });
    vi.stubGlobal("fetch", f);
    await fetchClusterReport("ds", "cell type", "CD14 Mono/x");
    expect(f.mock.calls[0]![0]).toBe("/api/v1/datasets/ds/report/cluster?column=cell+type&value=CD14+Mono%2Fx");
    await fetchGeneReport("ds", "ISG15");
    expect(f.mock.calls[1]![0]).toBe("/api/v1/datasets/ds/report/gene?gene=ISG15");
    expect(clusterReportHtmlUrl("ds", "cluster", "<b>")).toBe(
      "/api/v1/datasets/ds/report/cluster.html?column=cluster&value=%3Cb%3E");
    expect(geneReportHtmlUrl("ds", "ISG15", "cluster")).toBe(
      "/api/v1/datasets/ds/report/gene.html?gene=ISG15&groupby=cluster");
  });

  it("rejects with the message and a gene suggestion from the detail", async () => {
    vi.stubGlobal("fetch", respond({ detail: { message: "Gene 'cd14' not found.", suggestion: "CD14" } }, 404));
    const err = await fetchGeneReport("ds", "cd14").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ReportError);
    expect(err).toMatchObject({ message: "Gene 'cd14' not found.", status: 404, suggestion: "CD14" });
    vi.stubGlobal("fetch", respond({ detail: "Column 'x' not found." }, 404));
    await expect(fetchClusterReport("ds", "x", "1")).rejects.toMatchObject({ message: "Column 'x' not found.", suggestion: null });
  });

  it("says the backend is down for nginx's error page", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>502</html>", { status: 502 })));
    await expect(fetchGeneReport("ds", "ISG15")).rejects.toThrow(/backend did not answer/);
  });
});
