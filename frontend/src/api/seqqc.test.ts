import { describe, it, expect, vi, afterEach } from "vitest";
import { installFakeXhr } from "@/test/fakeXhr";
import { attachSeqMetrics, getSeqMetrics, removeSeqMetrics } from "./seqqc";

const respond = (body: unknown, status = 200) =>
  vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

describe("seqqc api", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("asks with allow_empty and reads an empty answer (or a 404) as none", async () => {
    const f = respond({ samples: [], n_samples: 0, metrics_info: [] });
    vi.stubGlobal("fetch", f);
    expect(await getSeqMetrics("d1")).toBeNull();
    expect(String((f.mock.calls[0] as unknown[])[0])).toContain("/datasets/d1/seq-metrics?allow_empty=true");
    vi.stubGlobal("fetch", respond({ detail: "No sequencing metrics are attached to this dataset." }, 404));
    expect(await getSeqMetrics("d1")).toBeNull();
  });

  it("sends the files and the target sample as multipart, and surfaces the server's sentence", async () => {
    const xhr = installFakeXhr((x) => x.respond(200, { samples: [{ sample: "s2" }], n_samples: 1, metrics_info: [] }));
    const file = new File(["x"], "metrics_summary.csv");
    await attachSeqMetrics("d1", [file], { column: "sample", sample: "s2" });
    const req = xhr.requests[0]!;
    const form = req.body as FormData;
    expect(req.method).toBe("POST");
    expect(req.url).toBe("/api/v1/datasets/d1/seq-metrics");
    expect((form.getAll("files")[0] as File).name).toBe("metrics_summary.csv");
    expect(form.get("column")).toBe("sample");
    expect(form.get("sample")).toBe("s2");
    installFakeXhr((x) => x.respond(400, { detail: "None of the 40 cells is in this raw matrix." }));
    await expect(attachSeqMetrics("d1", [file])).rejects.toThrow("None of the 40 cells is in this raw matrix.");
    installFakeXhr((x) => x.respond(502, "<html>bad gateway</html>"));
    await expect(attachSeqMetrics("d1", [file])).rejects.toThrow(/backend did not answer/);
  });

  it("removes one sample by name", async () => {
    const f = respond({ samples: [], n_samples: 0, metrics_info: [] });
    vi.stubGlobal("fetch", f);
    expect(await removeSeqMetrics("d1", "a b")).toBeNull();
    expect(String((f.mock.calls[0] as unknown[])[0])).toContain("seq-metrics?sample=a%20b");
  });
});
