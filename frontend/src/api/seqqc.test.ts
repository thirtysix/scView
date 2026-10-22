import { describe, it, expect, vi, afterEach } from "vitest";
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
    const f = respond({ samples: [{ sample: "s2" }], n_samples: 1, metrics_info: [] });
    vi.stubGlobal("fetch", f);
    const file = new File(["x"], "metrics_summary.csv");
    await attachSeqMetrics("d1", [file], { column: "sample", sample: "s2" });
    const init = (f.mock.calls[0] as unknown[])[1] as RequestInit;
    const form = init.body as FormData;
    expect(init.method).toBe("POST");
    expect((form.getAll("files")[0] as File).name).toBe("metrics_summary.csv");
    expect(form.get("column")).toBe("sample");
    expect(form.get("sample")).toBe("s2");
    vi.stubGlobal("fetch", respond({ detail: "None of the 40 cells is in this raw matrix." }, 400));
    await expect(attachSeqMetrics("d1", [file])).rejects.toThrow("None of the 40 cells is in this raw matrix.");
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>bad gateway</html>", { status: 502 })));
    await expect(attachSeqMetrics("d1", [file])).rejects.toThrow(/backend did not answer/);
  });

  it("removes one sample by name", async () => {
    const f = respond({ samples: [], n_samples: 0, metrics_info: [] });
    vi.stubGlobal("fetch", f);
    expect(await removeSeqMetrics("d1", "a b")).toBeNull();
    expect(String((f.mock.calls[0] as unknown[])[0])).toContain("seq-metrics?sample=a%20b");
  });
});
