import { describe, it, expect, vi, afterEach } from "vitest";
import {
  collectionLabel,
  collectionOptionLabel,
  deleteCustomCollection,
  describeCoverage,
  GeneSetUploadError,
  listCustomCollections,
  pruneDeletedCustom,
  renameCustomCollection,
  uploadCustomCollection,
  type CustomCollection,
} from "./genesets";

afterEach(() => vi.unstubAllGlobals());

const col = (over: Partial<CustomCollection> = {}): CustomCollection => ({
  id: "custom-abc",
  name: "Lab",
  created_at: "",
  source_filename: "lab.gmt",
  format: "gmt",
  n_sets: 3,
  n_genes: 52,
  category: "Custom",
  available: true,
  ...over,
});

describe("custom gene-set collections API", () => {
  it("lists for a dataset when given one", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(async () => new Response(JSON.stringify({ collections: [col()] })));
    vi.stubGlobal("fetch", fetchMock);
    await expect(listCustomCollections("ds1")).resolves.toHaveLength(1);
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/v1/datasets/ds1/genesets/custom");
    await listCustomCollections();
    expect(fetchMock.mock.calls[1]![0]).toBe("/api/v1/genesets/custom");
  });

  it("uploads the file and name as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ...col(), warnings: [] }), { status: 201 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["S\t\tA\n"], "lab.gmt");
    await uploadCustomCollection(file, "Lab");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/v1/genesets/custom/upload");
    expect(init.method).toBe("POST");
    const body = init.body as FormData;
    expect((body.get("file") as File).name).toBe("lab.gmt");
    expect(body.get("name")).toBe("Lab");
    // The browser must set the multipart boundary itself.
    expect(init.headers).toBeUndefined();
  });

  it("a rejected file throws its message and every problem", async () => {
    const detail = { message: "bad.gmt: 2 problems found, nothing was saved.", problems: ["line 1: x", "line 2: y"] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail }), { status: 400 })));
    const err = await uploadCustomCollection(new File(["x"], "bad.gmt"), "x").catch((e) => e);
    expect(err).toBeInstanceOf(GeneSetUploadError);
    expect(err.message).toBe(detail.message);
    expect(err.problems).toEqual(detail.problems);
  });

  it("a plain string detail becomes the message; a non-JSON body is kept", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "already exists" }), { status: 400 })),
    );
    await expect(renameCustomCollection("custom-abc", "Lab")).rejects.toThrow(/^already exists$/);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Bad gateway", { status: 502 })));
    await expect(deleteCustomCollection("custom-abc")).rejects.toThrow("Error 502: Bad gateway");
  });

  it("rename PATCHes JSON and delete accepts a 204", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(col({ name: "New" }))))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(renameCustomCollection("custom-abc", "New")).resolves.toMatchObject({ name: "New" });
    expect(fetchMock.mock.calls[0]![1].method).toBe("PATCH");
    expect(JSON.parse(fetchMock.mock.calls[0]![1].body)).toEqual({ name: "New" });
    await expect(deleteCustomCollection("custom-abc")).resolves.toBeUndefined();
    expect(fetchMock.mock.calls[1]![0]).toBe("/api/v1/genesets/custom/custom-abc");
  });
});

describe("custom collection helpers", () => {
  it("drops deleted custom ids from a selection and keeps MSigDB ids", () => {
    const sel = new Set(["h.all", "custom-gone", "custom-kept"]);
    expect([...pruneDeletedCustom(sel, ["custom-kept"])]).toEqual(["h.all", "custom-kept"]);
  });

  it("labels results and options", () => {
    expect(collectionLabel({ collection: "h.all" })).toBe("h.all");
    expect(collectionLabel({ collection: "custom-abc", collection_name: "Lab (deleted)" })).toBe("Lab (deleted)");
    expect(describeCoverage(col())).toBe("3 sets");
    expect(describeCoverage(col({ n_genes_present: 41 }))).toBe("3 sets · 41/52 genes in this dataset");
    expect(collectionOptionLabel({ id: "h.all", name: "Hallmark", n_sets: 50 })).toBe("Hallmark (50 sets)");
    expect(collectionOptionLabel(col({ n_genes_present: 41 }))).toBe("Lab (3 sets, 41/52 genes in data)");
  });
});
