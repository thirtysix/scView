import { describe, it, expect, vi, afterEach } from "vitest";
import type { DatasetInfo } from "@/api/types";
import { encodeSelection, SESSION_FORMAT, SESSION_VERSION, type SessionFile } from "@/lib/session";
import {
  buildShareUrl,
  canCompress,
  describeSharedView,
  encodeShareValue,
  fromBase64Url,
  hasShareHash,
  matchSharedDataset,
  openShareHash,
  readShareHash,
  SHARE_URL_MAX,
  ShareLinkError,
  toBase64Url,
  type ShareDeps,
} from "./shareUrl";

const here = { origin: "http://localhost:3000", pathname: "/", search: "" };

const session = (extra: Partial<SessionFile> = {}): SessionFile => ({
  format: SESSION_FORMAT,
  version: SESSION_VERSION,
  saved_at: "2026-10-21T10:00:00.000Z",
  dataset: { id: "kang-id", name: "Kang 2018", n_cells: 13836, cells_fingerprint: "fp-kang" },
  view: {
    panel: "unified",
    subtab: "de",
    embedding: "X_umap",
    color_by: "cluster",
    color_by_chosen: true,
    camera: { target: [1.5, -2.25], zoom: 3.2 },
    highlight: { column: "cluster", value: "CD14 Mono" },
    overlay: { type: "expression", gene: "ISG15" },
  },
  de: { mode: "groups", split_by: "stim", group_a: "STIM", group_b: "CTRL", test: "pseudobulk", replicate_key: "Donor" },
  colors: { categorical_palette: "cvd", point_size: 3 },
  ...extra,
});

const hashOf = (url: string) => url.slice(url.indexOf("#"));

/** `n` distinct cells of `nCells`, spread out (a lasso's cells are not in obs order). */
function spread(n: number, nCells: number): Set<number> {
  const out = new Set<number>();
  let x = 12345;
  while (out.size < n) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    out.add(x % nCells);
  }
  return out;
}

const ds = (over: Partial<DatasetInfo>): DatasetInfo => ({
  id: "x",
  name: "x",
  filename: "x.h5ad",
  n_cells: 13836,
  n_genes: 100,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  created_at: "",
  ...over,
});

const notFound = () => new Error('API error 404: {"detail":"Dataset not found."}');

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("base64url", () => {
  it("round-trips any bytes without + / or =", () => {
    const bytes = new Uint8Array(1000).map((_, i) => (i * 37) % 256);
    const text = toBase64Url(bytes);
    expect(text).toMatch(/^[A-Za-z0-9_-]+$/);
    expect([...fromBase64Url(text)]).toEqual([...bytes]);
  });

  it("refuses text that is not base64url", () => {
    expect(() => fromBase64Url("ab+c")).toThrow(ShareLinkError);
    expect(() => fromBase64Url("abcde")).toThrow(ShareLinkError); // a cut-short length
  });
});

describe("a link round trip", () => {
  it("compresses with deflate-raw where the browser can, and reads back the same view", async () => {
    expect(canCompress()).toBe(true); // Node's streams
    const s = session();
    const { url, dropped } = await buildShareUrl(s, here);
    expect(url.startsWith("http://localhost:3000/#view=v1.z.")).toBe(true);
    expect(dropped).toEqual([]);
    const back = await readShareHash(hashOf(url));
    const rest: SessionFile = { ...s };
    delete rest.saved_at;
    expect(back).toEqual(rest);
    // Far shorter than the JSON itself.
    expect(url.length).toBeLessThan(JSON.stringify(s).length);
  });

  it("falls back to plain base64url JSON without CompressionStream", async () => {
    vi.stubGlobal("CompressionStream", undefined);
    expect(canCompress()).toBe(false);
    const s = session({ selection: encodeSelection([3, 1, 2], 13836) });
    const { url } = await buildShareUrl(s, here);
    expect(url).toContain("#view=v1.j.");
    const back = await readShareHash(hashOf(url));
    expect(back?.selection).toEqual({ n_cells: 13836, indices: [1, 2, 3] });
    expect(back?.view?.overlay).toEqual({ type: "expression", gene: "ISG15" });
  });

  it("reads a plain link anywhere, and a compressed one only where the browser can decompress", async () => {
    const plain = await encodeShareValue(session(), false);
    const packed = await encodeShareValue(session(), true);
    vi.stubGlobal("DecompressionStream", undefined);
    expect((await readShareHash(`#view=${plain}`))?.dataset?.id).toBe("kang-id");
    await expect(readShareHash(`#view=${packed}`)).rejects.toThrow(/cannot decompress/);
  });

  it("still reads a link a chat app took the full stop or bracket after it into", async () => {
    const { url } = await buildShareUrl(session(), here);
    expect((await readShareHash(`${hashOf(url)}.`))?.dataset?.id).toBe("kang-id");
    expect((await readShareHash(`${hashOf(url)}),`))?.dataset?.id).toBe("kang-id");
  });

  it("keeps the page's path and query, and puts the view only in the fragment", async () => {
    const { url } = await buildShareUrl(session(), { origin: "http://10.0.0.5:3000", pathname: "/app/", search: "?x=1" });
    expect(url.startsWith("http://10.0.0.5:3000/app/?x=1#view=v1.")).toBe(true);
    expect(new URL(url).search).toBe("?x=1"); // nothing of the view in the query string
  });
});

describe("what a link carries", () => {
  it("never the user's gene lists or gates", async () => {
    const s = {
      ...session(),
      gene_lists: [{ name: "IFN", genes: ["ISG15", "IFI6"] }],
      gates: [{ name: "T cells", filter: {} }],
    } as unknown as SessionFile;
    const { url } = await buildShareUrl(s, here);
    const json = JSON.stringify(await readShareHash(hashOf(url)));
    expect(json).not.toContain("gene_lists");
    expect(json).not.toContain("gates");
    expect(json).not.toContain("saved_at");
  });

  it("drops them from a hand-made link too", async () => {
    const crafted = { ...session(), gene_lists: [{ name: "Injected", genes: ["A"] }], gates: [{ name: "g" }] };
    const value = `v1.j.${toBase64Url(new TextEncoder().encode(JSON.stringify(crafted)))}`;
    const back = await readShareHash(`#view=${value}`);
    expect(back).not.toHaveProperty("gene_lists");
    expect(back).not.toHaveProperty("gates");
  });
});

describe("the size cap", () => {
  it("keeps a few hundred lassoed cells", async () => {
    const s = session({ selection: encodeSelection(spread(300, 13836), 13836) });
    const { url, dropped } = await buildShareUrl(s, here);
    expect(dropped).toEqual([]);
    expect(url.length).toBeLessThanOrEqual(SHARE_URL_MAX);
    expect((await readShareHash(hashOf(url)))?.selection).toEqual(s.selection);
  });

  it("drops a large selection first and says how many cells it held", async () => {
    const s = session({
      dataset: { id: "big", name: "Big", n_cells: 200_000, cells_fingerprint: "fp" },
      selection: encodeSelection(spread(80_000, 200_000), 200_000),
    });
    const { url, dropped } = await buildShareUrl(s, here);
    expect(dropped).toEqual(["the selection of 80,000 cells"]);
    expect(url.length).toBeLessThanOrEqual(SHARE_URL_MAX);
    const back = await readShareHash(hashOf(url));
    expect(back?.selection).toBeUndefined();
    expect(back?.view?.camera).toEqual({ target: [1.5, -2.25], zoom: 3.2 }); // still there
  });

  it("then the camera", async () => {
    const s = session({ selection: encodeSelection(spread(300, 13836), 13836) });
    const full = (await buildShareUrl(s, here)).url.length;
    const noSelection = (await buildShareUrl({ ...s, selection: null }, here)).url.length;
    // A cap the link fits only without both.
    const { url, dropped } = await buildShareUrl(s, here, { maxLength: noSelection - 5 });
    expect(full).toBeGreaterThan(noSelection);
    expect(dropped).toEqual(["the selection of 300 cells", "the zoom and position (camera)"]);
    const back = await readShareHash(hashOf(url));
    expect(back?.view?.camera).toBeNull();
    expect(back?.view?.overlay).toEqual({ type: "expression", gene: "ISG15" });
  });

  it("keeps a short filter selection longer than the camera", async () => {
    const filter = { op: "and" as const, layer: "", items: [{ kind: "category" as const, column: "stim", values: ["STIM"] }] };
    const s = session({ selection: { filter } });
    const withoutCamera = (await buildShareUrl({ ...s, view: { ...s.view, camera: null } }, here)).url.length;
    const { url, dropped } = await buildShareUrl(s, here, { maxLength: withoutCamera });
    expect(dropped).toEqual(["the zoom and position (camera)"]);
    expect((await readShareHash(hashOf(url)))?.selection).toEqual({ filter });
  });

  it("drops a long gene-set overlay last", async () => {
    const genes = Array.from({ length: 4000 }, (_, i) => `GENE${(i * 7919) % 100000}X${i}`);
    const s = session({ view: { ...session().view, camera: null, overlay: { type: "score", name: "Huge set", genes } } });
    const { url, dropped } = await buildShareUrl(s, here);
    expect(dropped).toEqual(["the gene-set score overlay “Huge set” (4,000 genes)"]);
    expect(url.length).toBeLessThanOrEqual(SHARE_URL_MAX);
  });
});

describe("reading a damaged link", () => {
  it.each([
    ["#view=garbage", /not in scView's link format/],
    ["#view=v1.z.ab*cd", /not valid base64/],
    ["#view=v1.z.AAAAAAAA", /does not decompress|not readable JSON/],
    ["#view=v9.z.AAAA", /newer scView/],
    [`#view=v1.j.${toBase64Url(new TextEncoder().encode('{"format":"other"}'))}`, /can't be used: This is not an scView session file/],
    [`#view=v1.j.${toBase64Url(new TextEncoder().encode("{not json"))}`, /not readable JSON/],
    [`#view=v1.j.${"A".repeat(250_000)}`, /too long/],
  ])("%s is refused with a plain reason", async (hash, reason) => {
    await expect(readShareHash(hash)).rejects.toThrow(reason);
    await expect(readShareHash(hash)).rejects.toBeInstanceOf(ShareLinkError);
  });

  it("stops inflating a link that would fill memory", async () => {
    const bomb = `{"format":"scview-session","version":1,"pad":"${" ".repeat(6 * 1024 * 1024)}"}`;
    const value = await (async () => {
      const cs = new CompressionStream("deflate-raw");
      const w = cs.writable.getWriter();
      void w.write(new TextEncoder().encode(bomb));
      void w.close();
      const chunks: number[] = [];
      const r = cs.readable.getReader();
      for (;;) {
        const { done, value: v } = await r.read();
        if (done) break;
        chunks.push(...v);
      }
      return toBase64Url(new Uint8Array(chunks));
    })();
    expect(value.length).toBeLessThan(20_000);
    await expect(readShareHash(`#view=v1.z.${value}`)).rejects.toThrow(/too large/);
  });

  it("is not a link at all without view=", async () => {
    expect(hasShareHash("")).toBe(false);
    expect(hasShareHash("#section-2")).toBe(false);
    expect(hasShareHash("#view=v1.z.x")).toBe(true);
    expect(await readShareHash("#section-2")).toBeNull();
  });
});

describe("matchSharedDataset", () => {
  const saved = session().dataset!;

  it("opens the dataset with the link's id when it exists", async () => {
    const getDataset = vi.fn().mockResolvedValue(ds({ id: "kang-id" }));
    const listDatasets = vi.fn();
    expect(await matchSharedDataset(saved, { getDataset, listDatasets })).toMatchObject({ by: "id", dataset: { id: "kang-id" } });
    expect(listDatasets).not.toHaveBeenCalled();
  });

  it("else finds the same cells under another id (another scView)", async () => {
    const getDataset = vi.fn().mockRejectedValue(notFound());
    const listDatasets = vi.fn().mockResolvedValue([
      ds({ id: "a", n_cells: 13836, cells_fingerprint: "other-cells" }),
      ds({ id: "mine", name: "kang copy", n_cells: 13836, cells_fingerprint: "fp-kang" }),
    ]);
    expect(await matchSharedDataset(saved, { getDataset, listDatasets })).toMatchObject({ by: "cells", dataset: { id: "mine" } });
  });

  it("asks for the fingerprint only of datasets with the same number of cells", async () => {
    const getDataset = vi.fn(async (id: string) => {
      if (id === "same-n") return ds({ id, n_cells: 13836, cells_fingerprint: "fp-kang" });
      throw notFound();
    });
    const listDatasets = vi.fn().mockResolvedValue([
      ds({ id: "paul", n_cells: 2730 }),
      ds({ id: "converting", n_cells: 13836, status: "converting" }),
      ds({ id: "same-n", n_cells: 13836 }), // never opened: the list has no fingerprint
    ]);
    expect(await matchSharedDataset(saved, { getDataset, listDatasets })).toMatchObject({ by: "cells", dataset: { id: "same-n" } });
    expect(getDataset.mock.calls.map((c) => c[0])).toEqual(["kang-id", "same-n"]);
  });

  it("is null when no dataset has those cells", async () => {
    const getDataset = vi.fn().mockRejectedValue(notFound());
    const listDatasets = vi.fn().mockResolvedValue([ds({ id: "a", n_cells: 13836, cells_fingerprint: "nope" })]);
    expect(await matchSharedDataset(saved, { getDataset, listDatasets })).toBeNull();
  });

  it("throws on a server error rather than guessing", async () => {
    const getDataset = vi.fn().mockRejectedValue(new Error("API error 503: busy"));
    await expect(matchSharedDataset(saved, { getDataset, listDatasets: vi.fn() })).rejects.toThrow(/503/);
  });
});

describe("openShareHash", () => {
  const report = { headline: "Opened “kang copy”.", notes: [], level: "ok" as const };
  const deps = (over: Partial<ShareDeps> = {}): ShareDeps => ({
    getDataset: vi.fn().mockRejectedValue(notFound()),
    listDatasets: vi.fn().mockResolvedValue([]),
    restoreSession: vi.fn().mockResolvedValue({ ...report }),
    ...over,
  });
  const linkHash = async (s: SessionFile = session()) => hashOf((await buildShareUrl(s, here)).url);

  it("says plainly when the library lacks the dataset, and restores nothing", async () => {
    const d = deps();
    const out = await openShareHash(await linkHash(), d);
    expect(out).toEqual({
      headline:
        "This link is for dataset “Kang 2018” (13,836 cells), which is not in this scView's library. " +
        "Load that file, then open the link again.",
      notes: [],
      level: "warning",
    });
    expect(d.restoreSession).not.toHaveBeenCalled();
  });

  it("restores onto the dataset with the same cells, under its own id", async () => {
    const d = deps({
      listDatasets: vi.fn().mockResolvedValue([ds({ id: "mine", name: "kang copy", cells_fingerprint: "fp-kang" })]),
    });
    const out = await openShareHash(await linkHash(), d);
    const restored = vi.mocked(d.restoreSession).mock.calls[0]![0];
    expect(restored.dataset).toMatchObject({ id: "mine", cells_fingerprint: "fp-kang" });
    expect(restored.view?.overlay).toEqual({ type: "expression", gene: "ISG15" });
    expect(out?.headline).toBe("Opened “kang copy”. It has the same cells as “Kang 2018”, the dataset the link was made on.");
  });

  it("reports a damaged link without trying to restore it", async () => {
    const d = deps();
    expect(await openShareHash("#view=v1.z.%%%", d)).toMatchObject({ level: "error", retry: false });
    expect(d.restoreSession).not.toHaveBeenCalled();
    expect(await openShareHash("#other", d)).toBeNull();
  });

  it("marks a server error as worth a reload", async () => {
    const d = deps({ getDataset: vi.fn().mockRejectedValue(new Error("API error 502: down")) });
    expect(await openShareHash(await linkHash(), d)).toMatchObject({
      level: "error",
      retry: true,
      headline: "The link's dataset could not be looked up (502: down). Reload the page to try again.",
    });
  });
});

describe("describeSharedView", () => {
  it("lists what the link restores, in words", () => {
    const s = session({ selection: encodeSelection([1, 2, 3], 13836) });
    expect(describeSharedView(s)).toEqual([
      "dataset “Kang 2018” (13,836 cells)",
      "the Unified View panel (DE tab)",
      "UMAP coloured by cluster",
      "the ISG15 expression overlay",
      "the highlighted group cluster = CD14 Mono",
      "the zoom and position",
      "the selection of 3 cells",
      "the DE settings",
      "the colours and point settings",
    ]);
  });

  it("counts a bitmask selection's cells", () => {
    const sel = encodeSelection(spread(9000, 13836), 13836);
    expect("bitmask" in sel).toBe(true);
    expect(describeSharedView(session({ selection: sel }))).toContain("the selection of 9,000 cells");
  });
});
