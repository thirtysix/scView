import { describe, it, expect } from "vitest";
import {
  cameraFromViewState,
  decodeSelection,
  encodeMask,
  encodeSelection,
  parseSession,
  readSessionFile,
  SESSION_MAX_BYTES,
  SessionError,
  sessionFilename,
  sessionFromStorage,
  sessionJson,
  type SessionFile,
} from "./session";

const full: SessionFile = {
  format: "scview-session",
  version: 1,
  saved_at: "2026-10-16T20:00:00.000Z",
  dataset: { id: "578b70cb-7924", name: "Kang", n_cells: 13836, cells_fingerprint: "abc123" },
  view: {
    panel: "unified",
    subtab: "de",
    embedding: "X_umap",
    color_by: "cell_type",
    color_by_chosen: true,
    expression_layer: "raw",
    group_by: "cell_type",
    split_by: null,
    show_context: true,
    compare_columns: ["cell_type", "stim"],
    labels: { key: "cell_type", show: false },
    camera: { target: [1.5, -2, 0], zoom: 3.25 },
    highlight: { column: "cell_type", value: "CD14 Mono" },
    overlay: { type: "score", name: "HALLMARK_INTERFERON_ALPHA_RESPONSE", genes: ["ISG15", "IFI6"] },
  },
  selection: {
    filter: {
      op: "and",
      layer: "raw",
      items: [
        { kind: "gene", gene: "CD14", op: ">", value: 1 },
        { kind: "category", column: "stim", values: ["STIM"] },
      ],
    },
  },
  de: {
    mode: "groups",
    split_by: "stim",
    group_a: "STIM",
    group_b: "CTRL",
    within_column: "cell_type",
    within_value: "CD14 Mono",
    test: "pseudobulk",
    replicate_key: "ind",
    fc_threshold: 1.5,
    min_pct: 0.05,
  },
  colors: {
    categorical_palette: "tableau20",
    continuous_colormap: "magma",
    point_size: 3,
    opacity: 0.6,
    background: "dark",
    max_rendered_cells: 50000,
  },
};

const json = (x: unknown) => JSON.parse(JSON.stringify(x));

describe("parseSession", () => {
  it("reads back every field of a saved session", () => {
    expect(parseSession(json(full))).toEqual(full);
  });

  it("ignores fields it does not know and accepts a partial session", () => {
    expect(parseSession({ ...json(full), extra: { x: 1 }, view: { panel: "markers", future: true } }).view).toEqual({
      panel: "markers",
    });
    expect(parseSession({ format: "scview-session", version: 1 })).toEqual({ format: "scview-session", version: 1 });
  });

  it("rejects other files and other versions, saying which", () => {
    expect(() => parseSession([1, 2])).toThrow(/not an scView session file/);
    expect(() => parseSession({ ...full, format: "kana" })).toThrow(/not an scView session file/);
    expect(() => parseSession({ ...full, version: 2 })).toThrow(/newer scView \(format version 2\)/);
    expect(() => parseSession({ ...full, version: "1" })).toThrow(/Unknown session format version 1/);
    expect(() => parseSession({ ...full, version: 0 })).toThrow(SessionError);
  });

  it.each([
    ["a dataset id that is a path", { dataset: { id: "../../admin" } }, /dataset.id is not a dataset id/],
    ["a numeric dataset id", { dataset: { id: 7 } }, /dataset.id must be/],
    ["an unknown panel", { view: { panel: "secret" } }, /view.panel must be one of/],
    ["a colour-by that is not text", { view: { color_by: ["a"] } }, /view.color_by must be/],
    ["a string boolean", { view: { color_by_chosen: "yes" } }, /view.color_by_chosen must be true or false/],
    ["one compare column", { view: { compare_columns: ["a"] } }, /view.compare_columns must list 2 to 4/],
    ["a camera without zoom", { view: { camera: { target: [0, 0] } } }, /view.camera.zoom/],
    ["a camera at infinity", { view: { camera: { target: [0, Infinity], zoom: 1 } } }, /view.camera.target\[1\]/],
    ["an unknown overlay", { view: { overlay: { type: "obs", column: "x" } } }, /view.overlay.type/],
    ["an unknown activity network", { view: { overlay: { type: "activity", network: "dorothea", source: "STAT1" } } }, /view.overlay.network/],
    ["a negative index", { selection: { n_cells: 10, indices: [1, -1] } }, /selection.indices must hold whole numbers/],
    ["a fractional index", { selection: { n_cells: 10, indices: [1.5] } }, /selection.indices/],
    ["more indices than cells", { selection: { n_cells: 1, indices: [0, 0] } }, /at most 1 cells/],
    ["a short bitmask", { selection: { n_cells: 100, bitmask: "AAAA" } }, /base64 of 13 bytes/],
    ["a bitmask that is not base64", { selection: { n_cells: 16, bitmask: "!!!!" } }, /selection.bitmask/],
    ["a bad filter", { selection: { filter: { op: "and", items: [{ kind: "gene", gene: "CD14", op: ">", value: "1" }] } } },
      /selection.filter.items\[0\].value must be a number/],
    ["an unknown DE test", { de: { test: "t-test" } }, /de.test must be one of/],
    ["a detection cut-off over 100%", { de: { min_pct: 10 } }, /de.min_pct/],
    ["an unknown palette", { colors: { categorical_palette: "toString" } }, /colors.categorical_palette/],
    ["a huge point size", { colors: { point_size: 1e6 } }, /colors.point_size/],
  ])("rejects %s", (_name, patch, message) => {
    expect(() => parseSession({ format: "scview-session", version: 1, ...patch })).toThrow(message);
  });
});

describe("readSessionFile", () => {
  it("reads a file, refusing one too large or not JSON", async () => {
    await expect(readSessionFile(new Blob([JSON.stringify(full)]))).resolves.toEqual(full);
    await expect(readSessionFile(new Blob(["{oops"]))).rejects.toThrow(/not JSON/);
    const big = { size: SESSION_MAX_BYTES + 1, text: async () => "{}" } as unknown as Blob;
    await expect(readSessionFile(big)).rejects.toThrow(/too large/);
  });
});

describe("selection encoding", () => {
  it("lists a few cells and packs many, both decoding to the same cells", () => {
    const few = encodeSelection(new Set([5, 1, 3]), 1000);
    expect(few).toEqual({ n_cells: 1000, indices: [1, 3, 5] });
    const many = new Set(Array.from({ length: 600 }, (_, i) => i * 1.5).filter(Number.isInteger));
    const packed = encodeSelection(many, 1000);
    expect("bitmask" in packed).toBe(true);
    expect(decodeSelection(packed as Extract<typeof packed, { bitmask: string }>, 1000).cells).toEqual(many);
    // round trip through the file check
    expect(parseSession({ format: "scview-session", version: 1, selection: packed }).selection).toEqual(packed);
  });

  it("never uses cells beyond the dataset, and says how many it ignored", () => {
    expect(decodeSelection({ n_cells: 20, indices: [0, 9, 10, 19] }, 10)).toEqual({ cells: new Set([0, 9]), ignored: 2 });
    const mask = encodeMask([0, 9, 10, 19], 20);
    expect(decodeSelection({ n_cells: 20, bitmask: mask }, 10)).toEqual({ cells: new Set([0, 9]), ignored: 2 });
  });

  it("packs like the backend: bit i % 8 of byte i // 8", () => {
    expect(atob(encodeMask([0, 3, 9], 10)).split("").map((c) => c.charCodeAt(0))).toEqual([0b00001001, 0b00000010]);
  });
});

describe("the reload snapshot", () => {
  it("reads the older flat snapshot, skipping values it no longer knows", () => {
    expect(
      sessionFromStorage({
        datasetId: "ds1",
        activePanel: "unified",
        colorBy: "leiden",
        embedding: "X_umap",
        categoricalPalette: "cvd",
        continuousColormap: "nope",
      }),
    ).toEqual({
      format: "scview-session",
      version: 1,
      dataset: { id: "ds1" },
      view: { panel: "unified", color_by: "leiden", color_by_chosen: true, embedding: "X_umap" },
      colors: { categorical_palette: "cvd" },
    });
  });

  it("reads the session format, and nothing malformed", () => {
    expect(sessionFromStorage(json(full))).toEqual(full);
    expect(sessionFromStorage({ format: "scview-session", version: 1, view: { panel: 3 } })).toBeNull();
    expect(sessionFromStorage("x")).toBeNull();
  });
});

describe("helpers", () => {
  it("writes lists of numbers on one line and reads back the same session", () => {
    const s: SessionFile = { ...full, selection: { n_cells: 50, indices: [1, 2, 30] } };
    const text = sessionJson(s);
    expect(text).toContain('"indices": [1, 2, 30]');
    expect(text).toContain('"target": [1.5, -2, 0]');
    expect(text).toContain('"genes": [\n');
    expect(parseSession(JSON.parse(text))).toEqual(s);
  });

  it("takes the camera fields of a view state", () => {
    expect(cameraFromViewState({ target: [1, 2, 0], zoom: 2, minZoom: -4, rotationOrbit: 30, width: 9 })).toEqual({
      target: [1, 2, 0],
      zoom: 2,
      rotationOrbit: 30,
    });
    expect(cameraFromViewState({ target: [NaN, 0], zoom: 1 })).toBeNull();
  });

  it("names the file after the dataset and the day", () => {
    expect(sessionFilename("Kang 2018 (IFN-β)", new Date("2026-10-16T21:00:00Z"))).toBe(
      "scview_Kang_2018_IFN_session_2026-10-16.json",
    );
  });
});
