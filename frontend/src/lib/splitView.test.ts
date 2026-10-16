import { describe, it, expect, vi, afterEach } from "vitest";
import type { ObsColumnInfo } from "@/api/types";
import { mapCategoryToColor, mapValueToColor } from "@/lib/colors";
import type { CapturableDeck } from "@/lib/scatterExport";
import {
  SPLIT_CONTEXT_LAYER,
  SPLIT_GAP,
  SPLIT_TITLE_H,
  buildSplitPanels,
  dataBounds,
  downloadSplitPng,
  fitSplitViewState,
  gatherPositions,
  layerInView,
  panelColors,
  panelOfLayer,
  sampleCellIndices,
  splitColumnOptions,
  splitGridLayout,
  splitLayerId,
  splitViewId,
  valueRange,
} from "./splitView";

afterEach(() => vi.restoreAllMocks());

const col = (name: string, values?: string[], dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique: values?.length ?? 5000,
  values,
});

const rgbaAt = (rgba: Uint8Array, s: number) => Array.from(rgba.slice(s * 4, s * 4 + 4));

describe("splitColumnOptions", () => {
  it("offers categorical columns with 2–6 values and disables bigger ones with the reason", () => {
    const opts = splitColumnOptions([
      col("stim", ["CTRL", "STIM"]),
      col("cell_type", Array.from({ length: 13 }, (_, i) => `t${i}`)),
      col("sample", ["a", "b", "c", "d", "e", "f"], "object"),
      col("is_doublet", ["False", "True"], "bool"),
      col("one", ["only"]),
      col("barcode"), // over the 100-value cap: no values listed
      { name: "n_genes", dtype: "int64", n_unique: 900 },
    ]);
    expect(opts.map((o) => o.name)).toEqual(["is_doublet", "sample", "stim", "cell_type"]);
    expect(opts.filter((o) => o.disabledReason == null)).toHaveLength(3);
    const ct = opts.find((o) => o.name === "cell_type")!;
    expect(ct.nValues).toBe(13);
    expect(ct.disabledReason).toMatch(/13 values.*at most 6/);
  });
});

describe("sampleCellIndices", () => {
  it("draws every cell when under the cap", () => {
    expect(sampleCellIndices(1000, 1000)).toBeNull();
  });

  it("returns a sorted, distinct, repeatable sample of the cap's size", () => {
    const a = sampleCellIndices(5000, 800)!;
    expect(a).toHaveLength(800);
    expect(new Set(a).size).toBe(800);
    for (let i = 1; i < a.length; i++) expect(a[i]!).toBeGreaterThan(a[i - 1]!);
    expect(a[a.length - 1]!).toBeLessThan(5000);
    expect(Array.from(sampleCellIndices(5000, 800)!)).toEqual(Array.from(a));
  });
});

describe("buildSplitPanels", () => {
  const codes = Int32Array.from([0, 1, 0, -1, 2, 0, 1]);
  const labels = ["CTRL", "STIM", "MOCK", "UNUSED"];

  it("gives one panel per value with cells, in the column's order", () => {
    const { panels, missing } = buildSplitPanels(codes, labels, null);
    expect(panels.map((p) => [p.label, p.count])).toEqual([
      ["CTRL", 3],
      ["STIM", 2],
      ["MOCK", 1],
    ]);
    expect(Array.from(panels[0]!.indices)).toEqual([0, 2, 5]);
    expect(Array.from(panels[1]!.indices)).toEqual([1, 6]);
    expect(missing).toBe(1);
    // Every cell with a value is in exactly one panel.
    const all = panels.flatMap((p) => Array.from(p.indices)).sort((x, y) => x - y);
    expect(all).toEqual([0, 1, 2, 4, 5, 6]);
  });

  it("draws only sampled cells but titles keep the full counts", () => {
    const { panels } = buildSplitPanels(codes, labels, Uint32Array.from([0, 1, 4]));
    expect(panels.map((p) => p.count)).toEqual([3, 2, 1]);
    expect(panels.map((p) => Array.from(p.indices))).toEqual([[0], [1], [4]]);
  });
});

describe("gatherPositions", () => {
  it("copies the chosen cells' coordinates", () => {
    const pos = Float32Array.from([0, 0, 1, 10, 2, 20, 3, 30]);
    expect(Array.from(gatherPositions(pos, Uint32Array.from([1, 3]), 2))).toEqual([1, 10, 3, 30]);
    const pos3 = Float32Array.from([0, 0, 0, 1, 2, 3]);
    expect(Array.from(gatherPositions(pos3, Uint32Array.from([1]), 3))).toEqual([1, 2, 3]);
  });
});

describe("shared colour scale", () => {
  // Cells 0-2 are CTRL (low expression), 3-5 STIM (high).
  const expr = Float32Array.from([0, 0.5, 1, 2, 3, 4]);
  const ctrl = Uint32Array.from([0, 1, 2]);
  const stim = Uint32Array.from([3, 4, 5]);

  it("takes one range over every cell, skipping missing values", () => {
    expect(valueRange(expr)).toEqual({ min: 0, max: 4 });
    expect(valueRange(Float32Array.from([NaN, 2, NaN, -1]))).toEqual({ min: -1, max: 2 });
    expect(valueRange(Float32Array.from([NaN]))).toEqual({ min: 0, max: 0 });
    expect(valueRange(null)).toEqual({ min: 0, max: 0 });
  });

  it("colours a panel against the shared range, not its own", () => {
    const range = valueRange(expr);
    const opts = { colorValues: expr, colorType: "continuous" as const, range, opacity: 1, colormap: "viridis" as const };
    const c = panelColors(ctrl, opts);
    const s = panelColors(stim, opts);
    // CTRL's brightest cell (1.0) is a quarter of the way up the shared scale,
    // not the top colour it would get from a per-panel range.
    expect(rgbaAt(c, 2)).toEqual([...mapValueToColor(1, 0, 4, "viridis").slice(0, 3), 255]);
    expect(rgbaAt(c, 2)).not.toEqual(rgbaAt(s, 2));
    expect(rgbaAt(s, 2)).toEqual([...mapValueToColor(4, 0, 4, "viridis").slice(0, 3), 255]);
  });

  it("gives a cell the same colour whichever panel draws it", () => {
    const range = valueRange(expr);
    const opts = { colorValues: expr, colorType: "continuous" as const, range, opacity: 0.8 };
    const alone = panelColors(Uint32Array.from([4]), opts);
    expect(rgbaAt(alone, 0)).toEqual(rgbaAt(panelColors(stim, opts), 1));
  });

  it("uses the scatter's category colours and fades cells outside a selection", () => {
    const codes = Int32Array.from([2, 0, 5]);
    const rgba = panelColors(Uint32Array.from([0, 2]), {
      colorValues: codes,
      colorType: "categorical",
      range: { min: 0, max: 1 },
      opacity: 1,
      selected: new Set([0]),
      palette: "tableau20",
    });
    expect(rgbaAt(rgba, 0)).toEqual([...mapCategoryToColor(2, "tableau20").slice(0, 3), 255]);
    expect(rgbaAt(rgba, 1)).toEqual([...mapCategoryToColor(5, "tableau20").slice(0, 3), Math.round(0.15 * 255)]);
  });

  it("treats float values as continuous even while the type still says categorical", () => {
    const rgba = panelColors(Uint32Array.from([5]), {
      colorValues: expr,
      colorType: "categorical",
      range: valueRange(expr),
      opacity: 1,
    });
    expect(rgbaAt(rgba, 0)).toEqual([...mapValueToColor(4, 0, 4).slice(0, 3), 255]);
  });
});

describe("splitGridLayout", () => {
  const check = (n: number, w: number, h: number, aspect = 1) => {
    const g = splitGridLayout(n, w, h, aspect);
    expect(g.cells).toHaveLength(n);
    for (const c of g.cells) {
      expect(c.view.y).toBe(c.title.y + SPLIT_TITLE_H);
      expect(c.view.x + c.view.width).toBeLessThanOrEqual(w);
      expect(c.view.y + c.view.height).toBeLessThanOrEqual(h);
      // One camera fits all: every view is the same size.
      expect([c.view.width, c.view.height]).toEqual([g.cells[0]!.view.width, g.cells[0]!.view.height]);
    }
    return g;
  };

  it("puts two panels side by side in a wide area and stacked in a tall one", () => {
    expect(check(2, 1000, 500)).toMatchObject({ cols: 2, rows: 1 });
    expect(check(2, 500, 1000)).toMatchObject({ cols: 1, rows: 2 });
  });

  it("tiles four and six panels", () => {
    expect(check(4, 900, 900)).toMatchObject({ cols: 2, rows: 2 });
    expect(check(6, 1200, 800)).toMatchObject({ cols: 3, rows: 2 });
  });

  it("leaves a gap between panels and fills rows left to right", () => {
    const g = check(3, 1200, 400);
    expect(g).toMatchObject({ cols: 3, rows: 1 });
    expect(g.cells[1]!.title.x - (g.cells[0]!.title.x + g.cells[0]!.title.width)).toBe(SPLIT_GAP);
  });

  it("follows the data's shape", () => {
    // Very wide data in a squarish area: stack the panels.
    expect(check(2, 800, 700, 4)).toMatchObject({ cols: 1, rows: 2 });
  });

  it("is empty before the area is measured", () => {
    expect(splitGridLayout(3, 0, 0).cells).toEqual([]);
  });
});

describe("fitSplitViewState", () => {
  it("centres the data and fits it, padded, into one panel", () => {
    const pos = Float32Array.from([0, 0, 10, 5]);
    const vs = fitSplitViewState(dataBounds(pos, 2), 2, 220, 110);
    expect(vs.target).toEqual([5, 2.5, 0]);
    expect(vs.zoom).toBeCloseTo(Math.log2(220 / 11));
  });

  it("starts 3D panels at the single plot's orbit", () => {
    const pos = Float32Array.from([0, 0, 0, 2, 4, 6]);
    const vs = fitSplitViewState(dataBounds(pos, 3), 3, 300, 200);
    expect(vs).toMatchObject({ target: [1, 2, 3], rotationX: 30, rotationOrbit: -30 });
  });
});

describe("layer and view ids", () => {
  it("draws the grey context in every view and each panel's cells in its own view", () => {
    expect(layerInView(SPLIT_CONTEXT_LAYER, splitViewId(4))).toBe(true);
    expect(layerInView(splitLayerId(2), splitViewId(2))).toBe(true);
    expect(layerInView(splitLayerId(2), splitViewId(1))).toBe(false);
    expect(layerInView(splitLayerId(1), splitViewId(12))).toBe(false);
    expect(panelOfLayer(splitLayerId(3))).toBe(3);
    expect(panelOfLayer(SPLIT_CONTEXT_LAYER)).toBeNull();
    expect(panelOfLayer(undefined)).toBeNull();
  });
});

describe("downloadSplitPng", () => {
  it("captures all panels in one redraw, titles them, then restores the live view", () => {
    const log: string[] = [];
    const source = document.createElement("canvas");
    Object.defineProperty(source, "clientWidth", { value: 600 });
    Object.defineProperty(source, "clientHeight", { value: 300 });
    const deck: CapturableDeck = {
      getCanvas: () => source,
      setProps: ({ useDevicePixels }) => {
        log.push(`dpr ${useDevicePixels}`);
        if (typeof useDevicePixels === "number") {
          source.width = 600 * useDevicePixels;
          source.height = 300 * useDevicePixels;
        }
      },
      redraw: (reason) => log.push(`redraw ${reason}`),
    };
    const ctx = {
      font: "",
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 1,
      textBaseline: "",
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      drawImage: vi.fn(() => log.push("copy")),
      scale: vi.fn(),
      translate: vi.fn(),
      fillText: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      measureText: (t: string) => ({ width: t.length * 7 }),
      createLinearGradient: () => ({ addColorStop: vi.fn() }),
    };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(() => {});

    const grid = splitGridLayout(2, 600, 300);
    downloadSplitPng(deck, {
      grid,
      panels: [
        { label: "CTRL", count: 7000 },
        { label: "STIM", count: 6000 },
      ],
      legend: { type: "continuous", title: "CD14", min: 0, max: 3 },
      colormap: "viridis",
      filename: "scview_kang_UMAP_CD14_split_stim",
    });

    expect(log.slice(0, 2)).toEqual(["dpr 3", "redraw png export"]);
    expect(log.slice(-2)).toEqual(["dpr true", "redraw png export done"]);
    expect(ctx.fillText).toHaveBeenCalledWith("CTRL", grid.cells[0]!.title.x + 4, expect.any(Number));
    expect(ctx.fillText).toHaveBeenCalledWith("STIM", grid.cells[1]!.title.x + 4, expect.any(Number));
    expect(ctx.strokeRect).toHaveBeenCalledTimes(2);
    expect(toBlob).toHaveBeenCalledTimes(1);
  });

  it("restores the live view even if the capture fails", () => {
    const setProps = vi.fn();
    const deck: CapturableDeck = { getCanvas: () => document.createElement("canvas"), setProps, redraw: vi.fn() };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    expect(() =>
      downloadSplitPng(deck, { grid: splitGridLayout(1, 100, 100), panels: [], legend: null, colormap: "viridis", filename: "f" }),
    ).toThrow();
    expect(setProps).toHaveBeenLastCalledWith({ useDevicePixels: true });
  });
});
