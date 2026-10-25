import { describe, it, expect, vi } from "vitest";
import {
  HYBRID_THRESHOLD,
  RASTER_SCALE,
  buildScatterSvg,
  countPoints,
  defaultPointMode,
  embeddingAxisLabels,
  projectPoints,
  wrapText,
  type ScatterSvgOptions,
  type SvgPanel,
} from "./scatterSvg";
import { OrthographicView } from "@deck.gl/core";
import { splitGridLayout } from "./splitView";
import { labelViewOf } from "./centroidLabels";
import type { Measure } from "./scatterExport";

const measure: Measure = (text) => text.length * 6;
const parse = (svg: string) => new DOMParser().parseFromString(svg, "image/svg+xml");

const panel = (xy: number[], rgba: number[], extra: Partial<SvgPanel> = {}): SvgPanel => ({
  view: { x: 0, y: 0, width: 300, height: 200 },
  sets: [{ xy: Float32Array.from(xy), colors: Uint8Array.from(rgba), radius: 2 }],
  ...extra,
});

const base = (over: Partial<ScatterSvgOptions> = {}): ScatterSvgOptions => ({
  width: 300,
  height: 200,
  panels: [panel([10, 10, 20, 20, 30, 30], [255, 0, 0, 204, 0, 128, 255, 204, 0, 128, 255, 31])],
  legend: {
    type: "categorical",
    title: "cell type",
    categories: ["B", "CD4 T", "NK"],
    colors: [[255, 0, 0], [0, 128, 255], [10, 20, 30]],
  },
  colormap: "viridis",
  mode: "vector",
  measure,
  ...over,
});

describe("buildScatterSvg (vector points)", () => {
  it("writes one circle per point, in draw order, with the screen's colours and opacity", () => {
    const out = buildScatterSvg(base());
    expect(out.circles).toBe(3);
    const doc = parse(out.svg);
    const circles = [...doc.querySelectorAll("g.points circle")];
    expect(circles.map((c) => [c.getAttribute("cx"), c.getAttribute("cy"), c.getAttribute("r")])).toEqual([
      ["10", "10", "2"],
      ["20", "20", "2"],
      ["30", "30", "2"],
    ]);
    expect(circles.map((c) => c.getAttribute("fill"))).toEqual(["#ff0000", "#0080ff", "#0080ff"]);
    // The common alpha sits on the group; a dimmed cell carries its own
    expect(doc.querySelector("g.points > g")!.getAttribute("fill-opacity")).toBe("0.8");
    expect(circles[2]!.getAttribute("fill-opacity")).toBe("0.122");
    expect(circles[0]!.getAttribute("fill-opacity")).toBeNull();
  });

  it("is on white and draws the legend entries, title and axis names", () => {
    const doc = parse(buildScatterSvg(base({ title: "Kang: cell type", axes: ["UMAP1", "UMAP2"] })).svg);
    expect(doc.querySelector("svg > rect")!.getAttribute("fill")).toBe("#ffffff");
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(["Kang: cell type", "UMAP1", "UMAP2", "cell type", "B", "CD4 T", "NK"]));
    const swatches = [...doc.querySelectorAll("g.legend circle")].map((c) => c.getAttribute("fill"));
    expect(swatches).toEqual(["#ff0000", "#0080ff", "#0a141e"]);
    expect(doc.querySelector("parsererror")).toBeNull();
  });

  it("draws a continuous legend as a gradient bar with its caption under the figure", () => {
    const doc = parse(
      buildScatterSvg(
        base({
          legend: {
            type: "continuous",
            title: "IFIT1",
            min: -2,
            max: 2,
            colormap: "RdBu",
            centre: 0,
            caption: "Colour range -2 to 2, centred on 0.",
          },
        }),
      ).svg,
    );
    expect(doc.querySelectorAll("linearGradient stop")).toHaveLength(33);
    expect(doc.querySelector("g.legend rect")!.getAttribute("fill")).toBe("url(#legend-bar-0)");
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(["IFIT1", "2.00", "0.00", "-2.00", "Colour range -2 to 2, centred on 0."]));
  });

  it("leaves out points outside the view and fully transparent ones", () => {
    const out = buildScatterSvg(base({ panels: [panel([10, 10, -50, 10, 20, 900, 40, 40], [1, 2, 3, 255, 1, 2, 3, 255, 1, 2, 3, 255, 1, 2, 3, 0])] }));
    expect(out.circles).toBe(1);
  });

  it("draws labels over the points with a white halo; faded ones in a faded group", () => {
    const doc = parse(
      buildScatterSvg(
        base({
          panels: [
            panel([10, 10], [0, 0, 0, 255], {
              labels: [
                { text: "B & T <x>", x: 50, y: 60, faded: false },
                { text: "NK", x: 90, y: 60, faded: true },
              ],
            }),
          ],
        }),
      ).svg,
    );
    const label = [...doc.querySelectorAll("g.panel > text")].filter((t) => t.textContent === "B & T <x>");
    expect(label.map((t) => t.getAttribute("stroke"))).toEqual(["#ffffff", null]);
    expect(doc.querySelector("g.labels-faded")!.getAttribute("opacity")).toBe("0.3");
    expect(doc.querySelector("g.labels-faded text")!.textContent).toBe("NK");
  });

  it("writes one-colour sets (a grid's grey context) with the colour on the group", () => {
    const doc = parse(
      buildScatterSvg(
        base({
          panels: [
            {
              view: { x: 0, y: 0, width: 300, height: 200 },
              sets: [{ xy: Float32Array.from([1, 1, 2, 2]), colors: [203, 213, 225, 89], radius: 2 }],
            },
          ],
        }),
      ).svg,
    );
    const g = doc.querySelector("g.points > g")!;
    expect(g.getAttribute("fill")).toBe("#cbd5e1");
    expect(g.querySelectorAll("circle")).toHaveLength(2);
  });

  it("puts a sampled note under the figure", () => {
    const doc = parse(buildScatterSvg(base({ notes: ["Showing 100 of 200 cells (sampled)"] })).svg);
    expect([...doc.querySelectorAll("text")].map((t) => t.textContent)).toContain("Showing 100 of 200 cells (sampled)");
  });
});

describe("points as an embedded image", () => {
  it("defaults to the image above the threshold", () => {
    expect(defaultPointMode(HYBRID_THRESHOLD)).toBe("vector");
    expect(defaultPointMode(HYBRID_THRESHOLD + 1)).toBe("raster");
  });

  it("embeds one image per panel and keeps every text vector", () => {
    const rasterize = vi.fn(() => "data:image/png;base64,AAAA");
    const out = buildScatterSvg(base({ mode: "raster", rasterize, title: "T", axes: ["UMAP1", "UMAP2"] }));
    expect(out.circles).toBe(0);
    expect(rasterize).toHaveBeenCalledWith(expect.any(Array), 300, 200, RASTER_SCALE);
    const doc = parse(out.svg);
    expect(doc.querySelectorAll("g.points circle")).toHaveLength(0);
    const img = doc.querySelector("g.points image")!;
    expect(img.getAttribute("xlink:href")).toBe("data:image/png;base64,AAAA");
    expect([img.getAttribute("width"), img.getAttribute("height")]).toEqual(["300", "200"]);
    expect([...doc.querySelectorAll("text")].map((t) => t.textContent)).toEqual(
      expect.arrayContaining(["T", "UMAP1", "B", "NK"]),
    );
    expect(doc.querySelectorAll("g.legend circle")).toHaveLength(3);
  });

  it("needs a rasteriser", () => {
    expect(() => buildScatterSvg(base({ mode: "raster" }))).toThrow(/rasteriser/);
  });
});

describe("grid figure", () => {
  it("places every panel where the grid layout puts it, with titles and frames, and ONE legend", () => {
    const grid = splitGridLayout(4, 600, 400, 1);
    const panels: SvgPanel[] = grid.cells.map((cell, i) => ({
      view: cell.view,
      title: { box: cell.title, label: `donor ${i}`, subtitle: `${i + 1} cells` },
      frame: true,
      sets: [{ xy: Float32Array.from([5, 5]), colors: Uint8Array.from([0, 0, 0, 255]), radius: 1 }],
    }));
    const doc = parse(buildScatterSvg(base({ width: 600, height: 400, panels })).svg);
    const groups = [...doc.querySelectorAll("g.panel")];
    expect(groups).toHaveLength(4);
    expect(groups.map((g) => g.getAttribute("transform"))).toEqual(
      grid.cells.map((c) => `translate(${c.view.x} ${c.view.y})`),
    );
    expect(grid.cols).toBe(2);
    // Panels sit side by side, then wrap, with the gap between them
    expect(grid.cells[1]!.view.x).toBe(grid.cells[0]!.view.width + 8);
    expect(grid.cells[2]!.view.y).toBe(grid.cells[0]!.view.y + grid.cells[0]!.view.height + 22 + 8);
    expect(doc.querySelectorAll("g.legend")).toHaveLength(1);
    expect(doc.querySelectorAll("clipPath")).toHaveLength(4);
    const texts = [...doc.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(["donor 0", "1 cells", "donor 3", "4 cells"]));
    expect(countPoints(panels)).toBe(4);
  });

  it("widens the figure for the legend strip", () => {
    const withLegend = buildScatterSvg(base());
    const without = buildScatterSvg(base({ legend: null }));
    expect(withLegend.width).toBeGreaterThan(without.width + 40);
  });
});

describe("at 100k points", () => {
  it("stays a few MB and builds fast as vector", () => {
    const n = 100_000;
    const xy = new Float32Array(n * 2);
    const rgba = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      xy[i * 2] = (i * 7.3) % 800;
      xy[i * 2 + 1] = (i * 3.1) % 600;
      rgba.set([(i * 13) % 256, (i * 7) % 256, 90, 204], i * 4);
    }
    const t0 = performance.now();
    const out = buildScatterSvg(
      base({ width: 800, height: 600, panels: [{ view: { x: 0, y: 0, width: 800, height: 600 }, sets: [{ xy, colors: rgba, radius: 2 }] }] }),
    );
    const ms = performance.now() - t0;
    expect(out.circles).toBe(n);
    // ~60 bytes a circle
    expect(out.svg.length).toBeLessThan(8_000_000);
    expect(ms).toBeLessThan(3000);
    console.info(`[scatterSvg] 100k vector: ${(out.svg.length / 1e6).toFixed(1)} MB in ${ms.toFixed(0)} ms`);
  });
});

describe("projectPoints", () => {
  const colors = Uint8Array.from([1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255]);

  it("projects 100k points through a real deck.gl viewport quickly, matching the label projection", () => {
    const view = new OrthographicView({ id: "ortho", flipY: false });
    const lv = labelViewOf(view, 800, 600, { target: [0, 0, 0], zoom: 2 })!;
    const n = 100_000;
    const pos = new Float32Array(n * 2).map((_, i) => ((i * 37) % 200) - 100);
    const t0 = performance.now();
    const out = projectPoints(lv.project, pos, 2, new Uint8Array(n * 4));
    const ms = performance.now() - t0;
    console.info(`[scatterSvg] projecting 100k points: ${ms.toFixed(0)} ms`);
    expect(ms).toBeLessThan(3000);
    // World (0, 0) is the view's centre; y grows upwards on screen (flipY false)
    const centre = projectPoints(lv.project, Float32Array.from([0, 0, 1, 1]), 2, new Uint8Array(8)).xy;
    expect([centre[0], centre[1]]).toEqual([400, 300]);
    expect([centre[2], centre[3]]).toEqual([404, 296]);
    expect(out.xy.length).toBe(n * 2);
  });

  it("projects 2D points in their draw order", () => {
    const out = projectPoints((p) => [p[0]! * 10, 100 - p[1]! * 10], Float32Array.from([0, 0, 1, 1, 2, 2]), 2, colors);
    expect([...out.xy]).toEqual([0, 100, 10, 90, 20, 80]);
    expect(out.colors).toBe(colors);
  });

  it("sorts 3D points far to near, carrying their colours", () => {
    const out = projectPoints(
      (p) => [p[0]!, p[1]!, p[2]!],
      Float32Array.from([0, 0, 0.2, 1, 1, 0.9, 2, 2, 0.5]),
      3,
      colors,
    );
    expect([...out.xy]).toEqual([1, 1, 2, 2, 0, 0]);
    expect([...out.colors].filter((_, i) => i % 4 === 0)).toEqual([2, 3, 1]);
  });
});

describe("embeddingAxisLabels and wrapText", () => {
  it("names the axes after the embedding", () => {
    expect(embeddingAxisLabels("X_umap")).toEqual(["UMAP1", "UMAP2"]);
    expect(embeddingAxisLabels("X_tsne")).toEqual(["tSNE1", "tSNE2"]);
    expect(embeddingAxisLabels("X_pca", [2, 3])).toEqual(["PC3", "PC4"]);
    expect(embeddingAxisLabels("X_pca")).toEqual(["PC1", "PC2"]);
    // A pair the embedding lacks is not what the plot shows
    expect(embeddingAxisLabels("X_pca", [2, 60], 50)).toEqual(["PC1", "PC2"]);
    expect(embeddingAxisLabels("X_umap", [2, 3], 2)).toEqual(["UMAP1", "UMAP2"]);
  });

  it("wraps a long caption to the width", () => {
    const lines = wrapText("aaaa bbbb cccc dddd", 60, "11px x", measure);
    expect(lines).toEqual(["aaaa bbbb", "cccc dddd"]);
  });
});
