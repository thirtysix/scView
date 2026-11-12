import { beforeAll, describe, expect, it } from "vitest";
import { buildMesh, type CellOutlines } from "./cellOutlines";
import {
  figureOutlines,
  omittedOutlinesNote,
  outlinesSvg,
  simplifyRing,
  MAX_FIGURE_OUTLINES,
} from "./outlineSvg";
import { buildScatterSvg, type ScatterSvgOptions } from "./scatterSvg";
import { svgToPdf } from "./pdfExport";

/** Rings (x, y in obsm["spatial"] units, y down as stored) -> outlines. */
function outlines(rings: number[][]): CellOutlines {
  const offsets = [0];
  for (const r of rings) offsets.push(offsets[offsets.length - 1]! + r.length / 2);
  return { nCells: rings.length, offsets: Int32Array.from(offsets), xy: Float32Array.from(rings.flat()) };
}
const square = (x: number, y: number, h = 1) => [x - h, y - h, x + h, y - h, x + h, y + h, x - h, y + h];

/** An orthographic camera like deck.gl's: world (x, y up) -> pixels from the top-left, y down. */
const camera = (cx: number, cy: number, zoom: number, width: number, height: number) => {
  const s = 2 ** zoom;
  return (p: number[]) => [width / 2 + (p[0]! - cx) * s, height / 2 - (p[1]! - cy) * s];
};

/** RGBA per cell. */
const colors = (...c: number[][]) => Uint8Array.from(c.flat());

function parse(svg: string): Document {
  return new DOMParser().parseFromString(svg, "image/svg+xml");
}

describe("simplifyRing", () => {
  it("drops vertices on a straight edge, keeps the corners", () => {
    // A 10 px square with an extra vertex half-way along each side
    const px = Float64Array.from([0, 0, 5, 0, 10, 0, 10, 5, 10, 10, 5, 10, 0, 10, 0, 5]);
    expect([...simplifyRing(px, 8, 0.3)]).toEqual([1, 0, 1, 0, 1, 0, 1, 0]);
  });

  it("keeps a vertex further off the line than the tolerance", () => {
    const px = Float64Array.from([0, 0, 5, 1, 10, 0, 10, 10, 0, 10]);
    expect(simplifyRing(px, 5, 0.3)[1]).toBe(1);
    expect(simplifyRing(px, 5, 2)[1]).toBe(0);
  });
});

describe("figureOutlines", () => {
  // Cell 0 at (10, 10), cell 1 at (50, 50), cell 2 at (90, 90); squares of 2 units.
  const o = outlines([square(10, 10), square(50, 50), square(90, 90)]);
  const mesh = buildMesh(o, Int32Array.from([0, 1, 2]));
  const rgba = colors([255, 0, 0, 128], [0, 0, 255, 128], [255, 0, 0, 128]);

  it("projects with the plot's camera, in tenths of a pixel, relative moves", () => {
    // 8 px a unit, cell 1 in the middle of a 400 x 300 view
    const f = figureOutlines(mesh, rgba, camera(50, -50, 3, 400, 300), { width: 400, height: 300 }, 8);
    expect(f.count).toBe(1);
    expect(f.omitted).toBe(false);
    const groups = f.outlines!.groups;
    expect(groups).toHaveLength(1);
    expect(groups[0]!.rgba).toEqual([0, 0, 255, 128]);
    // (49, 49) stored = world (49, -49) = pixel (192, 142); then 16 px right, 16 px down, 16 px left
    expect(groups[0]!.d).toEqual(["M192 142l16 0l0 16l-16 0z"]);
    expect(f.outlines!.edge).toEqual({ rgba: [30, 41, 59, 130], width: 1 });
  });

  it("only writes the outlines that reach into the view", () => {
    // Zoomed so the view holds cells 0 and 1 only
    const f = figureOutlines(mesh, rgba, camera(30, -30, 2, 200, 200), { width: 200, height: 200 }, 4);
    expect(f.count).toBe(2);
    const ds = f.outlines!.groups.flatMap((g) => g.d);
    expect(ds.join("").split("z").filter(Boolean)).toHaveLength(2);
  });

  it("keeps a ring that only grazes the view (the clip path trims it)", () => {
    // Cell 1's square spans pixels 192..208 at zoom 3; a view 200 px wide ends inside it
    const f = figureOutlines(mesh, rgba, camera(50, -50, 3, 400, 300), { width: 200, height: 300 }, 8);
    expect(f.count).toBe(1);
  });

  it("groups rings by colour, in the order the colours first appear", () => {
    const f = figureOutlines(mesh, rgba, camera(50, -50, 0, 200, 200), { width: 200, height: 200 }, 1);
    expect(f.count).toBe(3);
    const groups = f.outlines!.groups;
    expect(groups.map((g) => g.rgba[2])).toEqual([0, 255]);
    expect(groups[0]!.d[0]!.split("z").filter(Boolean)).toHaveLength(2);
  });

  it("no edge when a cell is under MIN_EDGE_PIXELS wide on screen, as on the plot", () => {
    const f = figureOutlines(mesh, rgba, camera(50, -50, 0, 200, 200), { width: 200, height: 200 }, 1);
    expect(f.outlines!.edge).toBeNull();
  });

  it("leaves them out over the limits, still counting what is in view", () => {
    const byCount = figureOutlines(mesh, rgba, camera(50, -50, 0, 200, 200), { width: 200, height: 200 }, 1, {
      maxOutlines: 2,
      maxVertices: 1e6,
    });
    expect(byCount).toMatchObject({ outlines: null, omitted: true, count: 3 });
    expect(omittedOutlinesNote(byCount, "cell", { maxOutlines: 2, maxVertices: 1e6 })).toBe(
      "Cell outlines left out (3 in view, over the 2 a figure holds): zoom in for them. The cells are drawn as points.",
    );
    const byVertices = figureOutlines(mesh, rgba, camera(50, -50, 0, 200, 200), { width: 200, height: 200 }, 1, {
      maxOutlines: 100,
      maxVertices: 10,
    });
    expect(byVertices.omitted).toBe(true);
    expect(omittedOutlinesNote(byVertices, "nucleus", { maxOutlines: 100, maxVertices: 10 })).toMatch(
      /^Nucleus outlines left out \(over 10 outline points in view\)/,
    );
  });

  it("the default limit is above what the plot ever outlines", () => {
    expect(MAX_FIGURE_OUTLINES).toBeGreaterThan(30_000);
  });

  it("drops sub-pixel cells and simplifies dense rings", () => {
    // A 64-vertex circle of radius 3 units, at 1 px a unit (r = 3 px)
    const ring: number[] = [];
    for (let i = 0; i < 64; i++) ring.push(50 + 3 * Math.cos((i / 64) * 2 * Math.PI), 50 + 3 * Math.sin((i / 64) * 2 * Math.PI));
    const tiny = square(60, 60, 0.01);
    const m = buildMesh(outlines([ring, tiny]), Int32Array.from([0, 1]));
    const f = figureOutlines(m, colors([1, 2, 3, 255], [1, 2, 3, 255]), camera(50, -50, 0, 100, 100), { width: 100, height: 100 }, 1);
    expect(f.count).toBe(2); // both reach into the view...
    const rings = f.outlines!.groups[0]!.d.join("").split("z").filter(Boolean);
    expect(rings).toHaveLength(1); // ...the sub-pixel one has nothing to draw
    expect(f.vertices).toBeLessThan(20);
    expect(f.vertices).toBeGreaterThanOrEqual(8);
  });

  it("splits a colour's rings over several paths past a few thousand", () => {
    const rings: number[][] = [];
    for (let i = 0; i < 2_500; i++) rings.push(square(i % 50 * 4, Math.floor(i / 50) * 4));
    const m = buildMesh(outlines(rings), Int32Array.from(rings.map((_, i) => i)));
    const c = new Uint8Array(rings.length * 4).fill(200);
    const f = figureOutlines(m, c, camera(100, -100, 2, 1000, 1000), { width: 1000, height: 1000 }, 4);
    expect(f.count).toBe(2_500);
    expect(f.outlines!.groups[0]!.d).toHaveLength(2);
  });
});

describe("outlinesSvg", () => {
  it("fills each group in its colour and opacity, edges stroked on the group", () => {
    const out: string[] = [];
    outlinesSvg(
      {
        groups: [
          { rgba: [255, 0, 0, 128], d: ["M0 0l1 0l0 1z"] },
          { rgba: [0, 0, 255, 255], d: ["M5 5l1 0l0 1z"] },
          { rgba: [0, 0, 0, 0], d: ["M9 9l1 0l0 1z"] },
        ],
        edge: { rgba: [30, 41, 59, 130], width: 1 },
      },
      "panel-clip-0",
      out,
    );
    const doc = parse(`<svg xmlns="http://www.w3.org/2000/svg">${out.join("")}</svg>`);
    const g = doc.querySelector("g.outlines")!;
    expect(g.getAttribute("clip-path")).toBe("url(#panel-clip-0)");
    expect(g.getAttribute("stroke")).toBe("#1e293b");
    expect(g.getAttribute("stroke-opacity")).toBe("0.51");
    expect(g.getAttribute("stroke-width")).toBe("1");
    const paths = [...g.querySelectorAll("path")];
    expect(paths.map((p) => [p.getAttribute("fill"), p.getAttribute("fill-opacity")])).toEqual([
      ["#ff0000", "0.502"],
      ["#0000ff", null],
      ["none", null], // see-through: the edge alone, as on screen
    ]);
  });

  it("without an edge, no stroke and no see-through paths", () => {
    const out: string[] = [];
    outlinesSvg({ groups: [{ rgba: [0, 0, 0, 0], d: ["M9 9l1 0l0 1z"] }], edge: null }, "c", out);
    expect(out.join("")).not.toMatch(/stroke|<path/);
  });
});

describe("buildScatterSvg with outlines", () => {
  const base = (outlinesIn?: ScatterSvgOptions["panels"][number]["outlines"]): ScatterSvgOptions => ({
    width: 200,
    height: 150,
    panels: [
      {
        view: { x: 0, y: 0, width: 200, height: 150 },
        sets: [{ xy: Float32Array.from([10, 10, 100, 80]), colors: [255, 0, 0, 200] as const, radius: 2 }],
        labels: [{ text: "B cells", x: 100, y: 70, faded: false }],
        ...(outlinesIn !== undefined ? { outlines: outlinesIn } : {}),
      },
    ],
    legend: null,
    colormap: "viridis",
    mode: "vector",
  });
  const shapes = { groups: [{ rgba: [0, 128, 0, 255] as const, d: ["M50 50l10 0l0 10l-10 0z"] }], edge: null };

  it("draws them over the points, under the labels, clipped to the panel", () => {
    const doc = parse(buildScatterSvg(base(shapes)).svg);
    const panel = doc.querySelector("g.panel")!;
    const kids = [...panel.children].map((c) => c.getAttribute("class") ?? c.tagName);
    expect(kids.indexOf("points")).toBeLessThan(kids.indexOf("outlines"));
    expect(kids.indexOf("outlines")).toBeLessThan(kids.indexOf("text"));
    expect(panel.querySelector("g.outlines")!.getAttribute("clip-path")).toBe("url(#panel-clip-0)");
    expect(panel.querySelectorAll("g.outlines path")).toHaveLength(1);
  });

  it("over the points image too, in raster mode", () => {
    const doc = parse(buildScatterSvg({ ...base(shapes), mode: "raster", rasterize: () => "data:image/png;base64,AA==" }).svg);
    const panel = doc.querySelector("g.panel")!;
    expect(panel.querySelector("g.points image")).not.toBeNull();
    expect(panel.querySelectorAll("g.outlines path")).toHaveLength(1);
  });

  it("nothing changes without them", () => {
    const plain = buildScatterSvg(base()).svg;
    expect(buildScatterSvg(base(null)).svg).toBe(plain);
    expect(plain).not.toMatch(/outlines|<path/);
  });
});

describe("svgToPdf with outlines", () => {
  beforeAll(() => {
    (SVGElement.prototype as unknown as { getBBox: () => DOMRect }).getBBox = function (this: SVGElement) {
      return { x: 0, y: 0, width: (this.textContent ?? "").length * 6, height: 12 } as DOMRect;
    };
  });

  it("carries the outline paths into the PDF", async () => {
    const o = outlines(Array.from({ length: 200 }, (_, i) => square((i % 20) * 10, Math.floor(i / 20) * 10, 4)));
    const m = buildMesh(o, Int32Array.from({ length: 200 }, (_, i) => i));
    const c = new Uint8Array(200 * 4).map((_, i) => (i % 4 === 3 ? 200 : (i * 37) % 256));
    const f = figureOutlines(m, c, camera(100, -50, 1, 400, 300), { width: 400, height: 300 }, 2);
    const svgOf = (withOutlines: boolean) =>
      buildScatterSvg({
        width: 400,
        height: 300,
        panels: [{ view: { x: 0, y: 0, width: 400, height: 300 }, sets: [], outlines: withOutlines ? f.outlines : null }],
        legend: null,
        colormap: "viridis",
        mode: "vector",
      });
    const sizeOf = async (s: { svg: string; width: number; height: number }) =>
      (await svgToPdf(s.svg, s.width, s.height)).size;
    const without = await sizeOf(svgOf(false));
    const withThem = await sizeOf(svgOf(true));
    expect(withThem).toBeGreaterThan(without + 500);
  });
});

describe("at the plot's own limit", () => {
  it("30,000 cells of 25 vertices, about 5 px across: a few MB, built fast", () => {
    // Xenium-like: 25 vertices a cell boundary; a 1000 x 750 px view of 30,000 cells of ~5 px
    const rings: number[][] = [];
    for (let j = 0; j < 150; j++) {
      for (let i = 0; i < 200; i++) {
        const ring: number[] = [];
        for (let v = 0; v < 25; v++) {
          const t = (v / 25) * 2 * Math.PI;
          const r = 2.4 * (1 + 0.15 * Math.sin(3 * t + i));
          ring.push(i * 5 + r * Math.cos(t), j * 5 + r * Math.sin(t));
        }
        rings.push(ring);
      }
    }
    const m = buildMesh(outlines(rings), Int32Array.from({ length: rings.length }, (_, i) => i));
    expect(m.nVertices).toBe(750_000); // more than the plot's own 400,000
    const c = new Uint8Array(rings.length * 4).map((_, i) => (i % 4 === 3 ? 204 : ((i >> 2) % 13) * 19));
    const t0 = performance.now();
    const f = figureOutlines(m, c, camera(500, -375, 0, 1000, 750), { width: 1000, height: 750 }, 1, {
      maxOutlines: 1e9,
      maxVertices: 1e9,
    });
    const out: string[] = [];
    outlinesSvg(f.outlines!, "c", out);
    const bytes = out.join("\n").length;
    const ms = performance.now() - t0;
    console.log(`[outlineSvg] ${f.count} outlines, ${f.vertices} vertices kept: ${(bytes / 1e6).toFixed(1)} MB in ${ms.toFixed(0)} ms`);
    expect(f.count).toBe(30_000);
    expect(f.vertices).toBeLessThan(m.nVertices / 2);
    expect(bytes).toBeLessThan(6e6);
  });
});
