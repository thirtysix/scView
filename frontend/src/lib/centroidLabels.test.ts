import { describe, it, expect } from "vitest";
import { OrthographicView } from "@deck.gl/core";
import {
  LABELS_AUTO_MAX_GROUPS,
  centroidLabelLayers,
  compareLabelKey,
  groupCentroids,
  labelViewOf,
  labelsShown,
  placeLabels,
} from "./centroidLabels";
import { layerInView, splitLabelLayerId } from "./splitView";

describe("labelsShown", () => {
  it("is on by default up to 40 groups and off above", () => {
    expect(LABELS_AUTO_MAX_GROUPS).toBe(40);
    expect(labelsShown(null, "leiden", 40)).toBe(true);
    expect(labelsShown(null, "leiden", 41)).toBe(false);
    expect(labelsShown(null, "leiden", 0)).toBe(false);
  });

  it("follows the user's choice for that colouring only", () => {
    const off = { key: "cell_type", show: false };
    expect(labelsShown(off, "cell_type", 13)).toBe(false);
    expect(labelsShown(off, "leiden", 13)).toBe(true);
    expect(labelsShown({ key: "fine", show: true }, "fine", 80)).toBe(true);
    expect(compareLabelKey(["a", "b"])).not.toBe(compareLabelKey(["a"]));
  });
});

describe("groupCentroids", () => {
  // Group 0: three cells near (1, 1) and one outlier at (100, 100).
  // Group 1: two cells at (10, 0) and (12, 2). Group 2 has no cells.
  const positions = Float32Array.from([0, 0, 1, 1, 2, 2, 100, 100, 10, 0, 12, 2]);
  const codes = Int32Array.from([0, 0, 0, 0, 1, 1]);
  const labels = ["B", "NK", "T"];

  it("puts each label among its group's cells, so an outlier cell does not drag it", () => {
    const c = groupCentroids(positions, 2, codes, labels);
    expect(c).toEqual([
      { code: 0, label: "B", position: [1, 1], count: 4 },
      { code: 1, label: "NK", position: [11, 1], count: 2 },
    ]);
  });

  it("labels a scattered group where most of its cells are, not at its median", () => {
    // "T": a 20 x 20 blob over [0, 10]^2. "Mk": 25 cells strewn over the same
    // square (platelets stuck to T cells) and 15 in a tight clump at (20, 20).
    const xy: number[] = [];
    const code: number[] = [];
    const add = (x: number, y: number, c: number) => {
      xy.push(x, y);
      code.push(c);
    };
    for (let a = 0; a < 20; a++) for (let b = 0; b < 20; b++) add(a / 2, b / 2, 0);
    for (let a = 0; a < 5; a++) for (let b = 0; b < 5; b++) add(a * 2, b * 2, 1);
    for (let j = 0; j < 15; j++) add(20 + (j % 4) * 0.05, 20 + Math.floor(j / 4) * 0.05, 1);
    const [t, mk] = groupCentroids(Float32Array.from(xy), 2, Int32Array.from(code), ["T", "Mk"]);
    // The median of Mk's x (and y) is 8: inside the T blob.
    const mkX = xy.filter((_, i) => i % 2 === 0 && code[i / 2] === 1).sort((p, q) => p - q);
    expect(mkX[mkX.length >> 1]).toBeLessThan(10);
    expect(mk!.position[0]).toBeCloseTo(20.07, 1);
    expect(mk!.position[1]).toBeCloseTo(20.07, 1);
    // The uniform blob keeps its label in the middle.
    expect(t!.position[0]).toBeGreaterThan(3.5);
    expect(t!.position[0]).toBeLessThan(6.5);
    expect(t!.position[1]).toBeGreaterThan(3.5);
    expect(t!.position[1]).toBeLessThan(6.5);
  });

  it("labels only groups with drawn cells (a sample, or one split panel)", () => {
    const c = groupCentroids(positions, 2, codes, labels, Uint32Array.from([4, 5]));
    expect(c.map((g) => g.label)).toEqual(["NK"]);
  });

  it("skips cells without a group or without finite coordinates", () => {
    const pos = Float32Array.from([0, 0, NaN, 5, 4, 4]);
    const c = groupCentroids(pos, 2, Int32Array.from([0, 0, -1]), ["B"]);
    expect(c).toEqual([{ code: 0, label: "B", position: [0, 0], count: 1 }]);
  });

  it("works in 3D", () => {
    const pos = Float32Array.from([0, 0, 0, 2, 2, 2, 4, 4, 9]);
    expect(groupCentroids(pos, 3, Int32Array.from([0, 0, 0]), ["B"])[0]!.position).toEqual([2, 2, 2]);
  });

  it("places 20 groups over 200k cells quickly, each on its own blob", () => {
    const n = 200_000;
    const k = 20;
    const pos = new Float32Array(n * 2);
    const code = new Int32Array(n);
    let seed = 7; // mulberry32
    const rand = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    const centre = (c: number) => [(c % 5) * 8, Math.floor(c / 5) * 8];
    for (let i = 0; i < n; i++) {
      const c = i % k;
      const [cx, cy] = centre(c);
      code[i] = c;
      pos[i * 2] = cx! + gauss();
      pos[i * 2 + 1] = cy! + gauss();
    }
    const names = Array.from({ length: k }, (_, c) => `group ${c}`);
    groupCentroids(pos, 2, code, names); // warm up the JIT
    const t0 = performance.now();
    const out = groupCentroids(pos, 2, code, names);
    const ms = performance.now() - t0;
    console.info(`groupCentroids: ${n} cells, ${k} groups in ${ms.toFixed(1)} ms`);
    expect(ms).toBeLessThan(1000);
    for (const g of out) {
      const [cx, cy] = centre(g.code);
      expect(Math.hypot(g.position[0]! - cx!, g.position[1]! - cy!)).toBeLessThan(1);
    }
  });
});

describe("placeLabels", () => {
  // World coordinates are the view's pixels; 6 px per character.
  const view = { width: 200, height: 100, project: (p: number[]) => [p[0]!, p[1]!] };
  const measure = (t: string) => t.length * 6;
  const at = (label: string, x: number, y: number, count = 10, code = 0) => ({ code, label, position: [x, y], count });

  it("leaves labels that fit where they are", () => {
    const placed = placeLabels([at("B", 50, 50), at("NK", 150, 50, 5, 1)], view, { measure });
    expect(placed.map((p) => [p.label, p.offset])).toEqual([
      ["B", [0, 0]],
      ["NK", [0, 0]],
    ]);
  });

  it("shifts a label inside the view when its text would cross an edge", () => {
    // 18 characters = 108 px + 4 px of halo: 56 px each side of the anchor.
    const placed = placeLabels(
      [at("CD4+ memory T cell", 10, 50), at("x", 100, 1, 10, 1), at("CD8+ T cell", 199, 99, 10, 2)],
      view,
      { measure },
    );
    const [left, top, corner] = placed.map((p) => p.offset);
    expect(left).toEqual([2 + 56 - 10, 0]);
    expect(top).toEqual([0, 2 + 8 - 1]);
    expect(corner).toEqual([200 - 2 - 35 - 199, 100 - 2 - 8 - 99]);
  });

  it("leaves out a label whose anchor is out of view", () => {
    const placed = placeLabels([at("B", -5, 50), at("NK", 100, 150, 10, 1), at("T", 100, 50, 10, 2)], view, { measure });
    expect(placed.map((p) => p.label)).toEqual(["T"]);
  });

  it("lets the bigger group keep its spot: the next moves a line up or down, the rest wait", () => {
    const placed = placeLabels(
      [at("Mk", 100, 50, 50, 0), at("CD8+ T", 100, 50, 900, 1), at("Act T", 102, 48, 400, 2), at("NK", 98, 50, 300, 3)],
      view,
      { measure },
    );
    expect(placed.map((p) => [p.label, p.offset])).toEqual([
      ["CD8+ T", [0, 0]],
      ["Act T", [0, -18]],
      ["NK", [0, 18]],
    ]);
  });

  it("places a highlighted group first, whatever its size", () => {
    const placed = placeLabels([at("CD8+ T", 100, 50, 900, 0), at("Mk", 100, 50, 50, 1)], view, {
      measure,
      highlight: "Mk",
    });
    expect(placed.map((p) => [p.label, p.offset])).toEqual([
      ["CD8+ T", [0, -18]],
      ["Mk", [0, 0]],
    ]);
  });

  it("makes room again as zooming in spreads the anchors", () => {
    const zoomed = (z: number) => ({ width: 200, height: 100, project: (p: number[]) => [100 + p[0]! * z, 50 + p[1]! * z] });
    const groups = [at("A", 0, 0, 3, 0), at("B", 0, 1, 2, 1), at("C", 0, 2, 1, 2)];
    expect(placeLabels(groups, zoomed(1), { measure }).map((p) => p.label)).toEqual(["A", "B"]);
    expect(placeLabels(groups, zoomed(20), { measure }).map((p) => p.label)).toEqual(["A", "B", "C"]);
  });

  it("projects through a deck.gl view (y up in the data, down on screen)", () => {
    const v = labelViewOf(new OrthographicView({ flipY: false }), 400, 300, { target: [0, 0, 0], zoom: 0 })!;
    expect(v.project([10, 10]).slice(0, 2)).toEqual([210, 140]);
    // An anchor 5 px from the left edge is pushed right; one at the top, down.
    const [l, t] = placeLabels([at("Monocyte", -195, 0), at("NK", 0, 148, 10, 1)], v, { measure });
    expect(l!.offset[0]).toBeGreaterThan(0);
    expect(t!.offset[1]).toBeGreaterThan(0);
    expect(labelViewOf(new OrthographicView({ flipY: false }), 0, 300, { target: [0, 0, 0], zoom: 0 })).toBeNull();
  });
});

describe("centroidLabelLayers", () => {
  const centroids = [
    { code: 0, label: "CD14 Mono", position: [1, 1], count: 10 },
    { code: 1, label: "x".repeat(40), position: [5, 5], count: 3 },
  ];

  it("draws every label with a halo that suits the background", () => {
    const [light] = centroidLabelLayers("labels", centroids, { dark: false });
    expect(light!.id).toBe("labels");
    expect(light!.props.data).toEqual(centroids);
    expect(light!.props.outlineColor).toEqual([255, 255, 255, 235]);
    expect(light!.props.fontSettings).toMatchObject({ sdf: true });
    expect(light!.props.outlineWidth).toBeGreaterThan(0);
    const getText = light!.props.getText as (d: (typeof centroids)[number]) => string;
    expect(getText(centroids[1]!)).toHaveLength(32);
    const [dark] = centroidLabelLayers("labels", centroids, { dark: true });
    expect(dark!.props.outlineColor).toEqual([15, 23, 42, 235]);
  });

  it("draws a placed label shifted by its offset", () => {
    const [layer] = centroidLabelLayers("labels", [{ ...centroids[0]!, offset: [12, -18] }, centroids[1]!], { dark: false });
    type Drawn = (typeof centroids)[number] & { offset?: [number, number] };
    const getOffset = layer!.props.getPixelOffset as (d: Drawn) => [number, number];
    expect(getOffset({ ...centroids[0]!, offset: [12, -18] })).toEqual([12, -18]);
    expect(getOffset(centroids[1]!)).toEqual([0, 0]);
  });

  it("fades the other labels when a group is highlighted", () => {
    const layers = centroidLabelLayers("labels", centroids, { dark: false, highlight: "CD14 Mono" });
    expect(layers.map((l) => [l.id, l.props.opacity, (l.props.data as unknown[]).length])).toEqual([
      ["labels", 0.3, 1],
      ["labels-hl", 1, 1],
    ]);
    // A highlight from another column changes nothing.
    expect(centroidLabelLayers("labels", centroids, { dark: false, highlight: "NK" })).toHaveLength(1);
  });

  it("routes a split panel's label layers to that panel's view only", () => {
    expect(layerInView(splitLabelLayerId(1), "split-1")).toBe(true);
    expect(layerInView(`${splitLabelLayerId(1)}-hl`, "split-1")).toBe(true);
    expect(layerInView(splitLabelLayerId(1), "split-0")).toBe(false);
  });
});
