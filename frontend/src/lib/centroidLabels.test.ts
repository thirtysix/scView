import { describe, it, expect } from "vitest";
import {
  LABELS_AUTO_MAX_GROUPS,
  centroidLabelLayers,
  compareLabelKey,
  groupCentroids,
  labelsShown,
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

  it("puts each label at the median of its group, so an outlier cell does not drag it", () => {
    const c = groupCentroids(positions, 2, codes, labels);
    expect(c).toEqual([
      { code: 0, label: "B", position: [1.5, 1.5], count: 4 },
      { code: 1, label: "NK", position: [11, 1], count: 2 },
    ]);
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
