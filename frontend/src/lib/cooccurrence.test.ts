import { describe, expect, it } from "vitest";
import type { CoOccurrencePlan, CoOccurrenceResult } from "@/api/spatialDomains";
import { csvHeaders, csvRows, curvesOf, distanceSource, firstUsable, matrixAt, reachOf, scoreScale, unitText } from "./cooccurrence";

const result: CoOccurrenceResult = {
  available: true,
  columns: ["c"],
  column: "c",
  labels: ["a", "b"],
  distances: [10, 20],
  distance_unit: "um",
  um_per_unit: 1,
  occ: [
    [
      [1.5, 1.2],
      [0, 0.8],
    ],
    [
      [0, 0.8],
      [2, 1.1],
    ],
  ],
  count: [
    [
      [40, 90],
      [0, 30],
    ],
    [
      [0, 30],
      [8, 20],
    ],
  ],
};

describe("co-occurrence slices", () => {
  it("gives one curve per label for the label asked", () => {
    const curves = curvesOf(result, "b");
    expect(curves.map((c) => c.label)).toEqual(["a", "b"]);
    expect(curves[1]).toEqual({ label: "b", distances: [10, 20], scores: [2, 1.1], pairs: [8, 20] });
    expect(curvesOf(result, "nope")).toEqual([]);
  });

  it("blanks a pair of labels with no pair of observations at that distance", () => {
    expect(matrixAt(result, 0)).toEqual([
      [1.5, null],
      [null, 2],
    ]);
    expect(matrixAt(result, 1)).toEqual([
      [1.2, 0.8],
      [0.8, 1.1],
    ]);
  });

  it("colours from 0 to the largest score, with the neutral colour at a score of 1", () => {
    const low = scoreScale([[1, 0.5]]); // nothing above 1: still a real range around 1
    expect([low.zmin, low.zmax]).toEqual([0, 2]);
    expect(low.colorscale[1]?.[0]).toBe(0.5);
    const high = scoreScale([
      [14, null],
      [null, 2],
    ]);
    expect(high.zmax).toBe(14);
    expect(high.colorscale[1]?.[0]).toBeCloseTo(1 / 14);
  });

  it("starts the heatmap at the shortest distance with enough pairs", () => {
    expect(firstUsable(result)).toBe(1); // 2 cells with 30 or more at the first, 3 at the second
    expect(firstUsable({ ...result, count: [[[0, 0], [0, 0]], [[0, 0], [0, 0]]] })).toBe(1); // none: the last
  });

  it("writes every score with its pairs in the CSV", () => {
    const rows = csvRows(result);
    expect(rows).toHaveLength(8);
    expect(rows[0]).toEqual(["a", "a", 10, 1.5, 40]);
    expect(rows[7]).toEqual(["b", "b", 20, 1.1, 20]);
    expect(csvHeaders(result)).toEqual(["label", "other_label", "distance_um", "score", "ordered_pairs"]);
    expect(csvHeaders({ ...result, distance_unit: "position units" })[2]).toBe("distance_units");
  });
});

describe("co-occurrence units", () => {
  it("says which units and where they came from", () => {
    expect(unitText("um")).toBe("µm");
    expect(unitText("position units")).toBe("position units");
    expect(distanceSource(result)).toBe("Distances are in microns.");
    expect(distanceSource({ distance_unit: "um", um_per_unit: 0.2738 })).toContain("1 pixel = 0.274 µm)");
    expect(distanceSource({ distance_unit: "um", um_per_unit: 0.73, distance_source: "the 100 um between spot centres" })).toContain(
      "1 pixel = 0.730 µm, from the 100 µm between spot centres)",
    );
    expect(distanceSource({ distance_unit: "position units", um_per_unit: null })).toContain("not known");
  });

  it("shows the default reach in the axis's unit", () => {
    const plan = { thresholds: [10, 100, 1000], um_per_unit: 0.5 } as CoOccurrencePlan;
    expect(reachOf(plan)).toBe(500);
  });
});
