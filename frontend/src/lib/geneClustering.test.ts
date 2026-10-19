import { describe, it, expect } from "vitest";
import { averageLinkage, clusterGenes, correlationDistance, leafOrder } from "./geneClustering";

// Four genes across three groups. g1 is g0 doubled (r = 1), g2 is g0 reversed
// (r = -1), g3 = [1, 3, 2] has r = 0.5 with g0 and g1 and r = -0.5 with g2
// (centred: [-1, 1, 0] against [-1, 0, 1] and [1, 0, -1], norms sqrt 2).
const G0 = [1, 2, 3];
const G1 = [2, 4, 6];
const G2 = [3, 2, 1];
const G3 = [1, 3, 2];

describe("correlationDistance", () => {
  it("is 1 - Pearson r", () => {
    expect(correlationDistance(G0, G1)).toBeCloseTo(0, 12);
    expect(correlationDistance(G0, G2)).toBeCloseTo(2, 12);
    expect(correlationDistance(G0, G3)).toBeCloseTo(0.5, 12);
    expect(correlationDistance(G2, G3)).toBeCloseTo(1.5, 12);
  });

  it("treats a flat profile as uncorrelated (distance 1), never NaN", () => {
    expect(correlationDistance([2, 2, 2], G0)).toBe(1);
    expect(correlationDistance([0, 0, 0], [0, 0, 0])).toBe(1);
    expect(correlationDistance([1], [2])).toBe(1);
  });
});

describe("averageLinkage", () => {
  it("merges the closest pair, then averages distances by cluster size", () => {
    // d01 = 0, d02 = 2, d03 = 0.5, d12 = 2, d13 = 0.5, d23 = 1.5.
    // 1) g0 + g1 at 0 -> node 4; d(4, g2) = (2 + 2) / 2 = 2, d(4, g3) = (0.5 + 0.5) / 2 = 0.5.
    // 2) node 4 + g3 at 0.5 -> node 5; d(5, g2) = (2 * 2 + 1 * 1.5) / 3 = 11 / 6.
    // 3) node 5 + g2 at 11 / 6.
    const { tree, order } = clusterGenes([G0, G1, G2, G3]);
    expect(tree.map((r) => [r[0], r[1], r[3]])).toEqual([
      [0, 1, 2],
      [4, 3, 3],
      [5, 2, 4],
    ]);
    expect(tree[0]![2]).toBeCloseTo(0, 12);
    expect(tree[1]![2]).toBeCloseTo(0.5, 12);
    expect(tree[2]![2]).toBeCloseTo(11 / 6, 12);
    // left to right: the g0/g1 pair, then g3, then the anti-correlated g2
    expect(order).toEqual([0, 1, 3, 2]);
  });

  it("orders by the profiles, not by where the genes were listed", () => {
    // The same genes listed as g2, g0, g3, g1: the outlier stays apart, the
    // identical pair together, and the pair keeps its listed order.
    const { order } = clusterGenes([G2, G0, G3, G1]);
    expect(order.map((i) => ["g2", "g0", "g3", "g1"][i])).toEqual(["g2", "g0", "g1", "g3"]);
  });

  it("breaks ties by input order, so equal profiles keep their place", () => {
    const tree = averageLinkage([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expect(tree.map((r) => [r[0], r[1]])).toEqual([
      [0, 1],
      [3, 2],
    ]);
    expect(leafOrder(tree, 3)).toEqual([0, 1, 2]);
  });

  it("handles one gene and none", () => {
    expect(clusterGenes([G0])).toEqual({ tree: [], order: [0] });
    expect(clusterGenes([])).toEqual({ tree: [], order: [] });
  });
});
