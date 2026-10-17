import { describe, it, expect } from "vitest";
import type { PagaRecord } from "@/api/trajectory";
import { branchEnds, nodeRadius, pagaLayers, treePath, visibleEdges } from "./paga";

// paul15-like: progenitors (MEP) between an erythroid and a myeloid branch.
const RECORD: PagaRecord = {
  groupby: "cluster",
  groups: ["MEP", "Ery1", "Ery2", "GMP", "Mono", "DC"],
  sizes: [100, 300, 200, 150, 400, 25],
  edges: [
    [0, 1, 0.9],
    [1, 2, 0.8],
    [0, 3, 0.7],
    [3, 4, 0.6],
    [1, 3, 0.05],
    [2, 4, 0.02],
  ],
  tree: [
    [0, 1],
    [1, 2],
    [0, 3],
    [3, 4],
  ],
  graph: "existing neighbour graph (use_rep=X_pca, k=15)",
  n_cells: 1175,
};

describe("PAGA edges and nodes", () => {
  it("keeps edges at or above the threshold, strongest first, and marks tree edges", () => {
    expect(visibleEdges(RECORD, 0.1).map((e) => [e.i, e.j])).toEqual([[0, 1], [1, 2], [0, 3], [3, 4]]);
    expect(visibleEdges(RECORD, 0).length).toBe(6);
    expect(visibleEdges(RECORD, 0.05).find((e) => e.i === 1 && e.j === 3)?.tree).toBe(false);
    expect(visibleEdges(RECORD, 0, true).every((e) => e.tree)).toBe(true);
    expect(visibleEdges(RECORD, 0.65, true).map((e) => e.weight)).toEqual([0.9, 0.8, 0.7]);
  });

  it("sizes nodes by area between 5 and 20 px", () => {
    expect(nodeRadius(400, 400)).toBe(20);
    expect(nodeRadius(0, 400)).toBe(5);
    expect(nodeRadius(100, 400)).toBeCloseTo(12.5);
    expect(nodeRadius(10, 0)).toBe(5);
  });
});

describe("branches along the spanning tree", () => {
  it("walks the tree from the root group to an end group", () => {
    expect(treePath(RECORD, "MEP", "Ery2")).toEqual(["MEP", "Ery1", "Ery2"]);
    expect(treePath(RECORD, "Ery2", "Mono")).toEqual(["Ery2", "Ery1", "MEP", "GMP", "Mono"]);
    expect(treePath(RECORD, "MEP", "MEP")).toEqual(["MEP"]);
    // DC has no tree edge; unknown groups have no path.
    expect(treePath(RECORD, "MEP", "DC")).toBeNull();
    expect(treePath(RECORD, "nope", "Mono")).toBeNull();
  });

  it("offers every other group the tree reaches as a branch end", () => {
    expect(branchEnds(RECORD, "MEP")).toEqual(["Ery1", "Ery2", "GMP", "Mono"]);
  });
});

describe("pagaLayers", () => {
  type AnyLayer = {
    id: string;
    props: { data: unknown[]; getColor?: unknown; getWidth?: unknown; pickable?: boolean };
  };
  type Edge = { i: number; j: number; weight: number; onPath: boolean };
  const positions = [[0, 0], [1, 0], [2, 0], [0, 1], null, [5, 5]];

  it("draws edges between placed groups (over a halo), nodes and their names, none pickable", () => {
    const layers = pagaLayers({ record: RECORD, positions, threshold: 0.1 }) as unknown as AnyLayer[];
    expect(layers.map((l) => l.id)).toEqual(["paga-edges-halo", "paga-edges", "paga-nodes", "paga-labels"]);
    const [halo, edges, nodes, labels] = layers;
    // Mono has no position: its edge (GMP-Mono) and node are left out.
    expect(edges!.props.data.length).toBe(3);
    expect(halo!.props.data).toBe(edges!.props.data);
    expect((nodes!.props.data as { group: string }[]).map((n) => n.group)).toEqual(["MEP", "Ery1", "Ery2", "GMP", "DC"]);
    expect((labels!.props.data as { label: string; offset: number[] }[])[0]).toMatchObject({ label: "MEP" });
    expect(layers.every((l) => l.props.pickable === false)).toBe(true);
  });

  it("sizes edges relative to the strongest one drawn, so weak graphs still read", () => {
    const weak: PagaRecord = { ...RECORD, edges: RECORD.edges.map(([i, j, w]) => [i, j, w / 5]) };
    const layers = pagaLayers({ record: weak, positions, threshold: 0 }) as unknown as AnyLayer[];
    const width = layers[1]!.props.getWidth as (d: Edge) => number;
    expect(width({ i: 0, j: 1, weight: 0.18, onPath: false })).toBeCloseTo(7); // the strongest (0.9 / 5)
    expect(width({ i: 0, j: 1, weight: 0.09, onPath: false })).toBeCloseTo(4);
    const haloWidth = layers[0]!.props.getWidth as (d: Edge) => number;
    expect(haloWidth({ i: 0, j: 1, weight: 0.18, onPath: false })).toBeCloseTo(10);
  });

  it("draws the chosen branch in amber", () => {
    const layers = pagaLayers({
      record: RECORD,
      positions,
      threshold: 0.1,
      path: ["MEP", "Ery1", "Ery2"],
    }) as unknown as AnyLayer[];
    const edges = layers[1]!.props.data as Edge[];
    expect(edges.filter((e) => e.onPath).map((e) => [e.i, e.j])).toEqual([[0, 1], [1, 2]]);
    const colour = layers[1]!.props.getColor as (d: Edge) => number[];
    expect(colour({ i: 0, j: 1, onPath: true, weight: 0.5 }).slice(0, 3)).toEqual([217, 119, 6]);
    expect(colour({ i: 0, j: 3, onPath: false, weight: 0.5 }).slice(0, 3)).toEqual([30, 41, 59]);
    const nodes = layers[2]!.props.data as { group: string; onPath: boolean }[];
    expect(nodes.filter((n) => n.onPath).map((n) => n.group)).toEqual(["MEP", "Ery1", "Ery2"]);
  });
});
