import { describe, expect, it } from "vitest";
import type { ClusterTree, ClusterTreeEdge, ClusterTreeNode } from "@/api/clustering";
import { edgeBins, layoutTree, nodeHover, nodeKey, treeTraces } from "./clusterTree";

const node = (resolution: number, cluster: string, size: number, extra: Partial<ClusterTreeNode> = {}): ClusterTreeNode => ({
  resolution,
  cluster,
  size,
  sc3_stability: 0.5,
  n_parents: 1,
  mixed: false,
  ...extra,
});
const edge = (from: string, to: string, count: number, in_prop: number): ClusterTreeEdge => ({
  from_resolution: 0.2,
  from,
  to_resolution: 0.5,
  to,
  count,
  in_prop,
});

// 0.2: 0 (60) 1 (40). 0.5: "0" comes from 1, "1" and "2" from 0: drawn by name they'd cross.
const tree: ClusterTree = {
  prop_filter: 0.1,
  mixed_prop: 0.2,
  nodes: [node(0.2, "0", 60), node(0.2, "1", 40), node(0.5, "0", 40), node(0.5, "1", 35), node(0.5, "2", 25, { mixed: true })],
  edges: [edge("1", "0", 40, 1), edge("0", "1", 35, 1), edge("0", "2", 25, 1)],
  n_mixed: 1,
};

describe("layoutTree", () => {
  it("puts each resolution on its own row and children under their parents", () => {
    const l = layoutTree(tree);
    expect(l.resolutions).toEqual([0.2, 0.5]);
    const x = (r: number, c: string) => l.byKey.get(nodeKey(r, c))!.x;
    expect(l.byKey.get(nodeKey(0.5, "0"))!.level).toBe(1);
    // Children of parent 0 sit on parent 0's side; child of parent 1 on the other.
    expect(Math.sign(x(0.5, "1") - x(0.5, "0"))).toBe(Math.sign(x(0.2, "0") - x(0.2, "1")));
    expect(Math.sign(x(0.5, "2") - x(0.5, "0"))).toBe(Math.sign(x(0.2, "0") - x(0.2, "1")));
    expect(l.halfWidth).toBe(1);
  });

  it("orders the first row by numeric cluster name", () => {
    const t: ClusterTree = { ...tree, nodes: [node(0.2, "10", 5), node(0.2, "2", 50), node(0.2, "B", 3)], edges: [] };
    const l = layoutTree(t);
    expect(l.nodes.map((n) => n.cluster)).toEqual(["2", "10", "B"]);
  });
});

describe("treeTraces", () => {
  it("draws edges in a few width/opacity groups, then one node trace", () => {
    const l = layoutTree(tree);
    const traces = treeTraces(tree, l) as unknown as { mode: string; x: (number | null)[]; marker?: { line: { color: string[] } } }[];
    const nodes = traces[traces.length - 1]!;
    expect(nodes.mode).toBe("markers+text");
    expect(nodes.marker!.line.color.filter((c) => c === "#dc2626")).toHaveLength(1);
    const segments = traces.slice(0, -1).reduce((s, t) => s + t.x.filter((v) => v === null).length, 0);
    expect(segments).toBe(3);
  });
});

describe("edgeBins / nodeHover", () => {
  it("widens edges with their cells and darkens them with the child's share", () => {
    expect(edgeBins(edge("0", "1", 100, 0.9), 100)).toEqual({ width: 7, alpha: 0.85 });
    expect(edgeBins(edge("0", "1", 1, 0.15), 100).alpha).toBe(0.25);
    expect(edgeBins(edge("0", "1", 1, 0.15), 100).width).toBeLessThan(7);
  });

  it("says where a node's cells came from", () => {
    const h = nodeHover(node(0.5, "2", 25, { mixed: true }), [edge("0", "2", 20, 0.8), edge("1", "2", 5, 0.2)]);
    expect(h).toContain("From: 0 (80%), 1 (20%)");
    expect(h).toContain("more than one parent");
  });
});
