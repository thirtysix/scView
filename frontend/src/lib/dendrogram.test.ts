import { describe, it, expect } from "vitest";
import { columnDendrogram, dendrogramRowOrder, dendrogramSegments, rowDendrogram } from "./dendrogram";

// Groups 0..3. A (1) and B (2) merge at 1; C (0) and D (3) at 2; the root at 10.
// scipy: row 0 -> node 4, row 1 -> node 5, row 2 joins 4 and 5.
const TREE = [
  [1, 2, 1, 2],
  [0, 3, 2, 2],
  [4, 5, 10, 4],
];
const ORDER = [1, 2, 0, 3]; // leaves_list of TREE

describe("dendrogramSegments", () => {
  it("draws each merge as a bracket from its children up to its height", () => {
    const lines = dendrogramSegments(TREE, ORDER)!;
    expect(lines.n).toBe(4);
    expect(lines.maxHeight).toBe(10);
    // A at position 0, B at 1: up from 0 to height 1, across, down to B.
    expect(lines.pos.slice(0, 5)).toEqual([0, 0, 1, 1, null]);
    expect(lines.height.slice(0, 5)).toEqual([0, 1, 1, 0, null]);
    // C at 2, D at 3, joined at 2.
    expect(lines.pos.slice(5, 10)).toEqual([2, 2, 3, 3, null]);
    expect(lines.height.slice(5, 10)).toEqual([0, 2, 2, 0, null]);
    // The root joins the midpoints (0.5 and 2.5) from their heights (1 and 2) at 10.
    expect(lines.pos.slice(10)).toEqual([0.5, 0.5, 2.5, 2.5, null]);
    expect(lines.height.slice(10)).toEqual([1, 10, 10, 2, null]);
  });

  it("puts leaves where the order puts them", () => {
    const lines = dendrogramSegments(TREE, [3, 0, 2, 1])!;
    expect(lines.pos.slice(0, 4)).toEqual([3, 3, 2, 2]); // A at 3, B at 2
  });

  it("returns null for a tree that does not match the order", () => {
    expect(dendrogramSegments(TREE, [0, 1, 2])).toBeNull();
    expect(dendrogramSegments([], [0])).toBeNull();
    expect(dendrogramSegments(TREE, [0, 0, 1, 2])).toBeNull();
    expect(dendrogramSegments([[0, 9, 1, 2], [4, 1, 2, 3], [5, 2, 3, 4]], ORDER)).toBeNull();
  });
});

describe("dendrogramRowOrder", () => {
  it("orders rows as the tree, keeping rows the tree lacks at the end", () => {
    expect(dendrogramRowOrder(["C", "A", "B", "D"], ORDER, ["A", "B", "C", "D", "E"])).toEqual([
      "A", "B", "C", "D", "E",
    ]);
    expect(dendrogramRowOrder(["C", "A", "B", "D"], [3, 0, 2, 1], ["A", "B", "C"])).toEqual(["C", "B", "A"]);
  });
});

describe("rowDendrogram", () => {
  it("aligns its leaves with category rows drawn top to bottom", () => {
    const { trace, layout } = rowDendrogram(dendrogramSegments(TREE, ORDER)!, [0.85, 1]);
    expect(trace).toMatchObject({ xaxis: "x2", yaxis: "y2", mode: "lines" });
    expect(layout.yaxis2!.range).toEqual([3.5, -0.5]);
    expect(layout.xaxis2!.domain).toEqual([0.85, 1]);
  });

  it("shares the plot's y domain when the plot does not fill the height", () => {
    const { layout } = rowDendrogram(dendrogramSegments(TREE, ORDER)!, [0.9, 1], [0, 0.8]);
    expect(layout.yaxis2!.domain).toEqual([0, 0.8]);
    expect(rowDendrogram(dendrogramSegments(TREE, ORDER)!, [0.9, 1]).layout.yaxis2!.domain).toBeUndefined();
  });
});

describe("columnDendrogram", () => {
  it("puts the leaves on the columns below and the root at the top", () => {
    const { trace, layout } = columnDendrogram(dendrogramSegments(TREE, ORDER)!, [0, 0.88], [0.85, 1]);
    expect(trace).toMatchObject({ xaxis: "x3", yaxis: "y3", mode: "lines" });
    // x carries the leaf positions (columns 0..3), y the merge heights.
    expect((trace as { x: unknown[] }).x.slice(0, 5)).toEqual([0, 0, 1, 1, null]);
    expect((trace as { y: unknown[] }).y.slice(0, 5)).toEqual([0, 1, 1, 0, null]);
    expect(layout.xaxis3!.range).toEqual([-0.5, 3.5]);
    expect(layout.xaxis3!.domain).toEqual([0, 0.88]);
    expect(layout.yaxis3!.domain).toEqual([0.85, 1]);
    expect((layout.yaxis3!.range as number[])[0]).toBe(0); // leaves at the bottom of the band
  });
});
