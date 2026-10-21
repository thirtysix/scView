import { describe, it, expect } from "vitest";
import { fakeResult } from "@/test/communicationFixture";
import {
  circleLayout,
  dotPlotData,
  filterRows,
  negLog10,
  pairMatrix,
  roleBars,
  sortRows,
  toRows,
} from "./communication";

describe("communication rows", () => {
  it("turns the column table into rows and filters by sender, receiver and gene", () => {
    const rows = toRows(fakeResult());
    expect(rows).toHaveLength(3);
    expect(rows[1]).toMatchObject({ source: "NK", receptor_complex: "IFNGR1_IFNGR2", cellphone_pvals: 0.01 });
    expect(filterRows(rows, { source: "NK", target: "", query: "" }).map((r) => r.ligand)).toEqual(["IFNG"]);
    expect(filterRows(rows, { source: "", target: "B", query: "" }).map((r) => r.ligand)).toEqual(["CD70"]);
    expect(filterRows(rows, { source: "", target: "", query: "ifngr2" }).map((r) => r.ligand)).toEqual(["IFNG"]);
  });

  it("sorts either way, keeping missing values last", () => {
    const rows = toRows(fakeResult());
    expect(sortRows(rows, "lr_means", false).map((r) => r.ligand)).toEqual(["CXCL10", "IFNG", "CD70"]);
    expect(sortRows(rows, "cellphone_pvals", true).map((r) => r.ligand)).toEqual(["CXCL10", "IFNG", "CD70"]);
    expect(sortRows(rows, "cellphone_pvals", false).map((r) => r.ligand)).toEqual(["IFNG", "CXCL10", "CD70"]);
    expect(sortRows(rows, "source", true).map((r) => r.source)).toEqual(["B", "Mono", "NK"]);
  });

  it("sizes dots by specificity and colours them by magnitude, strongest pair on top", () => {
    const dot = dotPlotData(toRows(fakeResult()), 2);
    expect(dot.y).toEqual(["IFNG → IFNGR1_IFNGR2", "CXCL10 → CXCR3"]);
    expect(dot.x).toEqual(["Mono → NK", "NK → Mono"]);
    const top = dot.points.find((p) => p.y === "CXCL10 → CXCR3")!;
    expect(top.size).toBeCloseTo(4);
    expect(top.color).toBeCloseTo(6);
    expect(negLog10(0)).toBe(300);
    expect(negLog10(null)).toBe(0);
  });
});

describe("between groups", () => {
  it("draws one edge per directed pair and reports autocrine on the node", () => {
    const m = [
      [5, 2, 0],
      [0, 0, 4],
      [1, 0, 0],
    ];
    const lay = circleLayout(["a", "b", "c"], m);
    expect(lay.nodes.map((n) => [n.autocrine, n.out, n.in])).toEqual([
      [5, 2, 1],
      [0, 4, 2],
      [0, 1, 4],
    ]);
    expect(lay.edges.map((e) => [e.source, e.target])).toEqual([
      [2, 0],
      [0, 1],
      [1, 2],
    ]);
    const strongest = lay.edges[2]!;
    expect(strongest.width).toBe(8);
    expect(strongest.x).toHaveLength(17);
    // The first node sits at the top.
    expect(lay.nodes[0]!.y).toBeCloseTo(1);
    expect(circleLayout(["a", "b"], [[3, 0], [0, 0]]).edges).toEqual([]);
  });

  it("weighs pairs and roles by count or summed strength", () => {
    const res = fakeResult();
    expect(pairMatrix(res, "count")).toBe(res.pair_counts);
    expect(pairMatrix(res, "strength")[2]).toEqual([0, 2.5, 0]);
    expect(roleBars(res, "count")).toEqual([
      { group: "B", out: 0, in: 0 },
      { group: "NK", out: 1, in: 1 },
      { group: "Mono", out: 1, in: 1 },
    ]);
    expect(roleBars(res, "strength")[2]).toEqual({ group: "Mono", out: 2.5, in: 1.2 });
  });
});
