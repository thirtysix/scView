import { describe, it, expect } from "vitest";
import { donorBoxFigure, donorCaption, donorsPerCondition, fewDonorsCaution, type DonorPoint } from "./donorPlot";

const pt = (group: string, condition: string, sample: string, value: number): DonorPoint => ({
  group, condition, sample, n_cells: 50, value,
});

// Three donors in CTRL, two in STIM; pDC has no STIM sample.
const POINTS: DonorPoint[] = [
  pt("B", "CTRL", "d1", 1), pt("B", "CTRL", "d2", 2), pt("B", "CTRL", "d3", 3),
  pt("B", "STIM", "d1", 4), pt("B", "STIM", "d2", 5),
  pt("pDC", "CTRL", "d1", 0.1),
];
const colorOf = (k: string, kind: "condition" | "group") => `${kind}:${k}`;
const opts = { groups: ["B", "pDC"], conditions: ["CTRL", "STIM"], colorOf, format: (v: number) => v.toFixed(1) };

describe("donor counts", () => {
  it("counts distinct donors per condition and captions them", () => {
    const n = donorsPerCondition(POINTS, ["CTRL", "STIM"]);
    expect(n).toEqual({ CTRL: 3, STIM: 2 });
    expect(donorCaption(n, ["CTRL", "STIM"])).toBe("n = 3: CTRL 3, STIM 2");
    expect(donorCaption({ "": 8 }, [""])).toBe("n = 8");
  });

  it("cautions only for conditions under three donors", () => {
    expect(fewDonorsCaution({ CTRL: 3, STIM: 2 }, ["CTRL", "STIM"])).toContain("Fewer than 3 donors (STIM: 2)");
    expect(fewDonorsCaution({ CTRL: 8, STIM: 8 }, ["CTRL", "STIM"])).toBeNull();
    expect(fewDonorsCaution({ "": 1 }, [""])).toContain("(all cells: 1)");
  });
});

describe("donorBoxFigure", () => {
  it("groups one box per condition on one axis, every point drawn", () => {
    const { data, layout } = donorBoxFigure(POINTS, { ...opts, facet: false });
    expect(data.map((t) => t.name)).toEqual(["CTRL", "STIM"]);
    expect(data[0]!.x).toEqual(["B", "B", "B", "pDC"]);
    expect(data[0]!.y).toEqual([1, 2, 3, 0.1]);
    expect(data[0]!.boxpoints).toBe("all");
    expect((data[1]!.marker as { color: string }).color).toBe("condition:STIM");
    expect((data[0]!.text as string[])[0]).toBe("d1: 1.0 (50 cells)");
    expect(layout.boxmode).toBe("group");
    expect((layout.xaxis as { categoryarray: string[] }).categoryarray).toEqual(["B", "pDC"]);
  });

  it("colours unsplit boxes by group", () => {
    const pts = POINTS.map((p) => ({ ...p, condition: "" }));
    const { data } = donorBoxFigure(pts, { ...opts, conditions: [""], facet: false });
    expect(data.map((t) => t.name)).toEqual(["B", "pDC"]);
    expect((data[0]!.marker as { color: string }).color).toBe("group:B");
  });

  it("gives each group its own panel and axis, titled by annotation", () => {
    const { data, layout, height } = donorBoxFigure(POINTS, { ...opts, facet: true });
    // B: CTRL + STIM, pDC: CTRL only (no empty box)
    expect(data.map((t) => [t.name, t.xaxis, t.yaxis])).toEqual([
      ["CTRL", "x", "y"],
      ["STIM", "x", "y"],
      ["CTRL", "x2", "y2"],
    ]);
    // The legend lists each condition once.
    expect(data.map((t) => t.showlegend)).toEqual([true, true, false]);
    expect(layout.grid).toMatchObject({ rows: 1, columns: 2, pattern: "independent" });
    expect((layout.annotations as { text: string; xref: string }[]).map((a) => [a.text, a.xref])).toEqual([
      ["B", "x domain"],
      ["pDC", "x2 domain"],
    ]);
    expect(height).toBe(210);
  });
});
