import { describe, it, expect } from "vitest";
import type { Data, Layout } from "plotly.js";
import { plotAriaLabel, plotTable, TABLE_MAX_POINTS } from "./plotA11y";

const layout: Partial<Layout> = {
  title: { text: "<b>Cells</b> per group" },
  xaxis: { title: { text: "Group" } },
  yaxis: { title: "Cells" as unknown as Partial<Layout["yaxis"]["title"]> },
};

describe("plotAriaLabel", () => {
  it("names the kind, the title and the axes, without HTML", () => {
    const data = [{ type: "bar", x: ["B", "NK"], y: [3, 4] }] as Data[];
    expect(plotAriaLabel(data, layout)).toBe("Bar chart: Cells per group. x axis: Group, y axis: Cells.");
  });

  it("calls mixed traces a chart, and copes with no title", () => {
    const data = [{ type: "bar" }, { type: "scatter" }] as Data[];
    expect(plotAriaLabel(data, {})).toBe("Chart.");
    expect(plotAriaLabel([{ type: "violin" }] as Data[], undefined)).toBe("Violin plot.");
  });
});

describe("plotTable", () => {
  it("lists a small bar or scatter figure's points, one series column when there are several", () => {
    const one = plotTable([{ type: "bar", x: ["B", "NK"], y: [3, 4.123456] }] as Data[], layout)!;
    expect(one.columns).toEqual(["Group", "Cells"]);
    expect(one.rows).toEqual([["B", 3], ["NK", 4.123]]);
    expect(one.caption).toBe("Bar chart: Cells per group. x axis: Group, y axis: Cells, as a table");

    const two = plotTable(
      [
        { type: "scatter", name: "CTRL", x: [1], y: [2] },
        { type: "scatter", name: "STIM", x: [1], y: [5] },
      ] as Data[],
      {},
    )!;
    expect(two.columns).toEqual(["Series", "x", "y"]);
    expect(two.rows).toEqual([["CTRL", 1, 2], ["STIM", 1, 5]]);
  });

  it("gives none for a big figure, other trace kinds, or ragged x/y", () => {
    const big = Array.from({ length: TABLE_MAX_POINTS + 1 }, (_, i) => i);
    expect(plotTable([{ type: "scatter", x: big, y: big }] as Data[], {})).toBeNull();
    expect(plotTable([{ type: "heatmap", z: [[1]] }] as Data[], {})).toBeNull();
    expect(plotTable([{ type: "bar", x: [1, 2], y: [1] }] as Data[], {})).toBeNull();
    expect(plotTable([], {})).toBeNull();
  });
});
