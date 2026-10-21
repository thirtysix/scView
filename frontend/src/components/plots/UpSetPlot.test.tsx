import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { UpSetPlot } from "./UpSetPlot";

const SETS = [
  { name: "CD14 Mono: STIM vs CTRL (Wilcoxon)", n: 1800, color: "rgb(230,159,0)", title: "CD14 Mono: 1,800 genes" },
  { name: "B: STIM vs CTRL (Wilcoxon)", n: 320, color: "rgb(86,180,233)" },
];
const COLS = [
  { sets: [0], n: 1400, title: "1,400 genes in only CD14 Mono" },
  { sets: [0, 1], n: 190, title: "190 genes in all 2 sets" },
  { sets: [1], n: 30, title: "30 genes in only B" },
];

const renderPlot = (selected: number | null = null, onSelect = vi.fn()) => {
  render(<UpSetPlot sets={SETS} columns={COLS} selected={selected} onSelect={onSelect} label="UpSet plot" />);
  return onSelect;
};

describe("UpSetPlot", () => {
  it("draws a bar and a column of dots per intersection, a bar per set", () => {
    renderPlot();
    expect(screen.getByRole("group", { name: "UpSet plot" })).toBeInTheDocument();
    expect(screen.getAllByTestId("upset-bar")).toHaveLength(3);
    expect(screen.getAllByTestId("upset-set-bar")).toHaveLength(2);
    expect(screen.getAllByTestId("upset-dot-on")).toHaveLength(4); // 1 + 2 + 1
    expect(screen.getAllByTestId("upset-dot-off")).toHaveLength(2);
    const heights = screen.getAllByTestId("upset-bar").map((b) => Number(b.getAttribute("height")));
    expect(heights[0]).toBeGreaterThan(heights[1]!);
    expect(heights[1]! / heights[0]!).toBeCloseTo(190 / 1400);
    // Set bars in their palette colours.
    expect(screen.getAllByTestId("upset-set-bar").map((b) => b.getAttribute("fill"))).toEqual([
      "rgb(230,159,0)",
      "rgb(86,180,233)",
    ]);
  });

  it("labels each column with its counts, for tooltips and screen readers", () => {
    renderPlot();
    const cols = screen.getAllByRole("button");
    expect(cols.map((c) => c.getAttribute("aria-label"))).toEqual(COLS.map((c) => c.title));
    expect(screen.getByText("1,400")).toBeInTheDocument();
    expect(screen.getByText("CD14 Mono: 1,800 genes")).toBeInTheDocument(); // set tooltip
    // A long set name is shortened on the plot, whole in its tooltip.
    expect(screen.getByText("CD14 Mono: STIM vs CTRL (W…")).toBeInTheDocument();
  });

  it("selects a column by click or keyboard, and marks the selected one", () => {
    const onSelect = renderPlot(1);
    const cols = screen.getAllByRole("button");
    expect(cols.map((c) => c.getAttribute("aria-pressed"))).toEqual(["false", "true", "false"]);
    fireEvent.click(cols[2]!);
    expect(onSelect).toHaveBeenLastCalledWith(2);
    fireEvent.keyDown(cols[0]!, { key: "Enter" });
    expect(onSelect).toHaveBeenLastCalledWith(0);
    fireEvent.keyDown(cols[0]!, { key: "a" });
    expect(onSelect).toHaveBeenCalledTimes(2);
  });
});
