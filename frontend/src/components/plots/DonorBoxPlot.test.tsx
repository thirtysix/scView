import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { DonorPoint } from "@/lib/donorPlot";

const { plotProps } = vi.hoisted(() => ({ plotProps: vi.fn() }));
vi.mock("@/components/plots/Plot", () => ({
  default: (p: { data: unknown[] }) => {
    plotProps(p);
    return <div data-testid="plot" />;
  },
}));

import { DonorBoxPlot } from "./DonorBoxPlot";

const pt = (group: string, condition: string, sample: string, value: number): DonorPoint => ({
  group, condition, sample, n_cells: 50, value,
});
const donors = ["d1", "d2", "d3", "d4", "d5"];
const PAIRED_POINTS: DonorPoint[] = [
  ...donors.map((d, i) => pt("B", "CTRL", d, i)),
  ...donors.map((d, i) => pt("B", "STIM", d, i + 2)),
  ...donors.slice(0, 3).map((d, i) => pt("pDC", "CTRL", d, i / 10)),
  ...donors.slice(0, 2).map((d, i) => pt("pDC", "STIM", d, i / 10 + 0.3)),
];
const base = {
  groups: ["B", "pDC"],
  conditions: ["CTRL", "STIM"],
  sampleKey: "Donor",
  colorOf: (k: string, kind: string) => `${kind}:${k}`,
  valueLabel: "ISG15",
  format: (v: number) => v.toFixed(2),
  plotName: "test",
  unit: "log2 CPM",
};
const TESTS = {
  B: { n_pairs: 5, mean_diff: 2, wilcoxon_p: 0.0625, ttest_p: 0.0001, caution: null },
  pDC: { n_pairs: 2, mean_diff: 0.3, wilcoxon_p: 0.5, ttest_p: 0.2, caution: "2 pairs: read these p-values as descriptive." },
};

/** Traces drawn on the last render. */
const traces = () => (plotProps.mock.lastCall![0] as { data: Record<string, unknown>[] }).data;

describe("DonorBoxPlot", () => {
  beforeEach(() => plotProps.mockClear());

  it("draws a line per donor and a mean per group when the design is paired", () => {
    render(
      <DonorBoxPlot {...base} points={PAIRED_POINTS} design={{ paired: true, n_pairs: 5, conditions: ["CTRL", "STIM"] }} pairedTests={TESTS} />,
    );
    const data = traces();
    const lines = data.filter((t) => t.legendgroup === "donors");
    expect(lines).toHaveLength(5 + 3); // B: 5 donors, pDC: 3 (two with both values, one with CTRL only)
    expect(data.filter((t) => t.name === "mean")).toHaveLength(2);
    const d2 = lines.find((t) => t.name === "d2" && t.xaxis === "x")!;
    expect(d2.x).toEqual(["CTRL", "STIM"]);
    expect(d2.y).toEqual([1, 3]);
    expect((d2.text as string[])[0]).toBe("d2: CTRL 1.00, STIM 3.00 (50 cells here)");
    const mean = data.find((t) => t.name === "mean" && t.xaxis === "x")!;
    expect(mean.y).toEqual([2, 4]);
    expect((mean.marker as { color: string[] }).color).toEqual(["condition:CTRL", "condition:STIM"]);
    expect(data.some((t) => t.type === "box")).toBe(false);
    const caption = screen.getByTestId("donor-caption").textContent!;
    expect(caption).toContain("One line per Donor (n = 5 donors, each in CTRL and STIM)");
    const table = screen.getByTestId("paired-tests").textContent!;
    expect(table).toContain("+2.00 log2 CPM");
    expect(table).toContain("0.0625");
    expect(table).toContain("1.00e-4");
    expect(table).toContain("0.5000*");
    expect(screen.getByTestId("paired-caution").textContent).toContain("Under 5 pairs");
    // The group titles carry the Wilcoxon p (and n when few).
    const titles = ((plotProps.mock.lastCall![0] as { layout: { annotations: { text: string }[] } }).layout.annotations).map((a) => a.text);
    expect(titles).toEqual(["B · p = 0.06", "pDC · p = 0.50 (n = 2)"]);
  });

  it("switches to boxes on the toggle and back", () => {
    render(
      <DonorBoxPlot {...base} points={PAIRED_POINTS} design={{ paired: true, n_pairs: 5, conditions: ["CTRL", "STIM"] }} pairedTests={TESTS} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Boxes" }));
    expect(traces().every((t) => t.type === "box")).toBe(true);
    expect(screen.getByTestId("donor-caption").textContent).toContain("One point per Donor (n = 5: CTRL 5, STIM 5)");
    expect(screen.queryByTestId("paired-tests")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Paired lines" }));
    expect(traces().some((t) => t.legendgroup === "donors")).toBe(true);
  });

  it("keeps boxes, no toggle and no tests for an unpaired design", () => {
    render(<DonorBoxPlot {...base} points={PAIRED_POINTS} design={{ paired: false, n_pairs: 0, conditions: ["CTRL", "STIM"] }} />);
    expect(traces().every((t) => t.type === "box")).toBe(true);
    expect(screen.queryByRole("button", { name: "Boxes" })).toBeNull();
    expect(screen.queryByTestId("paired-tests")).toBeNull();
    expect(screen.getByTestId("donor-caption").textContent).toContain("One point per Donor (n = 5: CTRL 5, STIM 5)");
  });

  it("draws trajectories over three conditions with no test", () => {
    const times = ["T0", "T1", "T2"];
    const points = times.flatMap((t, i) => donors.slice(0, 3).map((d, j) => pt("B", t, d, i + j)));
    render(
      <DonorBoxPlot {...base} conditions={times} points={points} design={{ paired: true, n_pairs: 3, conditions: times }} pairedTests={{}} />,
    );
    const lines = traces().filter((t) => t.legendgroup === "donors");
    expect(lines).toHaveLength(3);
    expect(lines[0]!.x).toEqual(times);
    const caption = screen.getByTestId("donor-caption").textContent!;
    expect(caption).toContain("n = 3 donors, each in T0, T1 and T2");
    expect(caption).toContain("there is no test");
    expect(screen.queryByTestId("paired-tests")).toBeNull();
  });
});
