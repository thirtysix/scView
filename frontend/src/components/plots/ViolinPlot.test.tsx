import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { plotProps } = vi.hoisted(() => ({ plotProps: { current: null as Record<string, unknown> | null } }));
// Plotly needs a real canvas; the traces and layout it is handed are what matter.
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plotProps.current = props;
    return <div data-testid="plot" />;
  },
}));

import { ViolinPlot } from "./ViolinPlot";
import { pseudoLog } from "@/lib/logAxis";

type Trace = { type: string; y: number[]; x?: string[]; customdata?: string[]; hoverinfo?: string };
type Axis = { title?: { text: string }; tickmode?: string; tickvals?: number[]; ticktext?: string[] };
const traces = () => plotProps.current!.data as Trace[];
const yaxis = () => (plotProps.current!.layout as { yaxis: Axis }).yaxis;

const DATA = { B: [0, 0, 1, 3, 1000], NK: [0, 2, 5, 9, 20] };

beforeEach(() => {
  plotProps.current = null;
});

describe("ViolinPlot log scale", () => {
  it("draws the values as they are by default", () => {
    render(<ViolinPlot data={DATA} yLabel="Counts" />);
    expect(traces().map((t) => t.type)).toEqual(["violin", "violin"]);
    expect(traces()[0]!.y).toEqual(DATA.B);
    expect(yaxis().title!.text).toBe("Counts");
    expect(yaxis().tickmode).toBeUndefined();
  });

  it("switches to a log axis that keeps the zeros, with ticks and hover in the original units", () => {
    render(<ViolinPlot data={DATA} yLabel="Counts" />);
    fireEvent.click(screen.getByLabelText("log scale"));
    const [b, nk, stats] = traces();
    expect(b!.y).toEqual(DATA.B.map(pseudoLog));
    expect(b!.y[0]).toBe(0); // a zero stays on the plot
    expect(b!.hoverinfo).toBe("skip");
    expect(nk!.type).toBe("violin");
    expect(yaxis().title!.text).toBe("Counts (log scale)");
    expect(yaxis().tickmode).toBe("array");
    expect(yaxis().ticktext).toEqual(["0", "1", "10", "100", "1k"]);
    expect(yaxis().tickvals![2]).toBeCloseTo(pseudoLog(10)); // "10" sits at log10(11)
    // Hover markers at each group's range and quartiles, labelled in counts.
    expect(stats!.type).toBe("scatter");
    expect(stats!.x).toEqual(["B", "B", "B", "B", "B", "NK", "NK", "NK", "NK", "NK"]);
    expect(stats!.customdata!.slice(0, 5)).toEqual([
      "max 1,000",
      "upper quartile 3",
      "median 1",
      "lower quartile 0",
      "min 0",
    ]);
    expect(stats!.y[0]).toBeCloseTo(pseudoLog(1000));

    fireEvent.click(screen.getByLabelText("log scale"));
    expect(traces()).toHaveLength(2);
    expect(traces()[0]!.y).toEqual(DATA.B);
  });
});
